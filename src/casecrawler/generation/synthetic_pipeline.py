from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from casecrawler.config import get_config
from casecrawler.generation.imaging_generator import ImagingGenerator
from casecrawler.generation.modality_plan import ModalityPlanner
from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.generation.text_generator import TextGenerator
from casecrawler.generation.timeseries_generator import TimeSeriesGenerator
from casecrawler.generation.timeseries_models import resolve_time_series_model_profile
from casecrawler.llm.factory import get_provider
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Modality, SyntheticRecord
from casecrawler.validation.synthetic_validator import SyntheticValidator


@dataclass(frozen=True)
class ImagingRequest:
    prompt: str
    modality: str
    body_region: str


class SyntheticPipeline:
    def __init__(
        self,
        structured_generator: StructuredGenerator | None = None,
        text_generator: TextGenerator | None = None,
        time_series_generator: TimeSeriesGenerator | None = None,
        imaging_generator: ImagingGenerator | None = None,
        modality_planner: ModalityPlanner | None = None,
        validator: SyntheticValidator | None = None,
        image_output_dir: str | None = None,
        image_backend: str | None = None,
    ) -> None:
        config = get_config()
        self._config = config
        self._structured_generator = structured_generator or StructuredGenerator()
        self._text_generator = text_generator or _text_generator_from_config(
            config.synthetic.clinical_text_backend,
            config.llm.provider,
            config.llm.model,
            config.llm.ollama_base_url,
        )
        self._time_series_generator = time_series_generator or _time_series_generator_from_config(
            config.synthetic.time_series_backend,
            config.synthetic.time_series_command,
        )
        self._imaging_generator = imaging_generator or ImagingGenerator(
            diffusers_model_id=config.synthetic.diffusers_model_id,
            imaging_model_profile=config.synthetic.imaging_model_profile,
        )
        self._modality_planner = modality_planner or ModalityPlanner()
        self._validator = validator
        self._image_output_dir = image_output_dir or config.synthetic.image_output_dir
        self._image_backend = image_backend or config.synthetic.imaging_backend

    async def generate(self, req: GenerationRequest) -> dict:
        dataset_id = f"ds-{uuid4()}"
        plan = self._modality_planner.build(req)
        text_generator = self._text_generator_for(req)
        imaging_generator = self._imaging_generator_for(req)
        time_series_generator = self._time_series_generator_for(req)
        records = []
        approved = 0
        for index in range(req.count):
            record = self._structured_generator.generate(
                dataset_id=dataset_id,
                req=req,
                index=index,
            )
            if Modality.TIME_SERIES in plan.modalities:
                record = time_series_generator.add_time_series(
                    record,
                    channels=plan.time_series_channels,
                )
            if Modality.IMAGING in plan.modalities:
                images = [
                    self._generate_image_asset(request, req, imaging_generator)
                    for request in _imaging_requests_for_record(
                        record,
                        plan.imaging_views or ["medical_image"],
                    )
                ]
                record = record.model_copy(update={"imaging": [*record.imaging, *images]})
            if Modality.CLINICAL_TEXT in plan.modalities:
                record = await text_generator.add_documents_async(record)
            validation = self._validator_for(req).validate(record)
            record = record.model_copy(update={"validation": validation})
            records.append(record)
            if validation.approved:
                approved += 1
        return {
            "dataset_id": dataset_id,
            "generated": len(records),
            "approved": approved,
            "plan": plan,
            "records": records,
        }

    def _validator_for(self, req: GenerationRequest) -> SyntheticValidator:
        if self._validator is not None:
            return self._validator
        return SyntheticValidator(threshold=req.validation_threshold)

    def _generate_image_asset(
        self,
        request: ImagingRequest,
        req: GenerationRequest,
        imaging_generator: ImagingGenerator,
    ):
        image_backend = req.imaging_backend or self._image_backend
        if image_backend == "diffusers":
            return imaging_generator.generate_diffusers(
                output_dir=self._image_output_dir,
                prompt=request.prompt,
                modality=request.modality,
                body_region=request.body_region,
            )
        if image_backend == "placeholder":
            return imaging_generator.generate_placeholder(
                output_dir=self._image_output_dir,
                prompt=request.prompt,
                modality=request.modality,
                body_region=request.body_region,
            )
        raise ValueError(f"Unknown synthetic imaging backend: {image_backend}")

    def _imaging_generator_for(self, req: GenerationRequest) -> ImagingGenerator:
        if not (req.imaging_model_profile or req.diffusers_model_id):
            return self._imaging_generator
        return ImagingGenerator(
            diffusers_model_id=(
                req.diffusers_model_id or self._config.synthetic.diffusers_model_id
            ),
            imaging_model_profile=(
                req.imaging_model_profile
                or self._config.synthetic.imaging_model_profile
            ),
        )

    def _time_series_generator_for(self, req: GenerationRequest) -> TimeSeriesGenerator:
        if not (
            req.time_series_backend
            or req.time_series_model_profile
            or req.time_series_command
        ):
            return self._time_series_generator
        if req.time_series_model_profile:
            resolve_time_series_model_profile(req.time_series_model_profile)
        backend = req.time_series_backend or (
            "external"
            if (req.time_series_model_profile or req.time_series_command)
            else self._config.synthetic.time_series_backend
        )
        command = req.time_series_command or self._config.synthetic.time_series_command
        return _time_series_generator_from_config(backend, command)

    def _text_generator_for(self, req: GenerationRequest) -> TextGenerator:
        if not (
            req.clinical_text_backend
            or req.llm_provider
            or req.llm_model
            or req.ollama_base_url
        ):
            return self._text_generator
        backend = req.clinical_text_backend or (
            "llm"
            if (req.llm_provider or req.llm_model)
            else self._config.synthetic.clinical_text_backend
        )
        return _text_generator_from_config(
            backend,
            req.llm_provider or self._config.llm.provider,
            req.llm_model or self._config.llm.model,
            req.ollama_base_url or self._config.llm.ollama_base_url,
        )


def _time_series_generator_from_config(
    backend: str,
    command: list[str] | None,
) -> TimeSeriesGenerator:
    if backend == "deterministic":
        return TimeSeriesGenerator()
    if backend == "external":
        if not command:
            raise ValueError(
                "synthetic.time_series_command is required when "
                "time_series_backend is 'external'."
            )
        return TimeSeriesGenerator(external_command=command)
    raise ValueError(f"Unknown synthetic time-series backend: {backend}")


def _text_generator_from_config(
    backend: str,
    provider_name: str,
    model: str,
    ollama_base_url: str,
) -> TextGenerator:
    if backend == "deterministic":
        return TextGenerator()
    if backend == "llm":
        provider = get_provider(provider_name, model, base_url=ollama_base_url)
        return TextGenerator(provider=provider)
    raise ValueError(f"Unknown synthetic clinical text backend: {backend}")


def _imaging_requests_for_record(
    record: SyntheticRecord,
    views: list[str],
) -> list[ImagingRequest]:
    modality, body_region, topic_prompt = _topic_imaging_spec(record.topic)
    diagnosis_terms = " ".join(
        diagnosis.display
        for encounter in record.encounters
        for diagnosis in encounter.diagnoses
    )
    requests = []
    for view in views:
        normalized_view = view.replace("_", " ")
        requests.append(
            ImagingRequest(
                prompt=" ".join(
                    part
                    for part in [
                        normalized_view,
                        topic_prompt,
                        diagnosis_terms,
                    ]
                    if part
                ),
                modality=modality,
                body_region=body_region,
            )
        )
    return requests


def _topic_imaging_spec(topic: str) -> tuple[str, str, str]:
    normalized = topic.lower().replace("-", " ").replace("_", " ")
    if "heart failure" in normalized or "edema" in normalized:
        return "XR", "chest", "pulmonary edema cardiomegaly small pleural effusion"
    if (
        "asthma" in normalized
        or "status asthmaticus" in normalized
        or "bronchospasm" in normalized
    ):
        return "XR", "chest", "hyperinflation bronchial wall thickening asthma"
    if "pneumonia" in normalized:
        return "XR", "chest", "right lower lobe opacity pneumonia"
    if "pulmonary embolism" in normalized or "pulmonary embolus" in normalized:
        return "CTA", "chest", "pulmonary arterial filling defect pulmonary embolism"
    if "pancreatitis" in normalized or "epigastric pain" in normalized:
        return (
            "CT",
            "abdomen",
            "acute pancreatitis peripancreatic inflammation fat stranding",
        )
    if (
        "appendicitis" in normalized
        or "right lower quadrant" in normalized
        or "rlq pain" in normalized
    ):
        return (
            "CT",
            "abdomen",
            "dilated appendix appendiceal wall thickening fat stranding",
        )
    if (
        "pyelonephritis" in normalized
        or "flank pain" in normalized
        or "urinary tract infection" in normalized
    ):
        return (
            "CT",
            "abdomen",
            "striated nephrogram perinephric stranding pyelonephritis",
        )
    if "meningitis" in normalized or "photophobia" in normalized:
        return "CT", "head", "noncontrast head CT no acute hemorrhage meningitis screen"
    if (
        "seizure" in normalized
        or "status epilepticus" in normalized
        or "postictal" in normalized
    ):
        return (
            "CT",
            "head",
            "noncontrast head CT no acute hemorrhage postictal seizure evaluation",
        )
    if "sepsis" in normalized or "infection" in normalized:
        return "XR", "chest", "portable chest x-ray possible lower lobe opacity"
    if "stroke" in normalized:
        return "CT", "head", "noncontrast head CT no acute hemorrhage"
    if "acute kidney injury" in normalized or "renal failure" in normalized:
        return "US", "abdomen", "renal ultrasound hydronephrosis evaluation"
    if "gi bleed" in normalized or "gastrointestinal bleed" in normalized:
        return "CT", "abdomen", "contrast CT abdomen active gastrointestinal bleeding"
    return "XR", "chest", topic
