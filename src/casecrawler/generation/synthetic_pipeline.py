from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import uuid4

from casecrawler.config import get_config
from casecrawler.generation.clinical_text_models import (
    resolve_clinical_text_model_profile,
)
from casecrawler.generation.imaging_generator import ImagingGenerator
from casecrawler.generation.imaging_models import resolve_imaging_model_profile
from casecrawler.generation.modality_plan import ModalityPlanner
from casecrawler.generation.recipes import apply_generation_recipe
from casecrawler.generation.retriever import Retriever
from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.generation.text_generator import TextGenerator
from casecrawler.generation.timeseries_generator import TimeSeriesGenerator
from casecrawler.generation.timeseries_models import (
    resolve_time_series_model_profile,
)
from casecrawler.llm.factory import get_provider
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import GroundingBundle, Modality, SyntheticRecord
from casecrawler.validation.synthetic_validator import SyntheticValidator

logger = logging.getLogger(__name__)


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
        retriever: Retriever | None = None,
    ) -> None:
        config = get_config()
        self._config = config
        self._retriever = retriever
        self._structured_generator = structured_generator or StructuredGenerator()
        self._text_generator = text_generator or _text_generator_from_config(
            config.synthetic.clinical_text_backend,
            config.llm.provider,
            config.llm.model,
            config.llm.ollama_base_url,
            config.synthetic.clinical_text_command
            or _clinical_text_profile_command(config.synthetic.clinical_text_model_profile),
            config.synthetic.clinical_text_noise_profile,
        )
        self._time_series_generator = time_series_generator or _time_series_generator_from_config(
            config.synthetic.time_series_backend,
            config.synthetic.time_series_command,
        )
        self._imaging_generator = imaging_generator or ImagingGenerator(
            diffusers_model_id=config.synthetic.diffusers_model_id,
            imaging_model_profile=config.synthetic.imaging_model_profile,
            external_command=config.synthetic.imaging_command,
        )
        self._modality_planner = modality_planner or ModalityPlanner()
        self._validator = validator
        self._image_output_dir = image_output_dir or config.synthetic.image_output_dir
        self._image_backend = image_backend or config.synthetic.imaging_backend

    async def generate(self, req: GenerationRequest) -> dict:
        req = apply_generation_recipe(req)
        dataset_id = f"ds-{uuid4()}"
        plan = self._modality_planner.build(req)
        text_generator = self._text_generator_for(req)
        imaging_generator = self._imaging_generator_for(req)
        time_series_generator = self._time_series_generator_for(req)
        grounding_bundle = self._fetch_grounding_for_topic(req)
        records = []
        approved = 0
        for index in range(req.count):
            record = await self._generate_validated_record(
                dataset_id=dataset_id,
                req=req,
                logical_index=index,
                plan=plan,
                text_generator=text_generator,
                imaging_generator=imaging_generator,
                time_series_generator=time_series_generator,
                grounding_bundle=grounding_bundle,
            )
            records.append(record)
            if record.validation and record.validation.approved:
                approved += 1
        return {
            "dataset_id": dataset_id,
            "generated": len(records),
            "approved": approved,
            "plan": plan,
            "records": records,
        }

    async def _generate_validated_record(
        self,
        *,
        dataset_id: str,
        req: GenerationRequest,
        logical_index: int,
        plan,
        text_generator,
        imaging_generator,
        time_series_generator,
        grounding_bundle: GroundingBundle | None = None,
    ) -> SyntheticRecord:
        attempts = req.max_validation_retries + 1
        last_record: SyntheticRecord | None = None
        for attempt in range(attempts):
            generation_index = logical_index + (attempt * req.count)
            record_req = _request_for_record_index(req, generation_index)
            record = self._structured_generator.generate(
                dataset_id=dataset_id,
                req=record_req,
                index=generation_index,
            )
            if Modality.TIME_SERIES in plan.modalities:
                record = time_series_generator.add_time_series(
                    record,
                    channels=plan.time_series_channels,
                )
                record = record.model_copy(
                    update={
                        "metadata": _with_time_series_generation_metadata(
                            record.metadata,
                            req,
                            self._config.synthetic.time_series_model_profile,
                        )
                    }
                )
            if Modality.IMAGING in plan.modalities:
                images = [
                    self._generate_image_asset(request, req, imaging_generator)
                    for request in _imaging_requests_for_record(
                        record,
                        plan.imaging_views or ["medical_image"],
                    )
                ]
                record = record.model_copy(
                    update={
                        "imaging": [*record.imaging, *images],
                        "metadata": _with_imaging_generation_metadata(
                            record.metadata,
                            req,
                            self._config.synthetic.imaging_model_profile,
                        ),
                    }
                )
            if Modality.CLINICAL_TEXT in plan.modalities:
                record = await text_generator.add_documents_async(record)
                record = record.model_copy(
                    update={
                        "metadata": _with_clinical_text_generation_metadata(
                            record.metadata,
                            req,
                            self._config.synthetic.clinical_text_backend,
                            self._config.llm.provider,
                            self._config.llm.model,
                            self._config.synthetic.clinical_text_model_profile,
                            self._config.synthetic.clinical_text_command,
                        )
                    }
                )
            if grounding_bundle is not None:
                record = record.model_copy(
                    update={
                        "metadata": {
                            **record.metadata,
                            "grounding": grounding_bundle.model_dump(),
                        }
                    }
                )
            validator = self._validator_for(record_req)
            validation = validator.validate(record)
            metadata = {
                **record.metadata,
                "validation_retry": {
                    "attempt": attempt,
                    "max_retries": req.max_validation_retries,
                    "regenerated": attempt > 0,
                },
            }
            if validation.modality_alignment_score is not None:
                metadata["image_validator_policy"] = validator.image_validator_policy()
            record = record.model_copy(
                update={"metadata": metadata, "validation": validation}
            )
            last_record = record
            if validation.approved:
                return record
        if last_record is None:
            raise RuntimeError("Synthetic pipeline produced no validation attempts.")
        return last_record

    def _validator_for(self, req: GenerationRequest) -> SyntheticValidator:
        if self._validator is not None:
            return self._validator
        return SyntheticValidator(
            threshold=req.validation_threshold,
            require_grounding=self._config.synthetic.grounding.enabled,
        )

    def _fetch_grounding_for_topic(
        self, req: GenerationRequest
    ) -> GroundingBundle | None:
        """Resolve a single grounding bundle for the request's topic.

        Returns ``None`` when grounding is disabled. Otherwise either
        delegates to the injected retriever, or builds a default one over the
        configured Chroma store. When retrieval returns nothing, the
        ``synthetic.grounding.fallback`` setting controls whether we proceed
        with an empty (but flagged) bundle or refuse to generate.
        """

        cfg = self._config.synthetic.grounding
        if not cfg.enabled:
            return None
        retriever = self._retriever or self._default_retriever()
        if retriever is None:
            if cfg.fallback == "fail":
                raise RuntimeError(
                    "synthetic.grounding.enabled=true but no retriever is "
                    "available. Configure storage.chroma_persist_dir and "
                    "ingest a knowledge base, or set "
                    "synthetic.grounding.fallback='template'."
                )
            return self._fallback_bundle(req.topic, "no_retriever_configured")
        try:
            bundle = retriever.fetch_grounding(
                topic=req.topic,
                modalities=[m.value for m in req.modalities],
                k=cfg.k,
            )
        except Exception:
            logger.exception(
                "Grounding retrieval failed for topic %r", req.topic
            )
            if cfg.fallback == "fail":
                raise
            return self._fallback_bundle(req.topic, "retrieval_error")
        if not bundle.citations:
            if cfg.fallback == "fail":
                raise RuntimeError(
                    f"Grounding retrieval for topic {req.topic!r} returned no "
                    "citations and synthetic.grounding.fallback='fail'."
                )
            return bundle.model_copy(
                update={
                    "fallback_used": True,
                    "fallback_reason": "no_chunks_in_index",
                }
            )
        return bundle

    def _default_retriever(self) -> Retriever | None:
        """Construct a Retriever lazily over the configured Chroma store.

        Returns None if Chroma is unavailable (e.g. in a CI environment
        without the persist dir). Errors are logged, not raised, so the
        ``fallback="template"`` path stays clean.
        """

        try:
            from casecrawler.pipeline.store import Store
        except Exception:
            logger.exception("Failed to import casecrawler.pipeline.store")
            return None
        try:
            store = Store(chroma_dir=self._config.storage.chroma_persist_dir)
            return Retriever(store)
        except Exception:
            logger.exception(
                "Failed to construct default Retriever; grounding will fall back."
            )
            return None

    @staticmethod
    def _fallback_bundle(topic: str, reason: str) -> GroundingBundle:
        from datetime import datetime

        return GroundingBundle(
            topic=topic,
            retrieved_at=datetime.now().isoformat(),
            citations=[],
            fallback_used=True,
            fallback_reason=reason,
        )

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
        if image_backend == "external":
            if not req.imaging_command:
                raise ValueError(
                    "External synthetic imaging backend requires imaging_command."
                )
            return imaging_generator.generate_external(
                output_dir=self._image_output_dir,
                prompt=request.prompt,
                modality=request.modality,
                body_region=request.body_region,
            )
        raise ValueError(f"Unknown synthetic imaging backend: {image_backend}")

    def _imaging_generator_for(self, req: GenerationRequest) -> ImagingGenerator:
        if not (req.imaging_model_profile or req.diffusers_model_id or req.imaging_command):
            return self._imaging_generator
        return ImagingGenerator(
            diffusers_model_id=(
                req.diffusers_model_id or self._config.synthetic.diffusers_model_id
            ),
            imaging_model_profile=(
                req.imaging_model_profile
                or self._config.synthetic.imaging_model_profile
            ),
            external_command=req.imaging_command or self._config.synthetic.imaging_command,
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
            or req.clinical_text_noise_profile
            or req.llm_provider
            or req.llm_model
            or req.ollama_base_url
            or req.clinical_text_model_profile
            or req.clinical_text_command
        ):
            return self._text_generator
        if req.clinical_text_model_profile:
            resolve_clinical_text_model_profile(req.clinical_text_model_profile)
        backend = req.clinical_text_backend or (
            "llm"
            if (req.llm_provider or req.llm_model)
            else (
                "external"
                if (req.clinical_text_model_profile or req.clinical_text_command)
                else self._config.synthetic.clinical_text_backend
            )
        )
        profile_command = _clinical_text_profile_command(req.clinical_text_model_profile)
        return _text_generator_from_config(
            backend,
            req.llm_provider or self._config.llm.provider,
            req.llm_model or self._config.llm.model,
            req.ollama_base_url or self._config.llm.ollama_base_url,
            req.clinical_text_command
            or profile_command
            or self._config.synthetic.clinical_text_command,
            req.clinical_text_noise_profile
            or self._config.synthetic.clinical_text_noise_profile,
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
    command: list[str] | None,
    noise_profile: str = "standard",
) -> TextGenerator:
    if backend == "deterministic":
        return TextGenerator(noise_profile=noise_profile)
    if backend == "llm":
        provider = get_provider(provider_name, model, base_url=ollama_base_url)
        return TextGenerator(provider=provider, noise_profile=noise_profile)
    if backend == "external":
        if not command:
            raise ValueError(
                "synthetic.clinical_text_command is required when "
                "clinical_text_backend is 'external'."
            )
        return TextGenerator(external_command=command, noise_profile=noise_profile)
    raise ValueError(f"Unknown synthetic clinical text backend: {backend}")


def _clinical_text_profile_command(profile_name: str | None) -> list[str] | None:
    profile = resolve_clinical_text_model_profile(profile_name)
    if profile is None:
        return None
    return profile.command_template


def _with_imaging_generation_metadata(
    metadata: dict,
    req: GenerationRequest,
    configured_profile: str | None,
) -> dict:
    profile_name = req.imaging_model_profile or configured_profile
    if not profile_name:
        return metadata
    profile = resolve_imaging_model_profile(profile_name)
    if profile is None:
        return metadata
    return {
        **metadata,
        "imaging_model_policy": {
            "profile": profile.name,
            "model_id": profile.model_id,
            "license": profile.license,
            "gated": profile.gated,
            "use_policy": profile.use_policy,
        },
    }


def _with_time_series_generation_metadata(
    metadata: dict,
    req: GenerationRequest,
    configured_profile: str | None,
) -> dict:
    profile_name = req.time_series_model_profile or configured_profile
    if not profile_name:
        return metadata
    profile = resolve_time_series_model_profile(profile_name)
    if profile is None:
        return metadata
    return {
        **metadata,
        "time_series_model_policy": {
            "profile": profile.name,
            "model_id": profile.model_id,
            "license": profile.license,
            "gated": profile.gated,
            "use_policy": profile.use_policy,
        },
    }


def _with_clinical_text_generation_metadata(
    metadata: dict,
    req: GenerationRequest,
    configured_backend: str,
    configured_provider: str,
    configured_model: str,
    configured_profile: str | None,
    configured_command: list[str] | None,
) -> dict:
    backend = req.clinical_text_backend or (
        "llm"
        if (req.llm_provider or req.llm_model)
        else (
            "external"
            if (req.clinical_text_model_profile or req.clinical_text_command)
            else configured_backend
        )
    )
    if backend == "deterministic":
        return {
            **metadata,
            "clinical_text_model_policy": {
                "backend": "deterministic",
                "provider": "casecrawler",
                "model_id": "casecrawler-template-clinical-documents",
                "license": "casecrawler",
                "gated": False,
                "use_policy": "deterministic_synthetic_templates_validate_outputs",
            },
        }
    if backend not in {"llm", "external"}:
        return metadata
    if backend == "external":
        profile_name = req.clinical_text_model_profile or configured_profile
        profile = resolve_clinical_text_model_profile(profile_name)
        return {
            **metadata,
            "clinical_text_model_policy": {
                "backend": "external",
                "profile": profile.name if profile else None,
                "model_id": profile.model_id if profile else None,
                "command": req.clinical_text_command
                or (profile.command_template if profile else None)
                or configured_command,
                "license": profile.license if profile else "external_command_terms",
                "gated": profile.gated if profile else False,
                "use_policy": (
                    profile.use_policy
                    if profile
                    else "wrap_external_clinical_text_generator_validate_outputs"
                ),
            },
        }
    provider = req.llm_provider or configured_provider
    model = req.llm_model or configured_model
    return {
        **metadata,
        "clinical_text_model_policy": {
            "backend": "llm",
            "provider": provider,
            "model_id": model,
            "license": "provider_terms",
            "gated": False,
            "use_policy": "synthetic_clinical_text_review_outputs_before_release",
        },
    }


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
    patient_context = f"{record.patient.age} year old {record.patient.sex}"
    requests = []
    for view in views:
        normalized_view = view.replace("_", " ")
        requests.append(
            ImagingRequest(
                prompt=" ".join(
                    part
                    for part in [
                        patient_context,
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


def _request_for_record_index(req: GenerationRequest, index: int) -> GenerationRequest:
    topics = _topic_mix(req.cohort_constraints)
    if not topics:
        return req
    return req.model_copy(update={"topic": topics[index % len(topics)]})


def _topic_mix(value) -> list[str]:
    weights = {}
    if isinstance(value, dict):
        if "topic_mix" not in value:
            return []
        weights = value.get("topic_mix_weights", {})
        value = value.get("topic_mix")
    if value is None:
        return []
    if isinstance(value, str):
        topics = []
        for item in value.split(","):
            topics.extend(_topic_mix_entry(item, weights))
    elif isinstance(value, list):
        topics = []
        for item in value:
            topics.extend(_topic_mix_entry(item, weights))
    else:
        raise ValueError("cohort_constraints.topic_mix must be a string or list.")
    if not topics:
        raise ValueError("cohort_constraints.topic_mix must contain at least one topic.")
    return topics


def _topic_mix_entry(value, weights) -> list[str]:
    weight = 1
    if isinstance(value, dict):
        topic = str(value.get("topic", "")).strip()
        weight = _topic_mix_weight(value.get("weight", _topic_weight(topic, weights)))
    else:
        raw = str(value).strip()
        topic, separator, raw_weight = raw.rpartition(":")
        if separator and raw_weight.strip():
            topic = topic.strip()
            weight = _topic_mix_weight(raw_weight.strip())
        else:
            topic = raw
            weight = _topic_mix_weight(_topic_weight(topic, weights))
    if not topic:
        return []
    return [topic] * weight


def _topic_weight(topic: str, weights) -> int:
    if not topic or weights is None:
        return 1
    if not isinstance(weights, dict):
        raise ValueError("cohort_constraints.topic_mix_weights must be a mapping.")
    return weights.get(topic, 1)


def _topic_mix_weight(value) -> int:
    if isinstance(value, bool):
        raise ValueError("cohort_constraints.topic_mix weights must be positive integers.")
    try:
        weight = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "cohort_constraints.topic_mix weights must be positive integers."
        ) from exc
    if weight < 1:
        raise ValueError("cohort_constraints.topic_mix weights must be positive integers.")
    return weight
