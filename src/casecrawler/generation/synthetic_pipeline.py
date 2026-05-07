from __future__ import annotations

from uuid import uuid4

from casecrawler.config import get_config
from casecrawler.generation.imaging_generator import ImagingGenerator
from casecrawler.generation.modality_plan import ModalityPlanner
from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.generation.text_generator import TextGenerator
from casecrawler.generation.timeseries_generator import TimeSeriesGenerator
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import Modality
from casecrawler.validation.synthetic_validator import SyntheticValidator


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
        self._structured_generator = structured_generator or StructuredGenerator()
        self._text_generator = text_generator or TextGenerator()
        self._time_series_generator = time_series_generator or TimeSeriesGenerator()
        self._imaging_generator = imaging_generator or ImagingGenerator(
            diffusers_model_id=config.synthetic.diffusers_model_id,
            imaging_model_profile=config.synthetic.imaging_model_profile,
        )
        self._modality_planner = modality_planner or ModalityPlanner()
        self._validator = validator or SyntheticValidator()
        self._image_output_dir = image_output_dir or config.synthetic.image_output_dir
        self._image_backend = image_backend or config.synthetic.imaging_backend

    async def generate(self, req: GenerationRequest) -> dict:
        dataset_id = f"ds-{uuid4()}"
        plan = self._modality_planner.build(req)
        records = []
        approved = 0
        for index in range(req.count):
            record = self._structured_generator.generate(
                dataset_id=dataset_id,
                req=req,
                index=index,
            )
            if Modality.CLINICAL_TEXT in plan.modalities:
                record = self._text_generator.add_documents(record)
            if Modality.TIME_SERIES in plan.modalities:
                record = self._time_series_generator.add_time_series(
                    record,
                    channels=plan.time_series_channels,
                )
            if Modality.IMAGING in plan.modalities:
                images = [
                    self._generate_image_asset(prompt=f"{req.topic} {view}")
                    for view in (plan.imaging_views or ["medical_image"])
                ]
                record = record.model_copy(update={"imaging": [*record.imaging, *images]})
            validation = self._validator.validate(record)
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

    def _generate_image_asset(self, prompt: str):
        if self._image_backend == "diffusers":
            return self._imaging_generator.generate_diffusers(
                output_dir=self._image_output_dir,
                prompt=prompt,
            )
        if self._image_backend == "placeholder":
            return self._imaging_generator.generate_placeholder(
                output_dir=self._image_output_dir,
                prompt=prompt,
            )
        raise ValueError(f"Unknown synthetic imaging backend: {self._image_backend}")
