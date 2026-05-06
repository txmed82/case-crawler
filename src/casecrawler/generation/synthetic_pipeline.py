from __future__ import annotations

from uuid import uuid4

from casecrawler.generation.structured_generator import StructuredGenerator
from casecrawler.generation.text_generator import TextGenerator
from casecrawler.models.dataset import GenerationRequest
from casecrawler.validation.synthetic_validator import SyntheticValidator


class SyntheticPipeline:
    def __init__(
        self,
        structured_generator: StructuredGenerator | None = None,
        text_generator: TextGenerator | None = None,
        validator: SyntheticValidator | None = None,
    ) -> None:
        self._structured_generator = structured_generator or StructuredGenerator()
        self._text_generator = text_generator or TextGenerator()
        self._validator = validator or SyntheticValidator()

    async def generate(self, req: GenerationRequest) -> dict:
        dataset_id = f"ds-{uuid4()}"
        records = []
        approved = 0
        for index in range(req.count):
            record = self._structured_generator.generate(
                dataset_id=dataset_id,
                req=req,
                index=index,
            )
            record = self._text_generator.add_documents(record)
            validation = self._validator.validate(record)
            record = record.model_copy(update={"validation": validation})
            records.append(record)
            if validation.approved:
                approved += 1
        return {
            "dataset_id": dataset_id,
            "generated": len(records),
            "approved": approved,
            "records": records,
        }
