from __future__ import annotations

import asyncio
from collections.abc import Callable

from pydantic import BaseModel, Field

from casecrawler.generation.synthetic_pipeline import SyntheticPipeline
from casecrawler.models.dataset import GenerationRequest
from casecrawler.models.synthetic import ComplexityProfile, Modality, SyntheticRecord


class OpenSourceScenarioResult(BaseModel):
    topic: str
    requested_count: int
    generated_count: int
    approved_count: int
    blocking_issue_count: int
    modalities: list[str]
    artifact_counts: dict[str, int] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            self.generated_count == self.requested_count
            and self.approved_count > 0
            and self.blocking_issue_count == 0
        )


class OpenSourceBenchmarkResult(BaseModel):
    scenarios: list[OpenSourceScenarioResult]

    @property
    def generated(self) -> int:
        return sum(scenario.generated_count for scenario in self.scenarios)

    @property
    def approved(self) -> int:
        return sum(scenario.approved_count for scenario in self.scenarios)

    @property
    def blocking_issues(self) -> int:
        return sum(scenario.blocking_issue_count for scenario in self.scenarios)

    @property
    def passed(self) -> bool:
        return all(scenario.passed for scenario in self.scenarios)

    def to_report(self) -> dict:
        return {
            "artifact_type": "casecrawler_open_source_benchmark",
            "passed": self.passed,
            "generated": self.generated,
            "approved": self.approved,
            "blocking_issues": self.blocking_issues,
            "scenarios": [
                {
                    **scenario.model_dump(),
                    "passed": scenario.passed,
                }
                for scenario in self.scenarios
            ],
        }


PipelineFactory = Callable[[], SyntheticPipeline]


class OpenSourceBenchmarkSuite:
    """Offline smoke suite for open-source contributors and CI."""

    def __init__(self, pipeline_factory: PipelineFactory | None = None) -> None:
        self._pipeline_factory = pipeline_factory or SyntheticPipeline

    def run_generation_smoke(
        self,
        requests: list[GenerationRequest] | None = None,
    ) -> OpenSourceBenchmarkResult:
        return asyncio.run(self.run_generation_smoke_async(requests=requests))

    async def run_generation_smoke_async(
        self,
        requests: list[GenerationRequest] | None = None,
    ) -> OpenSourceBenchmarkResult:
        scenarios = []
        resolved_requests = (
            requests if requests is not None else default_open_source_requests()
        )
        for req in resolved_requests:
            result = await self._pipeline_factory().generate(req)
            scenarios.append(
                _scenario_result(
                    req=req,
                    records=result["records"],
                    generated_count=result["generated"],
                    approved_count=result["approved"],
                )
            )
        return OpenSourceBenchmarkResult(scenarios=scenarios)


def default_open_source_requests(count: int = 2) -> list[GenerationRequest]:
    modalities = [
        Modality.STRUCTURED_EHR,
        Modality.CLINICAL_TEXT,
        Modality.LABS,
        Modality.VITALS,
        Modality.TIME_SERIES,
    ]
    return [
        GenerationRequest(
            topic="sepsis",
            count=count,
            modalities=modalities,
            cohort_constraints={"encounter_count": 2},
        ),
        GenerationRequest(
            topic="heart failure exacerbation",
            count=count,
            complexity=ComplexityProfile.COMPLEX,
            modalities=modalities,
        ),
    ]


def _scenario_result(
    *,
    req: GenerationRequest,
    records: list[SyntheticRecord],
    generated_count: int,
    approved_count: int,
) -> OpenSourceScenarioResult:
    return OpenSourceScenarioResult(
        topic=req.topic,
        requested_count=req.count,
        generated_count=generated_count,
        approved_count=approved_count,
        blocking_issue_count=sum(
            1
            for record in records
            for issue in (record.validation.issues if record.validation else [])
            if issue.severity == "error"
        ),
        modalities=[modality.value for modality in req.modalities],
        artifact_counts={
            "records": len(records),
            "documents": sum(len(record.documents) for record in records),
            "labs": sum(len(record.labs) for record in records),
            "vitals": sum(len(record.vitals) for record in records),
            "time_series_channels": sum(len(record.time_series) for record in records),
            "imaging_assets": sum(len(record.imaging) for record in records),
        },
    )
