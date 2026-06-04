from __future__ import annotations

from pydantic import BaseModel

from casecrawler.generation.blueprint_generator import (
    ClinicalBlueprintGenerator,
    ProviderFactory,
)
from casecrawler.generation.blueprint_planner import CohortPlanner
from casecrawler.models.blueprint import (
    BlueprintGenerationRequest,
    ClinicalBlueprint,
    CohortPlan,
)
from casecrawler.storage.dataset_store import DatasetStore


class BlueprintPipelineResult(BaseModel):
    dataset_id: str
    plan: CohortPlan
    blueprints: list[ClinicalBlueprint]

    @property
    def generated_count(self) -> int:
        return len(self.blueprints)


class BlueprintPipeline:
    def __init__(
        self,
        *,
        provider_factory: ProviderFactory | None = None,
        planner: CohortPlanner | None = None,
        generator: ClinicalBlueprintGenerator | None = None,
    ) -> None:
        if planner is not None:
            self._planner = planner
        elif provider_factory is None:
            self._planner = CohortPlanner()
        else:
            self._planner = CohortPlanner(provider_factory=provider_factory)

        if generator is not None:
            self._generator = generator
        elif provider_factory is None:
            self._generator = ClinicalBlueprintGenerator()
        else:
            self._generator = ClinicalBlueprintGenerator(
                provider_factory=provider_factory
            )

    async def generate(
        self,
        request: BlueprintGenerationRequest,
        *,
        dataset_id: str,
        store: DatasetStore | None = None,
    ) -> BlueprintPipelineResult:
        plan = await self._planner.plan(request, dataset_id=dataset_id, store=store)
        blueprints: list[ClinicalBlueprint] = []
        for archetype in plan.archetypes:
            for sequence_index in range(archetype.target_count):
                blueprints.append(
                    await self._generator.generate_for_archetype(
                        request,
                        plan=plan,
                        archetype=archetype,
                        dataset_id=dataset_id,
                        sequence_index=sequence_index,
                        store=store,
                    )
                )
        return BlueprintPipelineResult(
            dataset_id=dataset_id,
            plan=plan,
            blueprints=blueprints,
        )
