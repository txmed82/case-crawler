from __future__ import annotations

import hashlib
from collections.abc import Callable
from uuid import uuid4

from casecrawler.llm.base import BaseLLMProvider
from casecrawler.llm.factory import get_provider
from casecrawler.models.blueprint import (
    BlueprintGenerationRequest,
    ClinicalBlueprint,
    CohortArchetype,
    CohortPlan,
    GenerationAttempt,
    GenerationAttemptStatus,
    GenerationRole,
    GenerationRolePolicy,
)
from casecrawler.storage.dataset_store import DatasetStore


ProviderFactory = Callable[[str, str], BaseLLMProvider]


class ClinicalBlueprintGenerator:
    def __init__(self, provider_factory: ProviderFactory = get_provider) -> None:
        self._provider_factory = provider_factory

    async def generate_for_archetype(
        self,
        request: BlueprintGenerationRequest,
        *,
        plan: CohortPlan,
        archetype: CohortArchetype,
        dataset_id: str,
        sequence_index: int = 0,
        store: DatasetStore | None = None,
    ) -> ClinicalBlueprint:
        policy = request.policy_for(GenerationRole.BLUEPRINT_GENERATOR)
        if policy is None:
            raise ValueError(
                "A blueprint_generator role policy is required for blueprint generation."
            )

        provider = self._provider_factory(policy.provider, policy.model)
        prompt = self._build_prompt(
            request,
            plan=plan,
            archetype=archetype,
            sequence_index=sequence_index,
        )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        try:
            result = await provider.generate_structured(
                prompt,
                ClinicalBlueprint,
                system=_BLUEPRINT_SYSTEM_PROMPT,
                temperature=policy.temperature,
            )
            blueprint = self._canonicalize_blueprint(
                ClinicalBlueprint.model_validate(result.data),
                plan=plan,
                archetype=archetype,
                dataset_id=dataset_id,
            )
        except Exception as err:
            if store is not None:
                store.save_generation_attempt(
                    self._attempt(
                        dataset_id=dataset_id,
                        policy=policy,
                        status=GenerationAttemptStatus.FAILED,
                        prompt_hash=prompt_hash,
                        errors=[str(err)],
                    )
                )
            raise

        if store is not None:
            store.save_blueprint(blueprint)
            store.save_generation_attempt(
                self._attempt(
                    dataset_id=dataset_id,
                    policy=policy,
                    status=GenerationAttemptStatus.SUCCEEDED,
                    prompt_hash=prompt_hash,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    artifact_id=blueprint.blueprint_id,
                )
            )
        return blueprint

    def _canonicalize_blueprint(
        self,
        raw_blueprint: ClinicalBlueprint,
        *,
        plan: CohortPlan,
        archetype: CohortArchetype,
        dataset_id: str,
    ) -> ClinicalBlueprint:
        return ClinicalBlueprint.model_validate(
            {
                **raw_blueprint.model_dump(),
                "blueprint_id": f"bp-{uuid4()}",
                "dataset_id": dataset_id,
                "cohort_plan_id": plan.plan_id,
                "archetype_name": archetype.name,
                "organ_system": archetype.organ_system,
                "setting": archetype.setting,
            }
        )

    def _build_prompt(
        self,
        request: BlueprintGenerationRequest,
        *,
        plan: CohortPlan,
        archetype: CohortArchetype,
        sequence_index: int,
    ) -> str:
        return "\n".join(
            [
                "Create one clinical blueprint for synthetic case generation.",
                f"User request: {request.request}",
                f"Cohort plan id: {plan.plan_id}",
                f"Blueprint index within archetype: {sequence_index}",
                f"Archetype: {archetype.name}",
                f"Organ system: {archetype.organ_system}",
                f"Setting: {archetype.setting}",
                f"Required modalities: {[item.value for item in archetype.required_modalities]}",
                f"Task targets: {archetype.task_targets}",
                f"Safety constraints: {archetype.safety_constraints}",
                (
                    "Return a ClinicalBlueprint with explicit diagnosis support, "
                    "uncertainty points where clinically appropriate, and evidence "
                    "claims separated into supported, inferred, and unsupported."
                ),
            ]
        )

    def _attempt(
        self,
        *,
        dataset_id: str,
        policy: GenerationRolePolicy,
        status: GenerationAttemptStatus,
        prompt_hash: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        artifact_id: str | None = None,
        errors: list[str] | None = None,
    ) -> GenerationAttempt:
        return GenerationAttempt(
            attempt_id=f"attempt-{uuid4()}",
            dataset_id=dataset_id,
            role=GenerationRole.BLUEPRINT_GENERATOR,
            status=status,
            provider=policy.provider,
            model=policy.model,
            prompt_hash=prompt_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            errors=errors or [],
            artifact_id=artifact_id,
        )


_BLUEPRINT_SYSTEM_PROMPT = (
    "You are a clinical blueprint generator. Produce medically plausible "
    "source-of-truth case plans as structured data only. Do not write final "
    "training examples or patient-facing advice."
)
