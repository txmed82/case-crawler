from __future__ import annotations

import hashlib
from collections.abc import Callable
from uuid import uuid4

from casecrawler.llm.base import BaseLLMProvider
from casecrawler.llm.factory import get_provider
from casecrawler.models.blueprint import (
    BlueprintGenerationRequest,
    CohortPlan,
    GenerationAttempt,
    GenerationAttemptStatus,
    GenerationRole,
    GenerationRolePolicy,
)
from casecrawler.storage.dataset_store import DatasetStore


ProviderFactory = Callable[[str, str], BaseLLMProvider]


class CohortPlanner:
    def __init__(self, provider_factory: ProviderFactory = get_provider) -> None:
        self._provider_factory = provider_factory

    async def plan(
        self,
        request: BlueprintGenerationRequest,
        *,
        dataset_id: str,
        store: DatasetStore | None = None,
    ) -> CohortPlan:
        policy = request.policy_for(GenerationRole.PLANNER)
        if policy is None:
            raise ValueError("A planner role policy is required for cohort planning.")

        provider = self._provider_factory(policy.provider, policy.model)
        prompt = self._build_prompt(request)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        try:
            result = await provider.generate_structured(
                prompt,
                CohortPlan,
                system=_PLANNER_SYSTEM_PROMPT,
                temperature=policy.temperature,
            )
            plan = self._canonicalize_plan(
                CohortPlan.model_validate(result.data),
                request,
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
            store.save_cohort_plan(plan)
            store.save_generation_attempt(
                self._attempt(
                    dataset_id=dataset_id,
                    policy=policy,
                    status=GenerationAttemptStatus.SUCCEEDED,
                    prompt_hash=prompt_hash,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    artifact_id=plan.plan_id,
                )
            )
        return plan

    def _canonicalize_plan(
        self,
        raw_plan: CohortPlan,
        request: BlueprintGenerationRequest,
    ) -> CohortPlan:
        return CohortPlan.model_validate(
            {
                **raw_plan.model_dump(),
                "plan_id": f"plan-{uuid4()}",
                "request": request.request,
                "target_count": request.target_count,
                "domains": request.domains or raw_plan.domains,
                "settings": request.settings or raw_plan.settings,
                "required_grounding": request.required_grounding,
                "created_by": GenerationRole.PLANNER,
            }
        )

    def _build_prompt(self, request: BlueprintGenerationRequest) -> str:
        return "\n".join(
            [
                "Create a cohort plan for synthetic clinical case generation.",
                f"User request: {request.request}",
                f"Target count: {request.target_count}",
                f"Domains: {request.domains or ['model selected']}",
                f"Settings: {request.settings or ['model selected']}",
                f"Required grounding: {request.required_grounding}",
                f"Diversity targets: {request.diversity_targets}",
                (
                    "Return a CohortPlan whose archetype target counts sum exactly "
                    "to the requested target count."
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
            role=GenerationRole.PLANNER,
            status=status,
            provider=policy.provider,
            model=policy.model,
            prompt_hash=prompt_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            errors=errors or [],
            artifact_id=artifact_id,
        )


_PLANNER_SYSTEM_PROMPT = (
    "You are a clinical cohort planner. Produce diverse, medically plausible "
    "case archetypes as structured data only. Do not create patient artifacts."
)
