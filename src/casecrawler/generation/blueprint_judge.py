from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from uuid import uuid4

from casecrawler.llm.base import BaseLLMProvider
from casecrawler.llm.factory import get_provider
from casecrawler.models.blueprint import (
    BlueprintGenerationRequest,
    ClinicalBlueprint,
    GenerationAttempt,
    GenerationAttemptStatus,
    GenerationRole,
    GenerationRolePolicy,
    JudgeReport,
)
from casecrawler.storage.dataset_store import DatasetStore


ProviderFactory = Callable[[str, str], BaseLLMProvider]


class BlueprintJudge:
    def __init__(self, provider_factory: ProviderFactory = get_provider) -> None:
        self._provider_factory = provider_factory

    async def evaluate(
        self,
        request: BlueprintGenerationRequest,
        blueprint: ClinicalBlueprint,
        *,
        store: DatasetStore | None = None,
    ) -> JudgeReport:
        policy = request.policy_for(GenerationRole.JUDGE)
        if policy is None:
            raise ValueError("A judge role policy is required for blueprint judging.")

        provider = self._provider_factory(policy.provider, policy.model)
        prompt = self._build_prompt(request, blueprint)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        try:
            result = await provider.generate_structured(
                prompt,
                JudgeReport,
                system=_JUDGE_SYSTEM_PROMPT,
                temperature=policy.temperature,
            )
            report = self._canonicalize_report(
                JudgeReport.model_validate(result.data),
                blueprint=blueprint,
            )
        except Exception as err:
            if store is not None:
                store.save_generation_attempt(
                    self._attempt(
                        blueprint=blueprint,
                        policy=policy,
                        status=GenerationAttemptStatus.FAILED,
                        prompt_hash=prompt_hash,
                        errors=[str(err)],
                    )
                )
            raise

        if store is not None:
            store.save_judge_report(report)
            store.save_generation_attempt(
                self._attempt(
                    blueprint=blueprint,
                    policy=policy,
                    status=GenerationAttemptStatus.SUCCEEDED,
                    prompt_hash=prompt_hash,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                )
            )
        return report

    def _canonicalize_report(
        self,
        raw_report: JudgeReport,
        *,
        blueprint: ClinicalBlueprint,
    ) -> JudgeReport:
        return JudgeReport.model_validate(
            {
                **raw_report.model_dump(),
                "report_id": f"judge-{uuid4()}",
                "dataset_id": blueprint.dataset_id,
                "artifact_id": blueprint.blueprint_id,
                "role": GenerationRole.JUDGE,
            }
        )

    def _build_prompt(
        self,
        request: BlueprintGenerationRequest,
        blueprint: ClinicalBlueprint,
    ) -> str:
        blueprint_json = json.dumps(
            blueprint.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return "\n".join(
            [
                "Evaluate this clinical blueprint before synthetic case generation.",
                f"User request: {request.request}",
                f"Blueprint id: {blueprint.blueprint_id}",
                f"Dataset id: {blueprint.dataset_id}",
                "Judge for clinical plausibility, internal consistency, grounding, "
                "safety, and usefulness as a source-of-truth case plan.",
                (
                    "Return a JudgeReport with a calibrated score, pass/fail decision, "
                    "rubric name, and concrete findings for each material issue."
                ),
                f"Blueprint JSON: {blueprint_json}",
            ]
        )

    def _attempt(
        self,
        *,
        blueprint: ClinicalBlueprint,
        policy: GenerationRolePolicy,
        status: GenerationAttemptStatus,
        prompt_hash: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        errors: list[str] | None = None,
    ) -> GenerationAttempt:
        return GenerationAttempt(
            attempt_id=f"attempt-{uuid4()}",
            dataset_id=blueprint.dataset_id,
            role=GenerationRole.JUDGE,
            status=status,
            provider=policy.provider,
            model=policy.model,
            prompt_hash=prompt_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            errors=errors or [],
            artifact_id=blueprint.blueprint_id,
        )


_JUDGE_SYSTEM_PROMPT = (
    "You are an independent clinical QA judge for medical AI synthetic-data "
    "blueprints. Evaluate only the supplied structured blueprint. Do not add "
    "new patient facts, final training examples, or patient-facing advice."
)
