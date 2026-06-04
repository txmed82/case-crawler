from __future__ import annotations

import hashlib
import json
import logging
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
logger = logging.getLogger(__name__)


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
            if store is not None:
                self._save_failed_attempt_best_effort(
                    store,
                    self._missing_policy_attempt(request=request, blueprint=blueprint),
                )
            raise ValueError("A judge role policy is required for blueprint judging.")

        provider = self._provider_factory(policy.provider, policy.model)
        prompt = self._build_prompt(request, blueprint)
        prompt_hash = self._prompt_hash(prompt, policy)

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
                self._save_failed_attempt_best_effort(
                    store,
                    self._attempt(
                        blueprint=blueprint,
                        policy=policy,
                        status=GenerationAttemptStatus.FAILED,
                        prompt_hash=prompt_hash,
                        errors=[str(err)],
                    ),
                )
            raise

        if store is not None:
            store.save_judge_report_with_attempt(
                report,
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

    def _prompt_hash(
        self,
        prompt: str,
        policy: GenerationRolePolicy,
    ) -> str:
        payload = {
            "model": policy.model,
            "provider": policy.provider,
            "schema": JudgeReport.__name__,
            "system": _JUDGE_SYSTEM_PROMPT,
            "temperature": policy.temperature,
            "user": prompt,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

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

    def _missing_policy_attempt(
        self,
        *,
        request: BlueprintGenerationRequest,
        blueprint: ClinicalBlueprint,
    ) -> GenerationAttempt:
        prompt_hash_payload = {
            "artifact_id": blueprint.blueprint_id,
            "reason": "missing_policy",
            "request": request.request,
            "role": GenerationRole.JUDGE.value,
        }
        prompt_hash = hashlib.sha256(
            json.dumps(
                prompt_hash_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return GenerationAttempt(
            attempt_id=f"attempt-{uuid4()}",
            dataset_id=blueprint.dataset_id,
            role=GenerationRole.JUDGE,
            status=GenerationAttemptStatus.FAILED,
            provider="unconfigured",
            model="unconfigured",
            prompt_hash=prompt_hash,
            errors=["missing judge role policy"],
            artifact_id=blueprint.blueprint_id,
            metadata={"reason": "missing_policy"},
        )

    def _save_failed_attempt_best_effort(
        self,
        store: DatasetStore,
        attempt: GenerationAttempt,
    ) -> None:
        try:
            store.save_generation_attempt(attempt)
        except Exception:
            logger.exception("Failed to persist judge failure audit.")


_JUDGE_SYSTEM_PROMPT = (
    "You are an independent clinical QA judge for medical AI synthetic-data "
    "blueprints. Evaluate only the supplied structured blueprint. Do not add "
    "new patient facts, final training examples, or patient-facing advice."
)
