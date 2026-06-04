from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from uuid import uuid4

from pydantic import BaseModel

from casecrawler.generation.blueprint_judge import BlueprintJudge
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


class BlueprintRepairResult(BaseModel):
    original_blueprint: ClinicalBlueprint
    final_blueprint: ClinicalBlueprint
    judge_reports: list[JudgeReport]
    repaired_blueprints: list[ClinicalBlueprint]
    repair_rounds: int
    passed: bool


class BlueprintRepairLoop:
    def __init__(
        self,
        *,
        provider_factory: ProviderFactory = get_provider,
        judge: BlueprintJudge | None = None,
    ) -> None:
        self._provider_factory = provider_factory
        self._judge = judge or BlueprintJudge(provider_factory=provider_factory)

    async def run(
        self,
        request: BlueprintGenerationRequest,
        blueprint: ClinicalBlueprint,
        *,
        store: DatasetStore | None = None,
    ) -> BlueprintRepairResult:
        current = blueprint
        judge_reports: list[JudgeReport] = []
        repaired_blueprints: list[ClinicalBlueprint] = []

        for round_index in range(request.max_repair_rounds + 1):
            judge_report = await self._judge.evaluate(request, current, store=store)
            judge_reports.append(judge_report)
            if judge_report.passed:
                return BlueprintRepairResult(
                    original_blueprint=blueprint,
                    final_blueprint=current,
                    judge_reports=judge_reports,
                    repaired_blueprints=repaired_blueprints,
                    repair_rounds=len(repaired_blueprints),
                    passed=True,
                )
            if round_index >= request.max_repair_rounds:
                break

            repair_policy = request.policy_for(GenerationRole.REPAIR)
            if repair_policy is None:
                raise ValueError(
                    "A repair role policy is required when judge repair is needed."
                )

            repair_round = round_index + 1
            repair_request_attempt = self._repair_requested_attempt(
                blueprint=current,
                policy=repair_policy,
                judge_report=judge_report,
                repair_round=repair_round,
            )
            if store is not None:
                store.save_generation_attempt(repair_request_attempt)

            current = await self._repair_blueprint(
                request,
                current,
                judge_report,
                policy=repair_policy,
                repair_round=repair_round,
                store=store,
            )
            repaired_blueprints.append(current)

        return BlueprintRepairResult(
            original_blueprint=blueprint,
            final_blueprint=current,
            judge_reports=judge_reports,
            repaired_blueprints=repaired_blueprints,
            repair_rounds=len(repaired_blueprints),
            passed=False,
        )

    async def _repair_blueprint(
        self,
        request: BlueprintGenerationRequest,
        blueprint: ClinicalBlueprint,
        judge_report: JudgeReport,
        *,
        policy: GenerationRolePolicy,
        repair_round: int,
        store: DatasetStore | None,
    ) -> ClinicalBlueprint:
        provider = self._provider_factory(policy.provider, policy.model)
        prompt = self._build_repair_prompt(
            request,
            blueprint=blueprint,
            judge_report=judge_report,
            repair_round=repair_round,
        )
        prompt_hash = self._prompt_hash(prompt, policy)

        try:
            result = await provider.generate_structured(
                prompt,
                ClinicalBlueprint,
                system=_REPAIR_SYSTEM_PROMPT,
                temperature=policy.temperature,
            )
            repaired = self._canonicalize_repair(
                ClinicalBlueprint.model_validate(result.data),
                source=blueprint,
                judge_report=judge_report,
                repair_round=repair_round,
            )
        except Exception as err:
            if store is not None:
                self._save_failed_attempt_best_effort(
                    store,
                    self._repair_attempt(
                        blueprint=blueprint,
                        policy=policy,
                        status=GenerationAttemptStatus.FAILED,
                        prompt_hash=prompt_hash,
                        repair_round=repair_round,
                        judge_report=judge_report,
                        errors=[str(err)],
                    ),
                )
            raise

        if store is not None:
            store.save_blueprint_with_attempt(
                repaired,
                self._repair_attempt(
                    blueprint=repaired,
                    policy=policy,
                    status=GenerationAttemptStatus.SUCCEEDED,
                    prompt_hash=prompt_hash,
                    repair_round=repair_round,
                    judge_report=judge_report,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                )
            )
        return repaired

    def _canonicalize_repair(
        self,
        raw_blueprint: ClinicalBlueprint,
        *,
        source: ClinicalBlueprint,
        judge_report: JudgeReport,
        repair_round: int,
    ) -> ClinicalBlueprint:
        metadata = {
            **raw_blueprint.metadata,
            "parent_blueprint_id": source.blueprint_id,
            "repair_round": repair_round,
            "judge_report_id": judge_report.report_id,
        }
        return ClinicalBlueprint.model_validate(
            {
                **raw_blueprint.model_dump(),
                "blueprint_id": f"bp-{uuid4()}",
                "dataset_id": source.dataset_id,
                "cohort_plan_id": source.cohort_plan_id,
                "archetype_name": source.archetype_name,
                "organ_system": source.organ_system,
                "setting": source.setting,
                "metadata": metadata,
            }
        )

    def _build_repair_prompt(
        self,
        request: BlueprintGenerationRequest,
        *,
        blueprint: ClinicalBlueprint,
        judge_report: JudgeReport,
        repair_round: int,
    ) -> str:
        blueprint_json = json.dumps(
            blueprint.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        report_json = json.dumps(
            judge_report.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return "\n".join(
            [
                "Repair this clinical blueprint so it can pass independent review.",
                f"User request: {request.request}",
                f"Repair round: {repair_round}",
                f"Blueprint id: {blueprint.blueprint_id}",
                f"Judge report id: {judge_report.report_id}",
                (
                    "Preserve the clinical intent, dataset lineage, and task target. "
                    "Change only fields needed to address judge findings."
                ),
                f"Blueprint JSON: {blueprint_json}",
                f"JudgeReport JSON: {report_json}",
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
            "schema": ClinicalBlueprint.__name__,
            "system": _REPAIR_SYSTEM_PROMPT,
            "temperature": policy.temperature,
            "user": prompt,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _repair_requested_attempt(
        self,
        *,
        blueprint: ClinicalBlueprint,
        policy: GenerationRolePolicy,
        judge_report: JudgeReport,
        repair_round: int,
    ) -> GenerationAttempt:
        payload = {
            "artifact_id": blueprint.blueprint_id,
            "judge_report_id": judge_report.report_id,
            "repair_round": repair_round,
            "status": GenerationAttemptStatus.REPAIR_REQUESTED.value,
        }
        return GenerationAttempt(
            attempt_id=f"attempt-{uuid4()}",
            dataset_id=blueprint.dataset_id,
            role=GenerationRole.REPAIR,
            status=GenerationAttemptStatus.REPAIR_REQUESTED,
            provider=policy.provider,
            model=policy.model,
            prompt_hash=_hash_payload(payload),
            artifact_id=blueprint.blueprint_id,
            metadata={
                "judge_report_id": judge_report.report_id,
                "repair_round": repair_round,
            },
        )

    def _repair_attempt(
        self,
        *,
        blueprint: ClinicalBlueprint,
        policy: GenerationRolePolicy,
        status: GenerationAttemptStatus,
        prompt_hash: str,
        repair_round: int,
        judge_report: JudgeReport,
        input_tokens: int = 0,
        output_tokens: int = 0,
        errors: list[str] | None = None,
    ) -> GenerationAttempt:
        return GenerationAttempt(
            attempt_id=f"attempt-{uuid4()}",
            dataset_id=blueprint.dataset_id,
            role=GenerationRole.REPAIR,
            status=status,
            provider=policy.provider,
            model=policy.model,
            prompt_hash=prompt_hash,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            errors=errors or [],
            artifact_id=blueprint.blueprint_id,
            metadata={
                "judge_report_id": judge_report.report_id,
                "repair_round": repair_round,
            },
        )

    def _save_failed_attempt_best_effort(
        self,
        store: DatasetStore,
        attempt: GenerationAttempt,
    ) -> None:
        try:
            store.save_generation_attempt(attempt)
        except Exception:
            logger.exception("Failed to persist blueprint repair failure audit.")


def _hash_payload(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


_REPAIR_SYSTEM_PROMPT = (
    "You are a clinical blueprint repair model. Return a corrected "
    "ClinicalBlueprint as structured data only. Do not create final synthetic "
    "case text, and do not add unsupported patient facts."
)
