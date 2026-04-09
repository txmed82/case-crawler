from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime

from casecrawler.generation.blueprint_reviewer import BlueprintReviewerAgent
from casecrawler.generation.case_planner import CasePlannerAgent
from casecrawler.generation.clinical_reviewer import ClinicalReviewerAgent
from casecrawler.generation.consistency_checker import ConsistencyCheckerAgent
from casecrawler.generation.lab_panels import LAB_PANELS
from casecrawler.generation.phase_renderer import PhaseRendererAgent
from casecrawler.generation.retriever import Retriever
from casecrawler.llm.base import BaseLLMProvider
from casecrawler.models.case import DifficultyLevel, GeneratedCase, GroundTruth, Patient


class MultiStepPipeline:
    def __init__(
        self,
        provider: BaseLLMProvider,
        retriever: Retriever,
        max_retries: int = 3,
        review_threshold: float = 0.7,
    ) -> None:
        self._planner = CasePlannerAgent(provider=provider)
        self._blueprint_reviewer = BlueprintReviewerAgent(provider=provider)
        self._renderer = PhaseRendererAgent(provider=provider)
        self._consistency_checker = ConsistencyCheckerAgent(provider=provider)
        self._clinical_reviewer = ClinicalReviewerAgent(provider=provider, threshold=review_threshold)
        self._retriever = retriever
        self._max_retries = max_retries
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _track_tokens(self, result) -> None:
        self._total_input_tokens += result.input_tokens
        self._total_output_tokens += result.output_tokens

    def _build_lab_panel_context(self) -> str:
        lines = []
        for panel in LAB_PANELS.values():
            components = ", ".join(
                f"{c.name} {c.reference_low}-{c.reference_high} {c.unit}"
                for c in panel.components
            )
            lines.append(f"{panel.name}: {components}")
        return "\n".join(lines)

    async def generate_one(
        self, topic: str, difficulty: str = "resident",
    ) -> GeneratedCase | None:
        # Stage 1: Retrieve
        chunks = self._retriever.retrieve(topic)
        context = self._retriever.format_context(chunks)
        source_refs = [
            {"type": c["source"], "reference": c["source_document_id"], "chunk_ids": [c["chunk_id"]]}
            for c in chunks
        ]
        lab_panel_context = self._build_lab_panel_context()

        retry_notes: list[str] | None = None
        for attempt in range(self._max_retries):
            # Stage 2: Plan with review
            plan_data = await self._plan_with_review(topic, difficulty, context, retry_notes)
            if plan_data is None:
                continue

            blueprint_data = plan_data.blueprint
            blueprint_json = blueprint_data.model_dump_json()

            # Stage 3: Render phases in parallel
            render_tasks = []
            for phase_bp in blueprint_data.phases:
                render_tasks.append(
                    self._renderer.render(
                        blueprint_json=blueprint_json,
                        phase_json=phase_bp.model_dump_json(),
                        difficulty=difficulty,
                        context=context,
                        lab_panel_context=lab_panel_context,
                    )
                )
            render_results = await asyncio.gather(*render_tasks)
            for r in render_results:
                self._track_tokens(r)

            phases = [r.data.phase for r in render_results]

            # Stage 4: Consistency check
            phases_json = json.dumps([p.model_dump() for p in phases])
            consistency_result = await self._consistency_checker.check(phases_json)
            self._track_tokens(consistency_result)

            # If issues found, re-render affected phases (max 2 iterations)
            for _ in range(2):
                if not consistency_result.data.issues:
                    break
                affected = {issue.phase_number for issue in consistency_result.data.issues}
                for phase_bp in blueprint_data.phases:
                    if phase_bp.phase_number in affected:
                        re_result = await self._renderer.render(
                            blueprint_json=blueprint_json,
                            phase_json=phase_bp.model_dump_json(),
                            difficulty=difficulty,
                            context=context,
                            lab_panel_context=lab_panel_context,
                        )
                        self._track_tokens(re_result)
                        idx = phase_bp.phase_number - 1
                        phases[idx] = re_result.data.phase
                phases_json = json.dumps([p.model_dump() for p in phases])
                consistency_result = await self._consistency_checker.check(phases_json)
                self._track_tokens(consistency_result)

            if consistency_result.data.issues:
                retry_notes = [f"Consistency issue: {i.issue}" for i in consistency_result.data.issues]
                continue

            # Build backward-compat fields
            vignette = "\n\n".join(p.narrative for p in phases)
            ground_truth = GroundTruth(
                diagnosis=blueprint_data.diagnosis,
                optimal_next_step=blueprint_data.phases[0].correct_action,
                rationale=blueprint_data.phases[0].key_reasoning,
                key_findings=[],
            )

            case = GeneratedCase(
                case_id=str(uuid.uuid4()),
                topic=topic,
                difficulty=DifficultyLevel(difficulty),
                specialty=plan_data.specialty,
                patient=plan_data.patient,
                blueprint=blueprint_data,
                phases=phases,
                vignette=vignette,
                decision_prompt="What would you do next?",
                ground_truth=ground_truth,
                decision_tree=[],
                complications=[],
                review=None,
                sources=source_refs,
                metadata={
                    "generated_at": datetime.now().isoformat(),
                    "model": render_results[0].model if render_results else "unknown",
                    "retry_count": attempt,
                    "total_input_tokens": self._total_input_tokens,
                    "total_output_tokens": self._total_output_tokens,
                    "multi_step": True,
                },
            )

            # Stage 5: Clinical review
            review_result = await self._clinical_reviewer.review(
                case_json=case.model_dump_json(), context=context,
            )
            self._track_tokens(review_result)

            case = case.model_copy(update={
                "review": review_result.data,
                "metadata": {
                    **case.metadata,
                    "total_input_tokens": self._total_input_tokens,
                    "total_output_tokens": self._total_output_tokens,
                },
            })

            if review_result.data.approved:
                return case

            retry_notes = review_result.data.notes

        return None

    async def _plan_with_review(
        self, topic: str, difficulty: str, context: str, retry_notes: list[str] | None,
    ):
        for _ in range(self._max_retries):
            plan_result = await self._planner.plan(
                topic=topic, difficulty=difficulty, context=context, retry_notes=retry_notes,
            )
            self._track_tokens(plan_result)

            review_result = await self._blueprint_reviewer.review(
                blueprint_json=plan_result.data.blueprint.model_dump_json(),
                context=context,
            )
            self._track_tokens(review_result)

            if review_result.data.approved:
                return plan_result.data

            retry_notes = review_result.data.notes

        return None

    async def generate_batch(
        self, topic: str, count: int, difficulty: str = "resident",
    ) -> dict:
        generated = []
        failed = 0
        for _ in range(count):
            case = await self.generate_one(topic=topic, difficulty=difficulty)
            if case:
                generated.append(case)
            else:
                failed += 1
        return {
            "cases": generated,
            "generated": len(generated),
            "failed": failed,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }
