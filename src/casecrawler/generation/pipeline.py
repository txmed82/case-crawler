from __future__ import annotations

import uuid
from datetime import datetime

from casecrawler.generation.case_generator import CaseGeneratorAgent
from casecrawler.generation.clinical_reviewer import ClinicalReviewerAgent
from casecrawler.generation.decision_tree_builder import DecisionTreeBuilderAgent
from casecrawler.generation.retriever import Retriever
from casecrawler.llm.base import BaseLLMProvider
from casecrawler.models.case import DifficultyLevel, GeneratedCase


class GenerationPipeline:
    def __init__(
        self,
        provider: BaseLLMProvider,
        retriever: Retriever,
        max_retries: int = 3,
        review_threshold: float = 0.7,
    ) -> None:
        self._case_gen = CaseGeneratorAgent(provider=provider)
        self._tree_builder = DecisionTreeBuilderAgent(provider=provider)
        self._reviewer = ClinicalReviewerAgent(provider=provider, threshold=review_threshold)
        self._retriever = retriever
        self._max_retries = max_retries
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    async def generate_one(
        self, topic: str, difficulty: str = "resident",
    ) -> GeneratedCase | None:
        """Generate a single case with retry logic. Returns None if all retries fail."""
        # Stage 1: Retrieve
        chunks = self._retriever.retrieve(topic)
        context = self._retriever.format_context(chunks)
        source_refs = [
            {"type": c["source"], "reference": c["source_document_id"], "chunk_ids": [c["chunk_id"]]}
            for c in chunks
        ]

        retry_notes: list[str] | None = None
        for attempt in range(self._max_retries):
            # Stage 2: Generate case
            gen_result = await self._case_gen.generate(
                topic=topic, difficulty=difficulty, context=context, retry_notes=retry_notes,
            )
            self._total_input_tokens += gen_result.input_tokens
            self._total_output_tokens += gen_result.output_tokens
            gen_data = gen_result.data

            # Stage 3: Build decision tree
            gt_json = gen_data.ground_truth.model_dump_json()
            tree_result = await self._tree_builder.build(
                vignette=gen_data.vignette,
                ground_truth_json=gt_json,
                difficulty=difficulty,
                context=context,
                retry_notes=retry_notes,
            )
            self._total_input_tokens += tree_result.input_tokens
            self._total_output_tokens += tree_result.output_tokens
            tree_data = tree_result.data

            # Assemble case for review
            case = GeneratedCase(
                case_id=str(uuid.uuid4()),
                topic=topic,
                difficulty=DifficultyLevel(difficulty),
                specialty=gen_data.specialty,
                patient=gen_data.patient,
                vignette=gen_data.vignette,
                decision_prompt=gen_data.decision_prompt,
                ground_truth=gen_data.ground_truth,
                decision_tree=tree_data.decision_tree,
                complications=tree_data.complications,
                review=None,  # placeholder
                sources=source_refs,
                metadata={
                    "generated_at": datetime.now().isoformat(),
                    "model": gen_result.model,
                    "retry_count": attempt,
                    "total_input_tokens": self._total_input_tokens,
                    "total_output_tokens": self._total_output_tokens,
                },
            )

            # Stage 4: Clinical review
            review_result = await self._reviewer.review(
                case_json=case.model_dump_json(),
                context=context,
            )
            self._total_input_tokens += review_result.input_tokens
            self._total_output_tokens += review_result.output_tokens

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

            # Rejected — set retry notes for next attempt
            retry_notes = review_result.data.notes

        # All retries exhausted
        return None

    async def generate_batch(
        self, topic: str, count: int, difficulty: str = "resident",
    ) -> dict:
        """Generate multiple cases sequentially. Returns summary."""
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
