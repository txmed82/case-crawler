from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClinicalTextModelProfile:
    name: str
    adapter_type: str
    reference: str
    notes: str
    model_id: str | None = None
    license: str | None = None
    gated: bool = False
    use_policy: str = "review_license_before_use"
    command_template: list[str] = field(default_factory=list)
    input_contract: dict[str, object] = field(default_factory=dict)
    output_contract: dict[str, object] = field(default_factory=dict)
    validation_requirements: list[str] = field(default_factory=list)


CLINICAL_TEXT_MODEL_PROFILES: dict[str, ClinicalTextModelProfile] = {
    "medgemma_4b_it": ClinicalTextModelProfile(
        name="medgemma_4b_it",
        adapter_type="external_command",
        reference="https://huggingface.co/google/medgemma-4b-it",
        model_id="google/medgemma-4b-it",
        license="health-ai-developer-foundations",
        gated=False,
        use_policy="health_ai_terms_review_outputs_before_release",
        command_template=["hf-note-sample", "--model", "google/medgemma-4b-it"],
        input_contract={
            "transport": "stdin",
            "stdin_json": ["record"],
            "record_schema": "SyntheticRecord.model_dump()",
            "prompt_policy": "Use only supplied synthetic facts; do not add PHI.",
        },
        output_contract={
            "transport": "stdout",
            "stdout_json": "ClinicalDocument[] or {'documents': ClinicalDocument[]}",
            "document_fields": [
                "note_type",
                "author_role",
                "clean_text",
                "messy_text",
                "extracted_facts",
            ],
            "optional_fields": ["document_id", "timestamp"],
        },
        validation_requirements=[
            "schema_valid_ClinicalDocument",
            "privacy_phi_scan",
            "memorization_risk_scan",
            "structured_fact_consistency",
            "required_note_type_coverage",
        ],
        notes=(
            "Medical text and image-capable Gemma profile; wrap a local or HF "
            "inference script that converts SyntheticRecord JSON into clinical "
            "documents, then validate before release."
        ),
    ),
    "meditron_7b": ClinicalTextModelProfile(
        name="meditron_7b",
        adapter_type="external_command",
        reference="https://huggingface.co/epfl-llm/meditron-7b",
        model_id="epfl-llm/meditron-7b",
        license="llama2",
        gated=False,
        use_policy="research_foundation_model_validate_and_align_outputs",
        command_template=["hf-note-sample", "--model", "epfl-llm/meditron-7b"],
        input_contract={
            "transport": "stdin",
            "stdin_json": ["record"],
            "record_schema": "SyntheticRecord.model_dump()",
            "prompt_policy": "Use retrieval-grounded synthetic facts only.",
        },
        output_contract={
            "transport": "stdout",
            "stdout_json": "ClinicalDocument[] or {'documents': ClinicalDocument[]}",
            "document_fields": [
                "note_type",
                "author_role",
                "clean_text",
                "messy_text",
                "extracted_facts",
            ],
            "optional_fields": ["document_id", "timestamp"],
        },
        validation_requirements=[
            "schema_valid_ClinicalDocument",
            "privacy_phi_scan",
            "memorization_risk_scan",
            "clinical_contradiction_scan",
            "human_review_before_release",
        ],
        notes=(
            "Medical foundation model profile for research note drafting; model "
            "card warns against unaligned production use, so outputs must be "
            "reviewed and benchmarked before fine-tuning release."
        ),
    ),
    "generic_external_note_generator": ClinicalTextModelProfile(
        name="generic_external_note_generator",
        adapter_type="external_command",
        reference="local_or_hosted_command",
        model_id=None,
        license=None,
        gated=False,
        use_policy="bring_your_own_model_validate_outputs",
        command_template=["hf-note-sample", "--model", "<model_id>"],
        input_contract={
            "transport": "stdin",
            "stdin_json": ["record"],
            "record_schema": "SyntheticRecord.model_dump()",
            "prompt_policy": "Emit only documents grounded in the input record.",
        },
        output_contract={
            "transport": "stdout",
            "stdout_json": "ClinicalDocument[] or {'documents': ClinicalDocument[]}",
            "document_fields": [
                "note_type",
                "author_role",
                "clean_text",
                "messy_text",
                "extracted_facts",
            ],
            "optional_fields": ["document_id", "timestamp"],
        },
        validation_requirements=[
            "schema_valid_ClinicalDocument",
            "privacy_phi_scan",
            "memorization_risk_scan",
            "fact_consistency_review",
        ],
        notes=(
            "Generic adapter profile for a local, hosted, or Hugging Face clinical "
            "note generator that speaks the CaseCrawler stdin/stdout contract."
        ),
    ),
}


def list_clinical_text_model_profiles() -> list[ClinicalTextModelProfile]:
    return list(CLINICAL_TEXT_MODEL_PROFILES.values())


def resolve_clinical_text_model_profile(
    name: str | None,
) -> ClinicalTextModelProfile | None:
    if name is None:
        return None
    try:
        return CLINICAL_TEXT_MODEL_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(CLINICAL_TEXT_MODEL_PROFILES))
        raise ValueError(
            f"Unknown clinical text model profile '{name}'. "
            f"Available profiles: {available}."
        ) from exc
