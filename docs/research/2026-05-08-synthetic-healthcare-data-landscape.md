# Synthetic Healthcare Data Landscape Notes

Date: 2026-05-08

Purpose: capture external research signals for the CaseCrawler synthetic healthcare data platform so implementation choices are traceable. This is not a completion audit.

## Avrix / Axivis Due Diligence

Searches for `Avrix` did not identify a healthcare synthetic-data platform. The current public `avrix.io` result is a game-key distribution marketplace, and `avrix.es` is an industrial VR/training company. Those findings do not map to CaseCrawler's clinical synthetic data objective.

The closest similarly named healthcare result found during the scan was AXIVIS, which presents itself as a medical intelligence/data platform focused on structured real-world clinical data, clinician validation, de-identification, and governance across oncology, radiology, genomics, ICU, neurology, and endocrinology. AXIVIS is useful as a product-positioning reference for provenance, clinician validation, and governance language, but it is not an open synthetic generator dependency.

Sources:
- https://avrix.io/
- https://www.avrix.es/
- https://axivis.ai/lab_screening.html

## Open Baselines And Reference Data

Synthea remains the primary open baseline for full synthetic patient histories. Its public project describes birth-to-death lifecycle simulation, rule modules, encounters, conditions, allergies, medications, observations/vitals, labs, procedures, care plans, and exports including FHIR, Bulk FHIR NDJSON, C-CDA, and CSV. CaseCrawler should keep Synthea as a first-class import/generation adapter rather than trying to replace its standards-shaped longitudinal EHR work.

Sources:
- https://synthetichealth.github.io/synthea/
- https://github.com/synthetichealth/synthea

## Hugging Face References

Hugging Face remains the best practical source for pluggable clinical text, structured extraction, time-series, and imaging references. The repo currently registers clinical note, note-to-FHIR, de-identification/coding, Synthea-derived, time-series, and radiology/image references in `src/casecrawler/integrations/huggingface.py`.

Important current anchors:
- `raman07/SynthCheX-75K-v2`: synthetic chest radiograph image-text dataset released with CheXGenBench. The dataset card reports 75,649 image-text samples, Apache-2.0 licensing, pathological annotations, and filtration using a medical VLM. This is a strong validation and benchmark reference for generated chest X-ray assets.
- `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`: MIT-licensed biomedical image-text model pretrained on PMC-15M figure-caption pairs. This is a practical optional image-text alignment validator, not a full clinical correctness oracle.
- `MuhangTian/TimeDiff` and RawMed-style profiles remain useful as external command/model-profile hooks for learned EHR time-series generation and validation, with deterministic local generation kept as the offline default.

Sources:
- https://huggingface.co/datasets/raman07/SynthCheX-75K-v2
- https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

## Risk Signals To Encode In Validation

Synthetic healthcare data is only useful for model training if it avoids three common failure modes:

1. Data leakage or over-similarity to validation data.
2. Cohorts that are coherent one record at a time but statistically unrealistic across populations.
3. Bias inherited from reference corpora, model pretraining, or manually selected topics.

CaseCrawler should treat benchmark scores, cohort distribution reports, source/reference provenance, privacy screens, and release gates as core product features rather than documentation extras. A generated dataset is not release-ready just because schema validation passes.

Sources:
- https://www.forbes.com/councils/forbestechcouncil/2023/05/26/the-dangers-of-using-synthetic-patient-data-to-build-healthcare-ai-models/
- https://www.ahrq.gov/data/innovations/syh-dr.html

## Implementation Implications

- Keep deterministic offline generation for repeatable CI and smoke tests.
- Keep external model adapters behind stable contracts: clinical text command, imaging command/diffusers profile, and time-series command/profile.
- Prefer reference importers and benchmark fixtures over hard-coded model assumptions.
- Release packages should include provenance, policy metadata, benchmark reports, dataset/model cards, quality reports, split manifests, and objective coverage artifacts.
- For imaging, focus first on chest X-ray generation and validation because reference datasets and evaluation work are strongest there. Broader CT/MR/multiregion image generation should stay profile-gated with explicit license and review policies.
- Treat named commercial/proprietary products as positioning references only unless they expose open models, datasets, or reproducible evaluation protocols.
