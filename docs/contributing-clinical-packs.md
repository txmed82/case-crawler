# Contributing Clinical Content

CaseCrawler is open source, but it must stay strict about clinical safety,
privacy, provenance, and testability. This guide describes how to contribute
condition-specific clinical content today and how to prepare work for future
YAML clinical packs.

## Ground Rules

- Do not use real patient data, screenshots, notes, identifiers, dates of care,
  accession numbers, medical record numbers, or copied chart text.
- Use public clinical concepts and cite public references when adding new
  condition patterns, benchmark fixtures, model profiles, or validation rules.
- Keep examples synthetic, concise, and reproducible.
- Prefer standard code systems such as SNOMED CT, LOINC, RxNorm, ICD-10-CM, and
  CPT when a code is known and license-compatible.
- Include tests for every contributor-visible behavior change.
- Treat generated outputs as synthetic research artifacts, not medical advice or
  clinician-reviewed truth.

## Current Contribution Points

Clinical coverage can be improved without changing the project architecture:

- Add or refine generation recipes in `src/casecrawler/generation/recipes.py`.
- Add deterministic structured patterns in
  `src/casecrawler/generation/structured_generator.py`.
- Add note style and signal variation in
  `src/casecrawler/generation/text_generator.py`.
- Add imaging profiles or templates in `src/casecrawler/generation/imaging_models.py`
  and `src/casecrawler/generation/imaging_templates.py`.
- Add public reference fixtures in `src/casecrawler/integrations/reference_fixtures.py`.
- Add validators in `src/casecrawler/validation/`.
- Add golden regression cases under `tests/fixtures/golden_cases/`.

## Minimal Clinical Content Checklist

For a new condition, specialty scenario, or benchmark slice, include:

- A clear topic or recipe name.
- One primary diagnosis concept and, when practical, a standard code.
- Required clinical artifacts such as structured EHR facts, labs, vitals,
  clinical text, time-series channels, imaging, or medication history.
- Plausible abnormal findings and at least one normal or negative finding when
  the clinical task depends on contrast.
- Validation expectations for required artifacts, privacy, and clinical
  consistency.
- Public citations for guidelines, datasets, or model cards used to shape the
  contribution.

## Future YAML Clinical Packs

The intended long-term format is a YAML file that describes condition-specific
generation and validation expectations. Until that loader exists, use this shape
as design guidance and keep implementation in the current Python extension
points.

```yaml
key: asthma_exacerbation
display: Asthma Exacerbation
keywords:
  - asthma
  - wheezing
diagnosis:
  system: SNOMED CT
  code: "195967001"
  display: Asthma
required_artifacts:
  - structured_ehr
  - vitals
  - clinical_text
clinical_signals:
  - tachypnea
  - wheezing
  - bronchodilator treatment
red_flags:
  - hypoxemia
citations:
  - label: Global Initiative for Asthma
    url: https://ginasthma.org/
```

## Tests

Run focused tests for the files you touched, then run the pull-request tier:

```bash
python -m pytest -q tests/test_generation_recipes.py
python -m pytest -q tests/test_structured_generator.py tests/test_text_generator.py
python -m pytest -q tests/test_synthetic_validator.py tests/test_dataset_quality.py
python -m pytest -q -m "not optional_backend and not network and not slow"
```

For changes that affect release packages, also run:

```bash
casecrawler generate-release-package "mixed acute care cohort" \
  --count 5 \
  --max-validation-retries 1 \
  --output-dir .context/release-smoke \
  --format sft_jsonl

casecrawler verify-split-package .context/release-smoke
```

## Pull Request Notes

In the PR body, describe:

- What clinical behavior changed.
- Whether any new source, citation, fixture, or model profile was added.
- Which tests and smoke commands were run.
- Known limitations, especially missing artifacts or non-clinician-reviewed
  assumptions.
