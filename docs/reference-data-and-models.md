# Reference Data And Model Adapters

CaseCrawler can run fully offline, but it is designed to pull in stronger models and reference datasets when available.

## Reference Catalog

```bash
casecrawler reference-datasets
casecrawler datasets capabilities
```

Registered reference sources include:

- Synthea FHIR/NDJSON/CSV imports
- Asclepius synthetic clinical notes
- SynthClinicalNotes
- augmented clinical notes
- MedSynth dialogue-note pairs
- BeTraC/Synth-DoPaCo doctor-patient transcript to SOAP rows
- clinical-note-to-FHIR references
- radiology report consistency references
- SynthCheX-75K-v2 image/text references
- synthetic chest X-ray pneumonia references
- Technetium-I de-identification and ICD-coding references
- bundled offline fixtures for smoke tests

## Import Hugging Face Or Local References

```bash
casecrawler import-reference-dataset asclepius \
  --dataset-id ds-asclepius-ref \
  --limit 100

casecrawler import-reference-dataset \
  --repo-id org/custom-synthetic-notes \
  --dataset-id ds-custom-ref \
  --note-field clinical_note \
  --question-field prompt \
  --answer-field completion \
  --split eval \
  --limit 100

casecrawler import-reference-dataset local-validation-notes \
  --path ./validation/local-notes.jsonl \
  --dataset-id ds-local-ref \
  --note-field clinical_note \
  --question-field prompt \
  --answer-field completion \
  --limit 100
```

Local imports support JSONL, NDJSON, JSON arrays, and `{"rows": [...]}`.

## Synthea

```bash
casecrawler import-synthea ./synthea/output/fhir --dataset-id ds-synthea-ref

casecrawler run-synthea \
  --synthea-executable ./synthea/run_synthea \
  --output-dir ./synthea/output/fhir \
  --dataset-id ds-synthea-ref \
  --population 100
```

Synthea imports accept FHIR JSON bundles, Bulk FHIR NDJSON resource directories, and Synthea CSV directories.

## Imaging

```bash
casecrawler imaging-models
```

Imaging profiles include Prompt2MedImage, MediSyn, CheXGenBench Sana, chest X-ray Stable Diffusion and DreamBooth profiles, symptom X-ray LoRA, and gated RoentGen profiles. Each profile exposes model id, license, gated status, use policy, prompt contract, output contract, and validation requirements.

Supported backends:

- `placeholder`: deterministic local PNG for offline tests
- `diffusers`: local Hugging Face/diffusers image generation
- `external`: command adapter returning an `ImagingAsset`

## Clinical Text

```bash
casecrawler clinical-text-models
```

Clinical text adapters include deterministic templates, LLM provider routing, MedGemma, Meditron, and generic external command profiles. Noisy text variants can use `standard`, `message`, `ocr`, or `heavy` profiles.

## Time Series

```bash
casecrawler timeseries-models
```

Time-series profiles include deterministic generation plus external-command profiles for TimeDiff, RawMed-style workflows, and MIRA-style generation or validation. External adapters return `TimeSeriesChannel[]`.

## Image Validators

Image/report validation includes deterministic lexical checks and optional BiomedCLIP or MedGemma image-text validators when the relevant extras and model access are available.
