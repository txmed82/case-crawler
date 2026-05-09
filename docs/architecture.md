# Architecture

CaseCrawler is organized as a dataset-first synthetic healthcare platform.

```text
Topic / GenerationRequest / Recipe
        |
        v
Modality plan
        |
        v
Structured generator -> patient, encounters, diagnoses, meds, allergies, orders, labs, vitals
        |
        v
Text generator       -> physician notes, nursing notes, lab reports, MAR, discharge, messy variants
        |
        v
Time-series generator -> longitudinal vitals/labs, ECG lead II, pleth-like channels
        |
        v
Imaging generator    -> radiology reports, image assets, backend provenance, file metadata
        |
        v
Synthetic validator  -> schema, clinical rules, privacy, utility, image/report alignment
        |
        v
DatasetStore         -> SQLite manifests, records, review decisions, export manifests
        |
        v
Exporters            -> raw, SFT, chat, tool, multimodal, time-series, DPO/RL, FHIR, parquet
```

## Main Modules

- `src/casecrawler/models/synthetic.py`: multimodal `SyntheticRecord` schema.
- `src/casecrawler/models/dataset.py`: generation requests, manifests, export formats, and human review decisions.
- `src/casecrawler/generation/`: deterministic generators plus clinical text, imaging, and time-series backend adapters.
- `src/casecrawler/validation/`: clinical rules, privacy checks, image alignment, quality reports, and benchmark comparisons.
- `src/casecrawler/integrations/`: Hugging Face reference imports and Synthea imports/runners.
- `src/casecrawler/export/`: fine-tuning exports, split packages, dataset/model cards, and objective coverage audits.
- `src/casecrawler/storage/`: SQLite dataset and review storage.
- `src/casecrawler/api/`: FastAPI routes.
- `ui/`: React/Vite dataset workbench.

## Backend Contracts

External model integrations are wrapped behind stable contracts:

- Clinical text command adapters receive record context on stdin and return `ClinicalDocument[]`.
- Imaging command adapters receive prompt, output directory, modality, and body region, then return an `ImagingAsset`.
- Time-series command adapters receive record/channel context and return `TimeSeriesChannel[]`.
- Diffusers imaging profiles and Hugging Face references declare license, gated status, use policy, input contract, output contract, and validation requirements.

The default deterministic path remains available for CI, local development, and reproducible smoke tests.
