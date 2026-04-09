# Multi-Step Case Architecture for AI Training Data

**Date:** 2026-04-09
**Status:** Draft
**Primary use case:** AI training (synthetic clinical reasoning episodes)
**Secondary use case:** Trainee education (interactive multi-step simulation)

## Problem

The current case generation pipeline produces single-turn, flat cases: one vignette, one decision point, one reveal. This is structurally a multiple-choice question bank, not a clinical reasoning simulation. Specific gaps:

- **No temporal evolution.** Patients don't change. Vitals don't trend. Labs don't return over time.
- **No structured diagnostics.** Lab values, imaging reports, and vitals are free-text baked into the vignette prose. No units, no reference ranges, no structured fields.
- **Binary correctness.** Every choice is correct or wrong. Real clinical decisions exist on a spectrum — suboptimal but defensible, correct but delayed, reasonable but not ideal.
- **Single decision point.** No sequential reasoning. A model can't learn to order a test, interpret the result, then decide the next step.
- **No usable training format.** No RL episodes, no SFT conversation pairs, no batch export.

## Design Decisions

Decisions made during brainstorming:

| Decision | Choice | Rationale |
|---|---|---|
| Primary use case | AI training data | Drives schema toward structured, exportable episodes over rich UI |
| Diagnostic depth | Temporal staging (Option C) | Labs, imaging, vitals arrive at different time points per phase |
| Output format | Format-agnostic with RL + SFT exporters | Don't lock into one training paradigm |
| Diagnostic sourcing | Hybrid with real data seeding | Template-constrained generation with retriever-seeded reference data |
| Migration strategy | Evolve existing model (Option B) | Codebase is young; single-step = degenerate multi-step |
| Phase count | Variable by difficulty | 3 (med student), 5 (resident), 7-10 (attending) |
| Generation approach | Plan-then-generate (Approach C) | Blueprint provides global consistency + training ground truth; phases renderable in parallel |

---

## 1. Evolved Data Model

### 1.1 Case Blueprint (Planner Output)

The blueprint is the "answer key" for the full episode — a structured skeleton generated before any detailed content.

```python
class CaseBlueprint(BaseModel):
    diagnosis: str
    clinical_arc: str              # e.g., "SAH presenting as worst headache, diagnosed via CT/LP, treated with surgical clipping"
    phase_count: int               # 3-10, driven by difficulty
    phases: list[PhaseBlueprint]
    branching_points: list[BranchPoint]
    expected_complications: list[str]

class PhaseBlueprint(BaseModel):
    phase_number: int
    time_offset: str               # "T+0", "T+30min", "T+2h"
    clinical_context: str          # what's happening at this moment
    available_diagnostics: list[str]  # results now available
    pending_diagnostics: list[str]    # ordered but not back yet
    decision_type: str             # "order_workup" | "interpret_results" | "start_treatment" | "consult" | "disposition"
    correct_action: str
    key_reasoning: str

class BranchPoint(BaseModel):
    phase_number: int
    branch_type: str               # "redirect" | "fork" | "terminal"
    trigger_action_quality: str    # which quality level triggers this branch
    description: str
```

### 1.2 Rendered Case Phase

```python
class CasePhase(BaseModel):
    phase_number: int
    time_offset: str
    narrative: str                 # prose vignette for this moment
    vitals: VitalSigns | None
    lab_results: list[LabResult]
    imaging_results: list[ImagingResult]
    clinical_update: str | None    # new symptoms, nurse pages, etc.
    decisions: list[PhaseDecision]
    phase_outcome: PhaseOutcome

class PhaseOutcome(BaseModel):
    optimal_next_phase: int | None     # default next phase on optimal path
    patient_status: str                # "stable", "deteriorating", "improving", "critical"
    narrative_transition: str          # what happens between this phase and the next
```

### 1.3 Structured Diagnostics

```python
class VitalSigns(BaseModel):
    hr: int
    bp_systolic: int
    bp_diastolic: int
    rr: int
    spo2: float
    temp_c: float
    gcs: int | None                # when neuro-relevant

class LabResult(BaseModel):
    panel: str                     # "CBC", "BMP", "coags", "ABG", "CSF", etc.
    values: list[LabValue]
    timestamp: str                 # when drawn / when resulted

class LabValue(BaseModel):
    name: str                      # "WBC", "Hgb", "Na", "K", etc.
    value: float
    unit: str                      # "K/uL", "g/dL", "mEq/L"
    reference_low: float
    reference_high: float
    flag: str | None               # "H", "L", "critical"

class ImagingResult(BaseModel):
    modality: str                  # "CT", "MRI", "XR", "US", "CTA"
    body_region: str
    indication: str                # "r/o SAH"
    findings: list[ImagingFinding]
    impression: str                # radiologist's one-liner
    timestamp: str

class ImagingFinding(BaseModel):
    structure: str                 # "basal cisterns", "sylvian fissure"
    observation: str               # "hyperdense material"
    severity: str | None           # "diffuse", "focal", "trace"
    laterality: str | None         # "left", "right", "bilateral"
```

### 1.4 Phase Decisions (5-Level Quality Grading)

Replaces the binary `is_correct` / `error_type` system.

```python
class PhaseDecision(BaseModel):
    action: str                    # "Order CT head without contrast"
    is_optimal: bool
    quality: str                   # "optimal" | "acceptable" | "suboptimal" | "harmful" | "catastrophic"
    reasoning: str                 # why someone would pick this
    clinical_outcome: str          # what happens if you pick this
    time_cost: str | None          # "delays diagnosis by ~2h"
    leads_to_phase: int | None     # which phase this branches to (if different from default next)
```

### 1.5 Evolved GeneratedCase

```python
class GeneratedCase(BaseModel):
    case_id: str
    topic: str
    difficulty: DifficultyLevel
    specialty: list[str]
    patient: Patient
    blueprint: CaseBlueprint | None = None   # None for legacy single-step cases
    phases: list[CasePhase] = []             # empty for legacy cases
    vignette: str                            # kept for backward compat
    decision_prompt: str
    ground_truth: GroundTruth
    decision_tree: list[DecisionChoice]      # kept for backward compat
    complications: list[Complication]
    review: ReviewResult | None = None
    sources: list[dict]
    metadata: dict
```

**Backward compatibility:** Legacy cases with `phases == []` work unchanged. New multi-step cases populate `vignette` (concatenated phase narratives) and `decision_tree` (phase 1 decisions mapped to old format) for backward consumers.

---

## 2. Generation Pipeline

### 2.1 Pipeline Stages

```
Retriever → Case Planner → [Blueprint Review] → Phase Renderer (×N parallel) → Consistency Pass → Clinical Reviewer → Export
```

**Stage 1: Retriever** (unchanged)

Same ChromaDB semantic search with credibility ranking. Additionally tags retrieved chunks that contain diagnostic criteria, expected lab abnormalities, and drug dosing — these become hard constraints for the phase renderer.

**Stage 2: Case Planner** (new, replaces Case Generator)

Single LLM call. Receives topic, difficulty, retrieved context. Produces `CaseBlueprint`.

Difficulty-calibrated phase counts:
- Medical student: 3 phases (present → workup → manage)
- Resident: 5 phases (present → initial workup → interpret → targeted workup → manage)
- Attending: 7-10 phases (adds reassessment, complication management, consult coordination, disposition)

The planner generates structure only — no prose, no lab values, no imaging reports.

**Stage 3: Blueprint Review** (new, lightweight gate)

Quick LLM call to validate the blueprint:
- Is the clinical arc medically sound?
- Do phases follow a logical temporal sequence?
- Are branching points at genuine decision points?
- Is phase count appropriate for difficulty?

Rejection retries the planner (max 3 attempts).

**Stage 4: Phase Renderer** (new, replaces Decision Tree Builder)

One LLM call per phase. Receives: full blueprint, current phase skeleton, retrieved context, lab panel templates with reference ranges.

Produces: `CasePhase` with narrative, structured vitals/labs/imaging, and decision options graded on the 5-level scale.

**Parallelization strategy:** Render all phases in parallel, then run a consistency pass. This is faster than sequential rendering. The blueprint provides enough constraint that phases are usually coherent without seeing each other's output.

**Stage 5: Consistency Pass** (new)

Single LLM call that receives all rendered phases and checks:
- **Vital sign continuity** — trends must be physiologically logical
- **Lab value coherence** — trending values (troponin, lactate, Hgb) must move in plausible directions
- **Temporal logic** — order-to-result times must be realistic (stat CBC ~30min, cultures ~48h)
- **Narrative continuity** — patient state must not contradict across phases
- **Decision coherence** — available actions must not include things already done

Output: `list[{phase_number, field, issue, suggested_fix}]`. Affected phases re-render with fix instructions. Max 2 consistency iterations; persistent failures reject the case.

**Stage 6: Clinical Reviewer** (evolved)

Same 3-axis scoring (accuracy, pedagogy, bias), expanded criteria:
- **Accuracy:** lab value plausibility, vital sign trends, imaging findings matching modality capabilities, diagnostic timing realism
- **Pedagogy:** phase count for difficulty, decision quality spectrum distribution, information reveal pacing
- **Bias:** unchanged

Rejection feedback targets specific failing phase(s) — only those re-render.

### 2.2 Quality Gate Summary

| Gate | What it checks | Failure action | Cost |
|---|---|---|---|
| Blueprint Review | Clinical arc, temporal logic, phase count | Retry planner (max 3) | ~500 tokens |
| Consistency Pass | Cross-phase coherence | Re-render affected phases (max 2 iterations) | ~3-7K tokens |
| Clinical Reviewer | Accuracy, pedagogy, bias | Re-render failing phases | ~3-5K tokens |

### 2.3 Token Budget Estimates

| Difficulty | Phases | Planner | Rendering | Consistency | Reviewer | Total (output) |
|---|---|---|---|---|---|---|
| Medical student | 3 | ~2K | ~6K | ~3K | ~3K | ~14K |
| Resident | 5 | ~3K | ~10K | ~5K | ~4K | ~22K |
| Attending | 8 | ~4K | ~16K | ~7K | ~5K | ~32K |

Input tokens roughly 2-3x output. Attending case total: ~100K tokens, ~$0.30-0.50 (Sonnet), ~$1-2 (Opus). At 1,000 cases: $300-2,000 depending on model and difficulty mix.

---

## 3. Diagnostic Templates & Reference Data

### 3.1 Lab Panel Templates

Static data shipped in the codebase — not generated, not retrieved. These are guardrails for the phase renderer.

```python
class LabPanel(BaseModel):
    name: str              # "CBC", "BMP", etc.
    components: list[LabComponent]

class LabComponent(BaseModel):
    name: str              # "WBC"
    unit: str              # "K/uL"
    reference_low: float   # 4.5
    reference_high: float  # 11.0
    critical_low: float | None
    critical_high: float | None
    precision: int         # decimal places
```

Panels to ship (~15):

| Panel | Key Components |
|---|---|
| CBC | WBC, Hgb, Hct, Plt, MCV, RDW |
| BMP | Na, K, Cl, CO2, BUN, Cr, Glucose, Ca |
| CMP | BMP + AST, ALT, ALP, albumin, total protein, bilirubin |
| Coags | PT, INR, PTT, fibrinogen |
| ABG | pH, pCO2, pO2, HCO3, lactate |
| CSF | WBC, RBC, protein, glucose, opening pressure |
| Troponin | troponin I or T (single value) |
| Lipase | lipase (single value) |
| D-dimer | d-dimer (single value) |
| BNP | BNP or NT-proBNP (single value) |
| UA | pH, specific gravity, leukocyte esterase, nitrites, protein, glucose, blood |
| Tox screen | qualitative + quantitative |
| Blood cultures | qualitative (pending/positive/negative + organism) |
| LFTs | AST, ALT, ALP, GGT, albumin, bilirubin (total/direct) |
| Thyroid | TSH, free T4, free T3 |
| Iron studies | serum iron, TIBC, ferritin, transferrin sat |

### 3.2 Imaging Report Constraints

Less rigid than labs — modalities vary too much for strict templates. Instead, modality-specific vocabulary constraints:

```python
class ImagingTemplate(BaseModel):
    modality: str
    valid_body_regions: list[str]
    terminology: dict[str, list[str]]  # category → valid terms, e.g. {"density": ["hyperdense", "hypodense", "isodense"]}
    report_format: str             # "findings → impression"
```

Examples:
- CT: "hyperdense", "hypodense", "isodense" — NOT "hyperintense" (that's MRI)
- MRI: "hyperintense on T2", "restricted diffusion" — NOT "hyperdense"
- XR: "opacity", "lucency", "consolidation" — NOT "enhancement"

### 3.3 Retriever-Seeded Constraints

Retrieved chunks tagged with diagnostic criteria become hard constraints in the renderer prompt:

> "The retrieved literature states SAH CT sensitivity is 95% within 6 hours. If this phase is within that window, CT findings must be positive. If beyond 7 days, CT may be negative and LP becomes the diagnostic step."

> "DKA diagnostic criteria from source: glucose >250, pH <7.3, bicarb <18, anion gap >12. Generated lab values for a DKA case must satisfy these thresholds."

---

## 4. Export Formats

### 4.1 RL Episode Format

```python
class Episode(BaseModel):
    case_id: str
    difficulty: str
    specialty: list[str]
    steps: list[TimeStep]

class TimeStep(BaseModel):
    step_number: int
    observation: Observation
    action_space: list[Action]
    optimal_action: str
    reward_table: dict[str, float]   # action_id → reward

class Observation(BaseModel):
    narrative: str
    vitals: dict | None
    new_lab_results: list[dict]
    new_imaging_results: list[dict]
    pending_orders: list[str]
    time_elapsed: str

class Action(BaseModel):
    id: str
    description: str
    quality: str
```

**Default reward mapping:**

| Quality | Reward |
|---|---|
| optimal | 1.0 |
| acceptable | 0.6 |
| suboptimal | 0.2 |
| harmful | -0.5 |
| catastrophic | -1.0 |

Custom reward mappings supported at export time.

### 4.2 SFT Conversation Format

```python
class Conversation(BaseModel):
    case_id: str
    system_prompt: str
    turns: list[Turn]

class Turn(BaseModel):
    role: str            # "system" | "assistant"
    content: str
```

Structure:
1. **system** → phase 1 narrative + vitals + labs
2. **assistant** → optimal action with clinical reasoning
3. **system** → phase 2 narrative + new results
4. **assistant** → next optimal action with reasoning
5. ...repeat through all phases

Assistant turns generated from blueprint's `correct_action` and `key_reasoning`, expanded into natural clinical reasoning prose.

**Wrong-path SFT variants:** For each case, also exportable with suboptimal assistant turns and the resulting consequences. Useful for training models to recognize and recover from errors.

### 4.3 CLI Export Commands

```bash
casecrawler export dataset.jsonl --format rl --difficulty resident --min-accuracy 0.8
casecrawler export dataset.jsonl --format sft --difficulty attending --include-wrong-paths
casecrawler export dataset.jsonl --format both --count 1000
```

**Metadata in every exported record:**
- `case_id` for traceability
- `difficulty` and `specialty` for stratified training
- `review_scores` (accuracy, pedagogy, bias) for quality filtering
- `source_credibility` — highest credibility level among sources used

---

## 5. Difficulty-Calibrated Phase Counts & Decision Complexity

### 5.1 Medical Student (3 phases)

| Phase | Content | Decisions |
|---|---|---|
| 1. Presentation | Vitals, chief complaint, focused history/exam | 2-3 options |
| 2. Initial workup | Order and interpret one diagnostic set (labs OR imaging) | 2-3 options |
| 3. Management | Start treatment or escalate | 2-3 options |

Quality distribution: 1 optimal, 1 harmful/catastrophic, 0-1 suboptimal. Clear separation.

### 5.2 Resident (5 phases)

| Phase | Content | Decisions |
|---|---|---|
| 1. Presentation | Vitals, history, exam with some missing info | 3-4 options |
| 2. Initial workup | Order labs AND imaging | 3-4 options |
| 3. Interpret results | Results return, some ambiguous | 3-4 options |
| 4. Targeted workup / early management | Refine diagnosis or start treatment | 3-4 options |
| 5. Reassess and disposition | Patient responds (or doesn't), final plan | 3-4 options |

Quality distribution: 1 optimal, 1-2 acceptable/suboptimal, 1 harmful. Nuanced distinctions.

### 5.3 Attending (7-10 phases)

| Phase | Content | Decisions |
|---|---|---|
| 1-5 | Same as resident but with more ambiguity, concurrent pathologies | 4-5 options |
| 6 | Complication management — something goes wrong even on correct path | 4-5 options |
| 7 | Consult coordination — specialist input, possibly conflicting | 4-5 options |
| 8-10 | Reassessment loops, disposition, handoff | 4-5 options |

Quality distribution: 1 optimal, 2 acceptable, 1-2 suboptimal, 0-1 catastrophic. Multiple defensible paths.

### 5.4 Branching Behavior

Three branch types based on decision quality:

- **Redirect** — suboptimal choice leads to the same next phase but with worse patient state (delayed, more unstable). Case continues but is harder. Most valuable for AI training — creates paired trajectories where the same case diverges based on decision quality.
- **Fork** — harmful choice leads to a different phase (e.g., wrong treatment triggers a complication that becomes its own management problem).
- **Terminal** — catastrophic choice ends the case early with a bad outcome. No further phases.

---

## 6. Backward Compatibility & Migration

### 6.1 Schema Evolution

New fields have defaults. Legacy cases with `phases == []` work unchanged.

For new multi-step cases, backward-compat fields are auto-populated:
- `vignette` ← concatenation of all phase narratives
- `decision_tree` ← phase 1 decisions mapped to `DecisionChoice` format

Utility: `is_multi_step(case) → bool` checks `len(case.phases) > 1`.

### 6.2 UI Compatibility

Existing `PlayCasePage` unchanged for single-step cases. New `PlayMultiStepPage` for multi-step cases:
- Shows one phase at a time
- Reveals diagnostics as they result at each phase's time offset
- Presents phase-specific decisions
- Tracks learner path through the case
- Debrief shows full trajectory vs. optimal path

Router dispatches based on `is_multi_step`.

### 6.3 Storage

No SQLite migration needed. Cases stored as JSON blobs. Old cases deserialize with new field defaults. New cases include everything.

### 6.4 CLI

```bash
# Existing command unchanged
casecrawler generate "SAH" --difficulty resident --count 10

# Multi-step generation
casecrawler generate "SAH" --difficulty resident --count 10 --multi-step

# Export (multi-step cases only)
casecrawler export dataset.jsonl --format rl --difficulty resident
```

`--multi-step` becomes default once pipeline is validated. `--single-step` flag preserved for backward compatibility.
