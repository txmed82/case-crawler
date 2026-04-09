# Multi-Step Case Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve the single-turn case generation pipeline into a multi-phase, temporally staged system with structured diagnostics, plan-then-generate architecture, and AI training data export.

**Architecture:** A Case Planner generates a structured blueprint (clinical arc, phases, branching points), then a Phase Renderer generates each phase in parallel with structured vitals/labs/imaging constrained by static lab panel templates and retriever-seeded reference data. A Consistency Pass checks cross-phase coherence before the existing Clinical Reviewer scores the full case. Exporters serialize cases into RL episode and SFT conversation formats.

**Tech Stack:** Python 3.12, Pydantic v2, asyncio, pytest, Click CLI, FastAPI, SQLite, ChromaDB

---

## File Structure

### New files to create:
| File | Responsibility |
|---|---|
| `src/casecrawler/models/diagnostics.py` | VitalSigns, LabValue, LabResult, ImagingFinding, ImagingResult models |
| `src/casecrawler/models/blueprint.py` | CaseBlueprint, PhaseBlueprint, BranchPoint models |
| `src/casecrawler/models/phase.py` | CasePhase, PhaseDecision, PhaseOutcome models |
| `src/casecrawler/models/export.py` | Episode, TimeStep, Observation, Action, Conversation, Turn models |
| `src/casecrawler/generation/lab_panels.py` | Static lab panel template data (LabPanel, LabComponent) |
| `src/casecrawler/generation/imaging_templates.py` | Static imaging modality vocabulary constraints |
| `src/casecrawler/generation/case_planner.py` | CasePlannerAgent — generates CaseBlueprint |
| `src/casecrawler/generation/blueprint_reviewer.py` | BlueprintReviewerAgent — lightweight blueprint validation |
| `src/casecrawler/generation/phase_renderer.py` | PhaseRendererAgent — renders one CasePhase from PhaseBlueprint |
| `src/casecrawler/generation/consistency_checker.py` | ConsistencyCheckerAgent — cross-phase coherence validation |
| `src/casecrawler/generation/multi_step_pipeline.py` | MultiStepPipeline — orchestrates plan-then-generate flow |
| `src/casecrawler/export/rl_exporter.py` | Export cases to RL episode format |
| `src/casecrawler/export/sft_exporter.py` | Export cases to SFT conversation format |
| `src/casecrawler/export/__init__.py` | Package init |
| `tests/test_diagnostics_models.py` | Tests for diagnostic models |
| `tests/test_blueprint_models.py` | Tests for blueprint models |
| `tests/test_phase_models.py` | Tests for phase/decision models |
| `tests/test_lab_panels.py` | Tests for lab panel templates |
| `tests/test_imaging_templates.py` | Tests for imaging templates |
| `tests/test_case_planner.py` | Tests for CasePlannerAgent |
| `tests/test_blueprint_reviewer.py` | Tests for BlueprintReviewerAgent |
| `tests/test_phase_renderer.py` | Tests for PhaseRendererAgent |
| `tests/test_consistency_checker.py` | Tests for ConsistencyCheckerAgent |
| `tests/test_multi_step_pipeline.py` | Tests for MultiStepPipeline |
| `tests/test_rl_exporter.py` | Tests for RL export |
| `tests/test_sft_exporter.py` | Tests for SFT export |
| `tests/test_export_models.py` | Tests for export data models |

### Existing files to modify:
| File | Changes |
|---|---|
| `src/casecrawler/models/case.py` | Add `blueprint` and `phases` fields to GeneratedCase, add `is_multi_step()` |
| `src/casecrawler/generation/prompts.py` | Add planner, blueprint reviewer, phase renderer, and consistency prompts; extend DIFFICULTY_RULES with phase counts |
| `src/casecrawler/cli.py` | Add `--multi-step` flag to `generate`, add `export` command |
| `src/casecrawler/api/routes/generate.py` | Add `multi_step` field to GenerateRequest |
| `src/casecrawler/api/routes/cases.py` | Add export endpoint with format param |
| `tests/test_case_models.py` | Add tests for evolved GeneratedCase with phases |
| `tests/test_generation_pipeline.py` | Verify existing pipeline still works unchanged |

---

### Task 1: Structured Diagnostic Models

**Files:**
- Create: `src/casecrawler/models/diagnostics.py`
- Test: `tests/test_diagnostics_models.py`

- [ ] **Step 1: Write failing tests for VitalSigns**

```python
# tests/test_diagnostics_models.py
from casecrawler.models.diagnostics import (
    ImagingFinding,
    ImagingResult,
    LabResult,
    LabValue,
    VitalSigns,
)


def test_vital_signs_basic():
    vs = VitalSigns(
        hr=88, bp_systolic=142, bp_diastolic=88, rr=18, spo2=97.0, temp_c=37.2, gcs=15,
    )
    assert vs.hr == 88
    assert vs.gcs == 15


def test_vital_signs_gcs_optional():
    vs = VitalSigns(
        hr=72, bp_systolic=120, bp_diastolic=80, rr=16, spo2=99.0, temp_c=36.8,
    )
    assert vs.gcs is None


def test_lab_value_flagged_high():
    lv = LabValue(
        name="WBC", value=18.5, unit="K/uL",
        reference_low=4.5, reference_high=11.0, flag="H",
    )
    assert lv.flag == "H"
    assert lv.value > lv.reference_high


def test_lab_value_no_flag():
    lv = LabValue(
        name="Na", value=140.0, unit="mEq/L",
        reference_low=136.0, reference_high=145.0, flag=None,
    )
    assert lv.flag is None


def test_lab_result_panel():
    lr = LabResult(
        panel="CBC",
        values=[
            LabValue(name="WBC", value=7.2, unit="K/uL", reference_low=4.5, reference_high=11.0, flag=None),
            LabValue(name="Hgb", value=13.5, unit="g/dL", reference_low=12.0, reference_high=16.0, flag=None),
        ],
        timestamp="T+30min",
    )
    assert lr.panel == "CBC"
    assert len(lr.values) == 2


def test_imaging_finding():
    f = ImagingFinding(
        structure="basal cisterns",
        observation="hyperdense material",
        severity="diffuse",
        laterality=None,
    )
    assert f.structure == "basal cisterns"
    assert f.laterality is None


def test_imaging_result():
    ir = ImagingResult(
        modality="CT",
        body_region="head",
        indication="r/o SAH",
        findings=[
            ImagingFinding(
                structure="basal cisterns",
                observation="hyperdense material",
                severity="diffuse",
                laterality="bilateral",
            ),
        ],
        impression="Acute subarachnoid hemorrhage",
        timestamp="T+45min",
    )
    assert ir.modality == "CT"
    assert len(ir.findings) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_diagnostics_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'casecrawler.models.diagnostics'`

- [ ] **Step 3: Implement diagnostic models**

```python
# src/casecrawler/models/diagnostics.py
from __future__ import annotations

from pydantic import BaseModel


class VitalSigns(BaseModel):
    hr: int
    bp_systolic: int
    bp_diastolic: int
    rr: int
    spo2: float
    temp_c: float
    gcs: int | None = None


class LabValue(BaseModel):
    name: str
    value: float
    unit: str
    reference_low: float
    reference_high: float
    flag: str | None = None


class LabResult(BaseModel):
    panel: str
    values: list[LabValue]
    timestamp: str


class ImagingFinding(BaseModel):
    structure: str
    observation: str
    severity: str | None = None
    laterality: str | None = None


class ImagingResult(BaseModel):
    modality: str
    body_region: str
    indication: str
    findings: list[ImagingFinding]
    impression: str
    timestamp: str
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_diagnostics_models.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/models/diagnostics.py tests/test_diagnostics_models.py
git commit -m "feat: add structured diagnostic models (VitalSigns, LabResult, ImagingResult)"
```

---

### Task 2: Blueprint & Phase Models

**Files:**
- Create: `src/casecrawler/models/blueprint.py`
- Create: `src/casecrawler/models/phase.py`
- Test: `tests/test_blueprint_models.py`
- Test: `tests/test_phase_models.py`

- [ ] **Step 1: Write failing tests for blueprint models**

```python
# tests/test_blueprint_models.py
from casecrawler.models.blueprint import BranchPoint, CaseBlueprint, PhaseBlueprint


def test_phase_blueprint():
    pb = PhaseBlueprint(
        phase_number=1,
        time_offset="T+0",
        clinical_context="Patient presents to ED with thunderclap headache",
        available_diagnostics=[],
        pending_diagnostics=[],
        decision_type="order_workup",
        correct_action="Order non-contrast CT head",
        key_reasoning="CT is 95% sensitive for SAH within 6 hours",
    )
    assert pb.phase_number == 1
    assert pb.decision_type == "order_workup"


def test_branch_point():
    bp = BranchPoint(
        phase_number=2,
        branch_type="redirect",
        trigger_action_quality="suboptimal",
        description="Ordering MRI instead of CT delays diagnosis by ~2 hours",
    )
    assert bp.branch_type == "redirect"


def test_case_blueprint_full():
    phases = [
        PhaseBlueprint(
            phase_number=1, time_offset="T+0",
            clinical_context="ED presentation",
            available_diagnostics=[], pending_diagnostics=[],
            decision_type="order_workup",
            correct_action="Order CT head",
            key_reasoning="Most sensitive initial test",
        ),
        PhaseBlueprint(
            phase_number=2, time_offset="T+45min",
            clinical_context="CT results available",
            available_diagnostics=["CT head"],
            pending_diagnostics=["CBC", "BMP"],
            decision_type="interpret_results",
            correct_action="Consult neurosurgery",
            key_reasoning="CT confirms SAH, needs surgical evaluation",
        ),
        PhaseBlueprint(
            phase_number=3, time_offset="T+2h",
            clinical_context="Neurosurgery consulted, labs back",
            available_diagnostics=["CT head", "CBC", "BMP"],
            pending_diagnostics=[],
            decision_type="start_treatment",
            correct_action="Nimodipine and ICU admission",
            key_reasoning="Prevent vasospasm, close monitoring",
        ),
    ]
    bp = CaseBlueprint(
        diagnosis="Aneurysmal subarachnoid hemorrhage",
        clinical_arc="SAH presenting as thunderclap headache, diagnosed via CT, surgical management",
        phase_count=3,
        phases=phases,
        branching_points=[
            BranchPoint(
                phase_number=1, branch_type="terminal",
                trigger_action_quality="catastrophic",
                description="Discharge leads to rebleed and death",
            ),
        ],
        expected_complications=["vasospasm", "rebleeding", "hydrocephalus"],
    )
    assert bp.phase_count == 3
    assert len(bp.phases) == 3
    assert len(bp.branching_points) == 1
    assert len(bp.expected_complications) == 3
```

- [ ] **Step 2: Write failing tests for phase models**

```python
# tests/test_phase_models.py
from casecrawler.models.diagnostics import (
    ImagingFinding,
    ImagingResult,
    LabResult,
    LabValue,
    VitalSigns,
)
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome


def test_phase_decision_optimal():
    pd = PhaseDecision(
        action="Order non-contrast CT head",
        is_optimal=True,
        quality="optimal",
        reasoning="Most sensitive initial test for SAH",
        clinical_outcome="CT shows hyperdense material in basal cisterns",
        time_cost=None,
        leads_to_phase=None,
    )
    assert pd.is_optimal is True
    assert pd.quality == "optimal"


def test_phase_decision_suboptimal():
    pd = PhaseDecision(
        action="Order MRI brain",
        is_optimal=False,
        quality="suboptimal",
        reasoning="MRI can detect SAH but takes longer and is less available in ED",
        clinical_outcome="Delayed diagnosis by ~2 hours, MRI eventually shows SAH",
        time_cost="delays diagnosis by ~2h",
        leads_to_phase=None,
    )
    assert pd.quality == "suboptimal"
    assert pd.time_cost is not None


def test_phase_decision_catastrophic():
    pd = PhaseDecision(
        action="Discharge with migraine diagnosis",
        is_optimal=False,
        quality="catastrophic",
        reasoning="Misidentified as primary headache",
        clinical_outcome="Patient rebleeds at home within 24 hours",
        time_cost=None,
        leads_to_phase=None,
    )
    assert pd.quality == "catastrophic"


def test_phase_outcome():
    po = PhaseOutcome(
        optimal_next_phase=2,
        patient_status="stable",
        narrative_transition="CT ordered stat. Patient resting in resus bay.",
    )
    assert po.optimal_next_phase == 2
    assert po.patient_status == "stable"


def test_phase_outcome_terminal():
    po = PhaseOutcome(
        optimal_next_phase=None,
        patient_status="critical",
        narrative_transition="Patient discharged against medical advice. Found unresponsive at home 6 hours later.",
    )
    assert po.optimal_next_phase is None


def test_case_phase_full():
    phase = CasePhase(
        phase_number=1,
        time_offset="T+0",
        narrative="A 42-year-old woman presents to the ED with sudden onset severe headache.",
        vitals=VitalSigns(
            hr=92, bp_systolic=168, bp_diastolic=95, rr=20, spo2=98.0, temp_c=37.4, gcs=14,
        ),
        lab_results=[],
        imaging_results=[],
        clinical_update=None,
        decisions=[
            PhaseDecision(
                action="Order non-contrast CT head",
                is_optimal=True, quality="optimal",
                reasoning="Most sensitive", clinical_outcome="SAH confirmed",
                time_cost=None, leads_to_phase=None,
            ),
            PhaseDecision(
                action="Discharge with migraine diagnosis",
                is_optimal=False, quality="catastrophic",
                reasoning="Misdiagnosis", clinical_outcome="Rebleed at home",
                time_cost=None, leads_to_phase=None,
            ),
        ],
        phase_outcome=PhaseOutcome(
            optimal_next_phase=2,
            patient_status="stable",
            narrative_transition="CT ordered stat.",
        ),
    )
    assert phase.phase_number == 1
    assert len(phase.decisions) == 2
    assert phase.vitals.hr == 92
    assert phase.phase_outcome.optimal_next_phase == 2


def test_case_phase_with_diagnostics():
    phase = CasePhase(
        phase_number=2,
        time_offset="T+45min",
        narrative="CT results are back. Labs still pending.",
        vitals=VitalSigns(
            hr=88, bp_systolic=155, bp_diastolic=90, rr=18, spo2=98.0, temp_c=37.2, gcs=14,
        ),
        lab_results=[
            LabResult(
                panel="CBC",
                values=[
                    LabValue(name="WBC", value=9.8, unit="K/uL", reference_low=4.5, reference_high=11.0, flag=None),
                    LabValue(name="Hgb", value=13.2, unit="g/dL", reference_low=12.0, reference_high=16.0, flag=None),
                    LabValue(name="Plt", value=245.0, unit="K/uL", reference_low=150.0, reference_high=400.0, flag=None),
                ],
                timestamp="T+40min",
            ),
        ],
        imaging_results=[
            ImagingResult(
                modality="CT", body_region="head", indication="r/o SAH",
                findings=[
                    ImagingFinding(
                        structure="basal cisterns",
                        observation="hyperdense material",
                        severity="diffuse",
                        laterality="bilateral",
                    ),
                ],
                impression="Acute subarachnoid hemorrhage, Fisher grade 3",
                timestamp="T+35min",
            ),
        ],
        clinical_update="Nurse reports patient is photophobic and increasingly drowsy.",
        decisions=[
            PhaseDecision(
                action="Consult neurosurgery stat",
                is_optimal=True, quality="optimal",
                reasoning="CT-confirmed SAH requires urgent neurosurgical evaluation",
                clinical_outcome="Neurosurgery evaluates within 30 minutes",
                time_cost=None, leads_to_phase=None,
            ),
        ],
        phase_outcome=PhaseOutcome(
            optimal_next_phase=3,
            patient_status="stable",
            narrative_transition="Neurosurgery consulted. CTA ordered to identify aneurysm.",
        ),
    )
    assert len(phase.lab_results) == 1
    assert len(phase.imaging_results) == 1
    assert phase.clinical_update is not None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_blueprint_models.py tests/test_phase_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement blueprint models**

```python
# src/casecrawler/models/blueprint.py
from __future__ import annotations

from pydantic import BaseModel


class PhaseBlueprint(BaseModel):
    phase_number: int
    time_offset: str
    clinical_context: str
    available_diagnostics: list[str]
    pending_diagnostics: list[str]
    decision_type: str
    correct_action: str
    key_reasoning: str


class BranchPoint(BaseModel):
    phase_number: int
    branch_type: str
    trigger_action_quality: str
    description: str


class CaseBlueprint(BaseModel):
    diagnosis: str
    clinical_arc: str
    phase_count: int
    phases: list[PhaseBlueprint]
    branching_points: list[BranchPoint]
    expected_complications: list[str]
```

- [ ] **Step 5: Implement phase models**

```python
# src/casecrawler/models/phase.py
from __future__ import annotations

from pydantic import BaseModel

from casecrawler.models.diagnostics import ImagingResult, LabResult, VitalSigns


class PhaseDecision(BaseModel):
    action: str
    is_optimal: bool
    quality: str
    reasoning: str
    clinical_outcome: str
    time_cost: str | None = None
    leads_to_phase: int | None = None


class PhaseOutcome(BaseModel):
    optimal_next_phase: int | None
    patient_status: str
    narrative_transition: str


class CasePhase(BaseModel):
    phase_number: int
    time_offset: str
    narrative: str
    vitals: VitalSigns | None = None
    lab_results: list[LabResult] = []
    imaging_results: list[ImagingResult] = []
    clinical_update: str | None = None
    decisions: list[PhaseDecision] = []
    phase_outcome: PhaseOutcome
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_blueprint_models.py tests/test_phase_models.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/casecrawler/models/blueprint.py src/casecrawler/models/phase.py tests/test_blueprint_models.py tests/test_phase_models.py
git commit -m "feat: add blueprint and phase models for multi-step cases"
```

---

### Task 3: Evolve GeneratedCase Model

**Files:**
- Modify: `src/casecrawler/models/case.py`
- Modify: `tests/test_case_models.py`

- [ ] **Step 1: Write failing tests for evolved GeneratedCase**

Add to `tests/test_case_models.py`:

```python
from casecrawler.models.blueprint import BranchPoint, CaseBlueprint, PhaseBlueprint
from casecrawler.models.diagnostics import LabResult, LabValue, VitalSigns
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome


def test_generated_case_legacy_compat():
    """Existing cases with no phases or blueprint still work."""
    case = GeneratedCase(
        case_id="legacy-1",
        topic="SAH",
        difficulty=DifficultyLevel.RESIDENT,
        specialty=["neurosurgery"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        vignette="A 42-year-old woman presents...",
        decision_prompt="What would you do next?",
        ground_truth=GroundTruth(
            diagnosis="SAH", optimal_next_step="CT head",
            rationale="Most sensitive", key_findings=["thunderclap headache"],
        ),
        decision_tree=[
            DecisionChoice(
                choice="CT head", is_correct=True, error_type=None,
                reasoning="Correct", outcome="Confirmed",
                consequence=None, next_decision=None,
            ),
        ],
        complications=[],
        sources=[],
        metadata={},
    )
    assert case.blueprint is None
    assert case.phases == []
    assert case.is_multi_step() is False


def test_generated_case_multi_step():
    """New multi-step cases include blueprint and phases."""
    blueprint = CaseBlueprint(
        diagnosis="SAH",
        clinical_arc="SAH via CT, surgical management",
        phase_count=2,
        phases=[
            PhaseBlueprint(
                phase_number=1, time_offset="T+0",
                clinical_context="ED presentation",
                available_diagnostics=[], pending_diagnostics=[],
                decision_type="order_workup",
                correct_action="Order CT head",
                key_reasoning="Most sensitive",
            ),
            PhaseBlueprint(
                phase_number=2, time_offset="T+45min",
                clinical_context="CT results available",
                available_diagnostics=["CT head"],
                pending_diagnostics=[],
                decision_type="start_treatment",
                correct_action="Consult neurosurgery",
                key_reasoning="CT confirms SAH",
            ),
        ],
        branching_points=[],
        expected_complications=["vasospasm"],
    )
    phases = [
        CasePhase(
            phase_number=1, time_offset="T+0",
            narrative="A 42-year-old woman presents with thunderclap headache.",
            vitals=VitalSigns(hr=92, bp_systolic=168, bp_diastolic=95, rr=20, spo2=98.0, temp_c=37.4),
            lab_results=[], imaging_results=[],
            decisions=[
                PhaseDecision(
                    action="Order CT head", is_optimal=True, quality="optimal",
                    reasoning="Most sensitive", clinical_outcome="SAH confirmed",
                ),
            ],
            phase_outcome=PhaseOutcome(optimal_next_phase=2, patient_status="stable", narrative_transition="CT ordered."),
        ),
        CasePhase(
            phase_number=2, time_offset="T+45min",
            narrative="CT results show hyperdense material in basal cisterns.",
            vitals=VitalSigns(hr=88, bp_systolic=155, bp_diastolic=90, rr=18, spo2=98.0, temp_c=37.2),
            lab_results=[], imaging_results=[],
            decisions=[
                PhaseDecision(
                    action="Consult neurosurgery", is_optimal=True, quality="optimal",
                    reasoning="CT confirms SAH", clinical_outcome="Neurosurgery evaluates",
                ),
            ],
            phase_outcome=PhaseOutcome(optimal_next_phase=None, patient_status="stable", narrative_transition="Admitted to ICU."),
        ),
    ]
    case = GeneratedCase(
        case_id="multi-1",
        topic="SAH",
        difficulty=DifficultyLevel.RESIDENT,
        specialty=["neurosurgery"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        vignette="A 42-year-old woman presents with thunderclap headache.",
        decision_prompt="What would you do next?",
        ground_truth=GroundTruth(
            diagnosis="SAH", optimal_next_step="CT head",
            rationale="Most sensitive", key_findings=["thunderclap headache"],
        ),
        decision_tree=[],
        complications=[],
        blueprint=blueprint,
        phases=phases,
        sources=[],
        metadata={},
    )
    assert case.is_multi_step() is True
    assert case.blueprint is not None
    assert len(case.phases) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_case_models.py::test_generated_case_legacy_compat tests/test_case_models.py::test_generated_case_multi_step -v`
Expected: FAIL (missing `blueprint`, `phases` fields and `is_multi_step` method)

- [ ] **Step 3: Add blueprint and phases fields to GeneratedCase**

In `src/casecrawler/models/case.py`, add imports and new fields:

```python
# Add to imports at top
from casecrawler.models.blueprint import CaseBlueprint
from casecrawler.models.phase import CasePhase
```

Add to `GeneratedCase` class, after `complications`:

```python
    blueprint: CaseBlueprint | None = None
    phases: list[CasePhase] = []
```

Add method to `GeneratedCase`:

```python
    def is_multi_step(self) -> bool:
        return len(self.phases) > 1
```

- [ ] **Step 4: Run all case model tests to verify pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_case_models.py -v`
Expected: All tests PASS (old and new)

- [ ] **Step 5: Run existing pipeline tests to verify backward compat**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_generation_pipeline.py -v`
Expected: All 3 existing tests PASS unchanged

- [ ] **Step 6: Commit**

```bash
git add src/casecrawler/models/case.py tests/test_case_models.py
git commit -m "feat: add blueprint and phases fields to GeneratedCase, add is_multi_step()"
```

---

### Task 4: Lab Panel Templates

**Files:**
- Create: `src/casecrawler/generation/lab_panels.py`
- Test: `tests/test_lab_panels.py`

- [ ] **Step 1: Write failing tests for lab panel data**

```python
# tests/test_lab_panels.py
from casecrawler.generation.lab_panels import LAB_PANELS, LabComponent, LabPanel, get_panel


def test_lab_panel_structure():
    panel = LAB_PANELS["CBC"]
    assert isinstance(panel, LabPanel)
    assert panel.name == "CBC"
    assert len(panel.components) >= 4


def test_lab_component_has_ranges():
    panel = LAB_PANELS["BMP"]
    na = next(c for c in panel.components if c.name == "Na")
    assert na.unit == "mEq/L"
    assert na.reference_low == 136.0
    assert na.reference_high == 145.0
    assert na.precision == 0


def test_all_panels_present():
    expected = [
        "CBC", "BMP", "CMP", "coags", "ABG", "CSF", "troponin",
        "lipase", "d_dimer", "BNP", "UA", "LFTs", "thyroid", "iron_studies",
    ]
    for name in expected:
        assert name in LAB_PANELS, f"Missing panel: {name}"


def test_get_panel_found():
    panel = get_panel("CBC")
    assert panel is not None
    assert panel.name == "CBC"


def test_get_panel_not_found():
    panel = get_panel("nonexistent")
    assert panel is None


def test_critical_ranges_present():
    """Critical values should be set for dangerous analytes."""
    bmp = LAB_PANELS["BMP"]
    k = next(c for c in bmp.components if c.name == "K")
    assert k.critical_low is not None
    assert k.critical_high is not None


def test_cmp_contains_bmp_components():
    """CMP should include all BMP components plus hepatic panel."""
    bmp_names = {c.name for c in LAB_PANELS["BMP"].components}
    cmp_names = {c.name for c in LAB_PANELS["CMP"].components}
    assert bmp_names.issubset(cmp_names)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_lab_panels.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement lab panel templates**

```python
# src/casecrawler/generation/lab_panels.py
from __future__ import annotations

from pydantic import BaseModel


class LabComponent(BaseModel):
    name: str
    unit: str
    reference_low: float
    reference_high: float
    critical_low: float | None = None
    critical_high: float | None = None
    precision: int = 1


class LabPanel(BaseModel):
    name: str
    components: list[LabComponent]


def get_panel(name: str) -> LabPanel | None:
    return LAB_PANELS.get(name)


_CBC_COMPONENTS = [
    LabComponent(name="WBC", unit="K/uL", reference_low=4.5, reference_high=11.0, critical_low=2.0, critical_high=30.0, precision=1),
    LabComponent(name="Hgb", unit="g/dL", reference_low=12.0, reference_high=16.0, critical_low=7.0, critical_high=20.0, precision=1),
    LabComponent(name="Hct", unit="%", reference_low=36.0, reference_high=46.0, precision=1),
    LabComponent(name="Plt", unit="K/uL", reference_low=150.0, reference_high=400.0, critical_low=50.0, critical_high=1000.0, precision=0),
    LabComponent(name="MCV", unit="fL", reference_low=80.0, reference_high=100.0, precision=1),
    LabComponent(name="RDW", unit="%", reference_low=11.5, reference_high=14.5, precision=1),
]

_BMP_COMPONENTS = [
    LabComponent(name="Na", unit="mEq/L", reference_low=136.0, reference_high=145.0, critical_low=120.0, critical_high=160.0, precision=0),
    LabComponent(name="K", unit="mEq/L", reference_low=3.5, reference_high=5.0, critical_low=2.5, critical_high=6.5, precision=1),
    LabComponent(name="Cl", unit="mEq/L", reference_low=98.0, reference_high=106.0, precision=0),
    LabComponent(name="CO2", unit="mEq/L", reference_low=23.0, reference_high=29.0, precision=0),
    LabComponent(name="BUN", unit="mg/dL", reference_low=7.0, reference_high=20.0, precision=0),
    LabComponent(name="Cr", unit="mg/dL", reference_low=0.7, reference_high=1.3, critical_high=10.0, precision=2),
    LabComponent(name="Glucose", unit="mg/dL", reference_low=70.0, reference_high=100.0, critical_low=40.0, critical_high=500.0, precision=0),
    LabComponent(name="Ca", unit="mg/dL", reference_low=8.5, reference_high=10.5, critical_low=6.0, critical_high=13.0, precision=1),
]

_HEPATIC_COMPONENTS = [
    LabComponent(name="AST", unit="U/L", reference_low=10.0, reference_high=40.0, precision=0),
    LabComponent(name="ALT", unit="U/L", reference_low=7.0, reference_high=56.0, precision=0),
    LabComponent(name="ALP", unit="U/L", reference_low=44.0, reference_high=147.0, precision=0),
    LabComponent(name="Albumin", unit="g/dL", reference_low=3.5, reference_high=5.0, precision=1),
    LabComponent(name="Total Protein", unit="g/dL", reference_low=6.0, reference_high=8.3, precision=1),
    LabComponent(name="Total Bilirubin", unit="mg/dL", reference_low=0.1, reference_high=1.2, precision=1),
]

LAB_PANELS: dict[str, LabPanel] = {
    "CBC": LabPanel(name="CBC", components=_CBC_COMPONENTS),
    "BMP": LabPanel(name="BMP", components=_BMP_COMPONENTS),
    "CMP": LabPanel(name="CMP", components=_BMP_COMPONENTS + _HEPATIC_COMPONENTS),
    "coags": LabPanel(name="coags", components=[
        LabComponent(name="PT", unit="sec", reference_low=11.0, reference_high=13.5, precision=1),
        LabComponent(name="INR", unit="", reference_low=0.8, reference_high=1.1, critical_high=5.0, precision=1),
        LabComponent(name="PTT", unit="sec", reference_low=25.0, reference_high=35.0, critical_high=100.0, precision=1),
        LabComponent(name="Fibrinogen", unit="mg/dL", reference_low=200.0, reference_high=400.0, critical_low=100.0, precision=0),
    ]),
    "ABG": LabPanel(name="ABG", components=[
        LabComponent(name="pH", unit="", reference_low=7.35, reference_high=7.45, critical_low=7.1, critical_high=7.6, precision=2),
        LabComponent(name="pCO2", unit="mmHg", reference_low=35.0, reference_high=45.0, precision=0),
        LabComponent(name="pO2", unit="mmHg", reference_low=80.0, reference_high=100.0, critical_low=40.0, precision=0),
        LabComponent(name="HCO3", unit="mEq/L", reference_low=22.0, reference_high=26.0, precision=0),
        LabComponent(name="Lactate", unit="mmol/L", reference_low=0.5, reference_high=2.0, critical_high=4.0, precision=1),
    ]),
    "CSF": LabPanel(name="CSF", components=[
        LabComponent(name="WBC", unit="cells/uL", reference_low=0.0, reference_high=5.0, precision=0),
        LabComponent(name="RBC", unit="cells/uL", reference_low=0.0, reference_high=0.0, precision=0),
        LabComponent(name="Protein", unit="mg/dL", reference_low=15.0, reference_high=45.0, precision=0),
        LabComponent(name="Glucose", unit="mg/dL", reference_low=40.0, reference_high=70.0, precision=0),
        LabComponent(name="Opening Pressure", unit="cmH2O", reference_low=6.0, reference_high=20.0, critical_high=30.0, precision=0),
    ]),
    "troponin": LabPanel(name="troponin", components=[
        LabComponent(name="Troponin I", unit="ng/mL", reference_low=0.0, reference_high=0.04, critical_high=0.4, precision=3),
    ]),
    "lipase": LabPanel(name="lipase", components=[
        LabComponent(name="Lipase", unit="U/L", reference_low=0.0, reference_high=160.0, precision=0),
    ]),
    "d_dimer": LabPanel(name="d_dimer", components=[
        LabComponent(name="D-dimer", unit="ng/mL", reference_low=0.0, reference_high=500.0, precision=0),
    ]),
    "BNP": LabPanel(name="BNP", components=[
        LabComponent(name="BNP", unit="pg/mL", reference_low=0.0, reference_high=100.0, precision=0),
    ]),
    "UA": LabPanel(name="UA", components=[
        LabComponent(name="pH", unit="", reference_low=4.5, reference_high=8.0, precision=1),
        LabComponent(name="Specific Gravity", unit="", reference_low=1.005, reference_high=1.030, precision=3),
        LabComponent(name="Leukocyte Esterase", unit="", reference_low=0.0, reference_high=0.0, precision=0),
        LabComponent(name="Nitrites", unit="", reference_low=0.0, reference_high=0.0, precision=0),
        LabComponent(name="Protein", unit="mg/dL", reference_low=0.0, reference_high=14.0, precision=0),
        LabComponent(name="Glucose", unit="mg/dL", reference_low=0.0, reference_high=0.0, precision=0),
        LabComponent(name="Blood", unit="", reference_low=0.0, reference_high=0.0, precision=0),
    ]),
    "LFTs": LabPanel(name="LFTs", components=[
        LabComponent(name="AST", unit="U/L", reference_low=10.0, reference_high=40.0, precision=0),
        LabComponent(name="ALT", unit="U/L", reference_low=7.0, reference_high=56.0, precision=0),
        LabComponent(name="ALP", unit="U/L", reference_low=44.0, reference_high=147.0, precision=0),
        LabComponent(name="GGT", unit="U/L", reference_low=9.0, reference_high=48.0, precision=0),
        LabComponent(name="Albumin", unit="g/dL", reference_low=3.5, reference_high=5.0, precision=1),
        LabComponent(name="Total Bilirubin", unit="mg/dL", reference_low=0.1, reference_high=1.2, precision=1),
        LabComponent(name="Direct Bilirubin", unit="mg/dL", reference_low=0.0, reference_high=0.3, precision=1),
    ]),
    "thyroid": LabPanel(name="thyroid", components=[
        LabComponent(name="TSH", unit="mIU/L", reference_low=0.4, reference_high=4.0, precision=2),
        LabComponent(name="Free T4", unit="ng/dL", reference_low=0.8, reference_high=1.8, precision=2),
        LabComponent(name="Free T3", unit="pg/mL", reference_low=2.3, reference_high=4.2, precision=1),
    ]),
    "iron_studies": LabPanel(name="iron_studies", components=[
        LabComponent(name="Serum Iron", unit="mcg/dL", reference_low=60.0, reference_high=170.0, precision=0),
        LabComponent(name="TIBC", unit="mcg/dL", reference_low=250.0, reference_high=370.0, precision=0),
        LabComponent(name="Ferritin", unit="ng/mL", reference_low=12.0, reference_high=300.0, precision=0),
        LabComponent(name="Transferrin Sat", unit="%", reference_low=20.0, reference_high=50.0, precision=0),
    ]),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_lab_panels.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/generation/lab_panels.py tests/test_lab_panels.py
git commit -m "feat: add static lab panel templates with reference ranges for 14 panels"
```

---

### Task 5: Imaging Templates

**Files:**
- Create: `src/casecrawler/generation/imaging_templates.py`
- Test: `tests/test_imaging_templates.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_imaging_templates.py
from casecrawler.generation.imaging_templates import IMAGING_TEMPLATES, ImagingTemplate, get_imaging_template


def test_ct_template():
    ct = IMAGING_TEMPLATES["CT"]
    assert isinstance(ct, ImagingTemplate)
    assert "head" in ct.valid_body_regions
    assert "hyperdense" in ct.terminology["density"]
    assert "hyperintense" not in ct.terminology.get("density", [])


def test_mri_template():
    mri = IMAGING_TEMPLATES["MRI"]
    assert "hyperintense" in mri.terminology["signal"]
    assert "hyperdense" not in mri.terminology.get("signal", [])


def test_xr_template():
    xr = IMAGING_TEMPLATES["XR"]
    assert "chest" in xr.valid_body_regions
    assert "opacity" in xr.terminology["density"]


def test_all_modalities_present():
    expected = ["CT", "MRI", "XR", "US", "CTA"]
    for mod in expected:
        assert mod in IMAGING_TEMPLATES, f"Missing modality: {mod}"


def test_get_imaging_template_found():
    t = get_imaging_template("CT")
    assert t is not None


def test_get_imaging_template_not_found():
    t = get_imaging_template("PET")
    assert t is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_imaging_templates.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement imaging templates**

```python
# src/casecrawler/generation/imaging_templates.py
from __future__ import annotations

from pydantic import BaseModel


class ImagingTemplate(BaseModel):
    modality: str
    valid_body_regions: list[str]
    terminology: dict[str, list[str]]
    report_format: str


def get_imaging_template(modality: str) -> ImagingTemplate | None:
    return IMAGING_TEMPLATES.get(modality)


IMAGING_TEMPLATES: dict[str, ImagingTemplate] = {
    "CT": ImagingTemplate(
        modality="CT",
        valid_body_regions=["head", "chest", "abdomen", "pelvis", "spine", "neck", "extremity"],
        terminology={
            "density": ["hyperdense", "hypodense", "isodense"],
            "enhancement": ["enhancing", "non-enhancing", "rim-enhancing"],
            "morphology": ["mass", "lesion", "collection", "effusion", "hemorrhage", "calcification"],
            "distribution": ["focal", "diffuse", "multifocal", "segmental"],
        },
        report_format="findings → impression",
    ),
    "MRI": ImagingTemplate(
        modality="MRI",
        valid_body_regions=["brain", "spine", "abdomen", "pelvis", "extremity", "chest", "neck"],
        terminology={
            "signal": ["hyperintense", "hypointense", "isointense"],
            "sequences": ["T1-weighted", "T2-weighted", "FLAIR", "DWI", "ADC", "post-contrast"],
            "findings": ["restricted diffusion", "enhancement", "edema", "mass effect", "herniation"],
            "morphology": ["mass", "lesion", "collection", "effusion"],
        },
        report_format="findings → impression",
    ),
    "XR": ImagingTemplate(
        modality="XR",
        valid_body_regions=["chest", "abdomen", "extremity", "spine", "pelvis"],
        terminology={
            "density": ["opacity", "lucency", "radiopaque", "radiolucent"],
            "findings": ["consolidation", "infiltrate", "effusion", "pneumothorax", "cardiomegaly", "fracture", "dislocation"],
            "distribution": ["focal", "diffuse", "bilateral", "unilateral", "lobar", "patchy"],
        },
        report_format="findings → impression",
    ),
    "US": ImagingTemplate(
        modality="US",
        valid_body_regions=["abdomen", "pelvis", "neck", "extremity", "chest", "cardiac"],
        terminology={
            "echogenicity": ["hyperechoic", "hypoechoic", "anechoic", "isoechoic", "heterogeneous"],
            "findings": ["mass", "collection", "free fluid", "thrombus", "calculus", "dilation"],
            "flow": ["hyperemic", "avascular", "reduced flow", "absent flow", "reversal of flow"],
        },
        report_format="findings → impression",
    ),
    "CTA": ImagingTemplate(
        modality="CTA",
        valid_body_regions=["head", "neck", "chest", "abdomen", "extremity"],
        terminology={
            "vascular": ["aneurysm", "stenosis", "occlusion", "dissection", "filling defect", "extravasation"],
            "density": ["hyperdense", "hypodense"],
            "morphology": ["saccular", "fusiform", "irregular", "smooth"],
        },
        report_format="findings → impression",
    ),
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_imaging_templates.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/generation/imaging_templates.py tests/test_imaging_templates.py
git commit -m "feat: add imaging modality templates with vocabulary constraints"
```

---

### Task 6: Extended Prompts for Multi-Step Pipeline

**Files:**
- Modify: `src/casecrawler/generation/prompts.py`

- [ ] **Step 1: Write failing test for new prompt functions**

```python
# tests/test_multi_step_prompts.py
from casecrawler.generation.prompts import (
    BLUEPRINT_REVIEWER_SYSTEM,
    CASE_PLANNER_SYSTEM,
    CONSISTENCY_CHECKER_SYSTEM,
    DIFFICULTY_RULES,
    PHASE_RENDERER_SYSTEM,
    build_blueprint_reviewer_prompt,
    build_case_planner_prompt,
    build_consistency_checker_prompt,
    build_phase_renderer_prompt,
)


def test_difficulty_rules_have_phase_counts():
    for level in ["medical_student", "resident", "attending"]:
        assert "phase_count_min" in DIFFICULTY_RULES[level]
        assert "phase_count_max" in DIFFICULTY_RULES[level]
    assert DIFFICULTY_RULES["medical_student"]["phase_count_min"] == 3
    assert DIFFICULTY_RULES["medical_student"]["phase_count_max"] == 3
    assert DIFFICULTY_RULES["resident"]["phase_count_min"] == 5
    assert DIFFICULTY_RULES["resident"]["phase_count_max"] == 5
    assert DIFFICULTY_RULES["attending"]["phase_count_min"] == 7
    assert DIFFICULTY_RULES["attending"]["phase_count_max"] == 10


def test_case_planner_system_prompt_exists():
    assert "clinical case planner" in CASE_PLANNER_SYSTEM.lower() or "blueprint" in CASE_PLANNER_SYSTEM.lower()


def test_build_case_planner_prompt():
    prompt = build_case_planner_prompt(
        topic="SAH", difficulty="resident", context="[Source 1] SAH content",
    )
    assert "SAH" in prompt
    assert "resident" in prompt
    assert "SAH content" in prompt
    assert "phase_count_min" in prompt or "5 phases" in prompt.lower() or "5" in prompt


def test_build_blueprint_reviewer_prompt():
    prompt = build_blueprint_reviewer_prompt(
        blueprint_json='{"diagnosis": "SAH"}', context="[Source 1] SAH content",
    )
    assert "SAH" in prompt


def test_build_phase_renderer_prompt():
    prompt = build_phase_renderer_prompt(
        blueprint_json='{"diagnosis": "SAH"}',
        phase_json='{"phase_number": 1}',
        difficulty="resident",
        context="[Source 1] SAH content",
        lab_panel_context="CBC: WBC 4.5-11.0 K/uL",
    )
    assert "SAH" in prompt
    assert "phase_number" in prompt or "Phase 1" in prompt or "1" in prompt
    assert "CBC" in prompt


def test_build_consistency_checker_prompt():
    prompt = build_consistency_checker_prompt(
        phases_json='[{"phase_number": 1}, {"phase_number": 2}]',
    )
    assert "phase_number" in prompt or "consistency" in prompt.lower()


def test_phase_renderer_system_prompt_exists():
    assert len(PHASE_RENDERER_SYSTEM) > 100


def test_consistency_checker_system_prompt_exists():
    assert len(CONSISTENCY_CHECKER_SYSTEM) > 100


def test_blueprint_reviewer_system_prompt_exists():
    assert len(BLUEPRINT_REVIEWER_SYSTEM) > 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_multi_step_prompts.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Add phase count fields to DIFFICULTY_RULES and new system prompts**

Add the following to `src/casecrawler/generation/prompts.py`. Insert the new keys into each existing `DIFFICULTY_RULES` dict entry, and add the new system prompts and builder functions after the existing ones.

Add to each difficulty level in `DIFFICULTY_RULES`:

```python
# medical_student — add these keys:
"phase_count_min": 3,
"phase_count_max": 3,
"decisions_per_phase": "2-3",
"quality_distribution": "1 optimal, 1 harmful/catastrophic, 0-1 suboptimal",

# resident — add these keys:
"phase_count_min": 5,
"phase_count_max": 5,
"decisions_per_phase": "3-4",
"quality_distribution": "1 optimal, 1-2 acceptable/suboptimal, 1 harmful",

# attending — add these keys:
"phase_count_min": 7,
"phase_count_max": 10,
"decisions_per_phase": "4-5",
"quality_distribution": "1 optimal, 2 acceptable, 1-2 suboptimal, 0-1 catastrophic",
```

Add the new system prompts and builder functions at the end of the file:

```python
CASE_PLANNER_SYSTEM = """You are a clinical case planner creating structured blueprints for multi-step medical simulations.

Your blueprints must:
- Define a realistic clinical arc from presentation through management
- Specify the exact number of phases appropriate for the difficulty level
- Place decision points at genuine clinical crossroads
- Include branching points where wrong decisions lead to different outcomes
- Anticipate realistic complications that could arise
- Ground all clinical reasoning in the provided source material

You produce STRUCTURE ONLY — no prose vignettes, no lab values, no imaging reports.
Each phase specifies: what's happening clinically, what diagnostics are available/pending,
what type of decision is being made, and what the correct action is with reasoning."""

BLUEPRINT_REVIEWER_SYSTEM = """You are a clinical blueprint reviewer validating the structural soundness of a multi-step case plan.

Evaluate the blueprint on four criteria:
1. Is the clinical arc medically sound? Does the diagnosis match the planned workup and management?
2. Do phases follow a logical temporal sequence? Are time offsets realistic?
3. Are branching points at genuine decision points where a clinician could reasonably go wrong?
4. Is the phase count appropriate for the stated difficulty level?

If ANY criterion fails, reject with specific feedback. Your feedback should tell the planner
exactly what to fix. Set approved=true only if all criteria pass."""

PHASE_RENDERER_SYSTEM = """You are a clinical case phase renderer. Given a case blueprint and a specific phase skeleton,
you generate the detailed clinical content for that phase.

You must produce:
1. A realistic narrative for this moment in the case
2. Structured vital signs (if clinically relevant at this phase)
3. Structured lab results using EXACT units and reference ranges from the provided templates
4. Structured imaging results using modality-appropriate terminology
5. Decision options graded on a 5-level scale: optimal, acceptable, suboptimal, harmful, catastrophic
6. A phase outcome describing what happens next on the optimal path

Rules:
- Lab values must use the exact unit and precision from the panel template
- Abnormal values must be physiologically consistent with the diagnosis
- Flag any value outside reference range (H/L) and any critical value as "critical"
- Imaging findings must use terminology appropriate for the modality (no "hyperdense" for MRI)
- Vital signs must be physiologically coherent
- Decision quality grades must have clear clinical reasoning
- Every phase must have exactly ONE optimal decision"""

CONSISTENCY_CHECKER_SYSTEM = """You are a clinical consistency checker reviewing a set of rendered case phases for cross-phase coherence.

Check for:
1. Vital sign continuity — trends must be physiologically logical. A patient with HR 120 should not
   suddenly have HR 68 without treatment explanation.
2. Lab value coherence — trending values (troponin, lactate, Hgb) must move in clinically plausible
   directions across phases.
3. Temporal logic — order-to-result times must be realistic (stat CBC ~30min, cultures ~48h, CT ~30-45min).
   A test ordered at T+10min cannot result at T+5min.
4. Narrative continuity — patient state must not contradict across phases. If intubated in phase 4,
   cannot be conversational in phase 5.
5. Decision coherence — available actions must not include things already done in prior phases.

For each issue found, report: phase_number, field, issue description, suggested fix.
If no issues found, return an empty list."""


def build_case_planner_prompt(topic: str, difficulty: str, context: str) -> str:
    rules = DIFFICULTY_RULES.get(difficulty, DIFFICULTY_RULES["resident"])
    return f"""Create a case blueprint for the following topic.

## Topic
{topic}

## Difficulty Level: {difficulty}
- Vignette style: {rules['vignette']}
- Knowledge level: {rules['knowledge']}
- Phase count: {rules['phase_count_min']}-{rules['phase_count_max']} phases
- Decisions per phase: {rules['decisions_per_phase']}
- Quality distribution: {rules['quality_distribution']}
- Complications: {rules['complications']}

## Medical Knowledge (from real sources)
{context}

## Instructions
Create a structured blueprint with:
1. The diagnosis and clinical arc
2. Phase-by-phase plan with time offsets, decision types, and correct actions
3. Branching points where wrong decisions cause different outcomes (redirect, fork, or terminal)
4. Expected complications that could arise during the case

Do NOT generate prose, lab values, or imaging reports — structure only."""


def build_blueprint_reviewer_prompt(blueprint_json: str, context: str) -> str:
    return f"""Review this case blueprint for structural soundness.

## Blueprint
{blueprint_json}

## Source Material
{context}

## Instructions
Evaluate: clinical arc validity, temporal sequence logic, branching point placement,
and phase count appropriateness. Set approved=true only if all criteria pass.
If rejecting, provide specific actionable feedback."""


def build_phase_renderer_prompt(
    blueprint_json: str,
    phase_json: str,
    difficulty: str,
    context: str,
    lab_panel_context: str,
) -> str:
    rules = DIFFICULTY_RULES.get(difficulty, DIFFICULTY_RULES["resident"])
    return f"""Render the detailed clinical content for this phase.

## Case Blueprint
{blueprint_json}

## Phase to Render
{phase_json}

## Difficulty Level: {difficulty}
- Decisions per phase: {rules['decisions_per_phase']}
- Quality distribution: {rules['quality_distribution']}
- Decision tree style: {rules['decision_tree']}

## Medical Knowledge (from real sources)
{context}

## Lab Panel Reference Ranges
{lab_panel_context}

## Instructions
Generate:
1. Narrative text for this clinical moment
2. Vital signs (if relevant)
3. Lab results using EXACT units and ranges from the panel reference above
4. Imaging results using modality-appropriate terminology
5. Decision options with 5-level quality grading
6. Phase outcome (next phase, patient status, transition narrative)"""


def build_consistency_checker_prompt(phases_json: str) -> str:
    return f"""Check these rendered case phases for cross-phase consistency.

## Rendered Phases
{phases_json}

## Instructions
Check vital sign continuity, lab value coherence, temporal logic,
narrative continuity, and decision coherence.

Return a list of issues found. Each issue should have:
- phase_number: which phase has the problem
- field: which field is inconsistent (e.g., "vitals.hr", "lab_results.Hgb")
- issue: what the inconsistency is
- suggested_fix: how to resolve it

If no issues found, return an empty list."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_multi_step_prompts.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Run existing tests to verify nothing broke**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_generation_pipeline.py -v`
Expected: All 3 existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/casecrawler/generation/prompts.py tests/test_multi_step_prompts.py
git commit -m "feat: add multi-step prompts (planner, blueprint reviewer, phase renderer, consistency checker)"
```

---

### Task 7: Case Planner Agent

**Files:**
- Create: `src/casecrawler/generation/case_planner.py`
- Test: `tests/test_case_planner.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_case_planner.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from casecrawler.generation.case_planner import CasePlannerAgent, CasePlannerOutput
from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.blueprint import BranchPoint, CaseBlueprint, PhaseBlueprint
from casecrawler.models.case import Patient


def _mock_planner_result():
    return StructuredGenerationResult(
        data=CasePlannerOutput(
            patient=Patient(age=42, sex="female", demographics="No significant PMH"),
            specialty=["neurosurgery", "emergency_medicine"],
            blueprint=CaseBlueprint(
                diagnosis="SAH",
                clinical_arc="SAH via CT, surgical management",
                phase_count=3,
                phases=[
                    PhaseBlueprint(
                        phase_number=1, time_offset="T+0",
                        clinical_context="ED presentation",
                        available_diagnostics=[], pending_diagnostics=[],
                        decision_type="order_workup",
                        correct_action="Order CT head",
                        key_reasoning="Most sensitive",
                    ),
                    PhaseBlueprint(
                        phase_number=2, time_offset="T+45min",
                        clinical_context="CT results",
                        available_diagnostics=["CT head"], pending_diagnostics=[],
                        decision_type="interpret_results",
                        correct_action="Consult neurosurgery",
                        key_reasoning="CT confirms SAH",
                    ),
                    PhaseBlueprint(
                        phase_number=3, time_offset="T+2h",
                        clinical_context="Neurosurgery consulted",
                        available_diagnostics=["CT head", "CBC"], pending_diagnostics=[],
                        decision_type="start_treatment",
                        correct_action="Nimodipine and ICU admission",
                        key_reasoning="Prevent vasospasm",
                    ),
                ],
                branching_points=[
                    BranchPoint(
                        phase_number=1, branch_type="terminal",
                        trigger_action_quality="catastrophic",
                        description="Discharge leads to rebleed",
                    ),
                ],
                expected_complications=["vasospasm"],
            ),
        ),
        input_tokens=500, output_tokens=400, model="test-model",
    )


@pytest.mark.asyncio
async def test_case_planner_generates_blueprint():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_planner_result()

    planner = CasePlannerAgent(provider=provider)
    result = await planner.plan(topic="SAH", difficulty="resident", context="SAH content")

    assert result.data.blueprint.diagnosis == "SAH"
    assert result.data.blueprint.phase_count == 3
    assert len(result.data.blueprint.phases) == 3
    provider.generate_structured.assert_called_once()


@pytest.mark.asyncio
async def test_case_planner_passes_retry_notes():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_planner_result()

    planner = CasePlannerAgent(provider=provider)
    await planner.plan(
        topic="SAH", difficulty="resident", context="SAH content",
        retry_notes=["Phase count too low"],
    )

    call_args = provider.generate_structured.call_args
    prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
    assert "Phase count too low" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_case_planner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement CasePlannerAgent**

```python
# src/casecrawler/generation/case_planner.py
from __future__ import annotations

from pydantic import BaseModel

from casecrawler.generation.prompts import (
    CASE_PLANNER_SYSTEM,
    build_case_planner_prompt,
    build_retry_prompt,
)
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult
from casecrawler.models.blueprint import CaseBlueprint
from casecrawler.models.case import Patient


class CasePlannerOutput(BaseModel):
    blueprint: CaseBlueprint
    patient: Patient
    specialty: list[str]


class CasePlannerAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def plan(
        self,
        topic: str,
        difficulty: str,
        context: str,
        retry_notes: list[str] | None = None,
    ) -> StructuredGenerationResult:
        prompt = build_case_planner_prompt(topic, difficulty, context)
        if retry_notes:
            prompt = build_retry_prompt(prompt, retry_notes)
        return await self._provider.generate_structured(
            prompt, CasePlannerOutput, system=CASE_PLANNER_SYSTEM,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_case_planner.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/generation/case_planner.py tests/test_case_planner.py
git commit -m "feat: add CasePlannerAgent for blueprint generation"
```

---

### Task 8: Blueprint Reviewer Agent

**Files:**
- Create: `src/casecrawler/generation/blueprint_reviewer.py`
- Test: `tests/test_blueprint_reviewer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_blueprint_reviewer.py
from unittest.mock import AsyncMock

import pytest

from casecrawler.generation.blueprint_reviewer import BlueprintReviewerAgent, BlueprintReviewResult
from casecrawler.llm.base import StructuredGenerationResult


def _mock_approved():
    return StructuredGenerationResult(
        data=BlueprintReviewResult(approved=True, notes=[]),
        input_tokens=300, output_tokens=50, model="test-model",
    )


def _mock_rejected():
    return StructuredGenerationResult(
        data=BlueprintReviewResult(approved=False, notes=["Phase count too low for attending"]),
        input_tokens=300, output_tokens=80, model="test-model",
    )


@pytest.mark.asyncio
async def test_blueprint_reviewer_approves():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_approved()

    reviewer = BlueprintReviewerAgent(provider=provider)
    result = await reviewer.review(blueprint_json='{"diagnosis":"SAH"}', context="SAH content")

    assert result.data.approved is True
    assert result.data.notes == []


@pytest.mark.asyncio
async def test_blueprint_reviewer_rejects():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_rejected()

    reviewer = BlueprintReviewerAgent(provider=provider)
    result = await reviewer.review(blueprint_json='{"diagnosis":"SAH"}', context="SAH content")

    assert result.data.approved is False
    assert len(result.data.notes) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_blueprint_reviewer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement BlueprintReviewerAgent**

```python
# src/casecrawler/generation/blueprint_reviewer.py
from __future__ import annotations

from pydantic import BaseModel

from casecrawler.generation.prompts import BLUEPRINT_REVIEWER_SYSTEM, build_blueprint_reviewer_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult


class BlueprintReviewResult(BaseModel):
    approved: bool
    notes: list[str]


class BlueprintReviewerAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def review(self, blueprint_json: str, context: str) -> StructuredGenerationResult:
        prompt = build_blueprint_reviewer_prompt(blueprint_json, context)
        return await self._provider.generate_structured(
            prompt, BlueprintReviewResult, system=BLUEPRINT_REVIEWER_SYSTEM,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_blueprint_reviewer.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/generation/blueprint_reviewer.py tests/test_blueprint_reviewer.py
git commit -m "feat: add BlueprintReviewerAgent for lightweight blueprint validation"
```

---

### Task 9: Phase Renderer Agent

**Files:**
- Create: `src/casecrawler/generation/phase_renderer.py`
- Test: `tests/test_phase_renderer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_phase_renderer.py
from unittest.mock import AsyncMock

import pytest

from casecrawler.generation.phase_renderer import PhaseRendererAgent, PhaseRendererOutput
from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.diagnostics import LabResult, LabValue, VitalSigns
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome


def _mock_render_result():
    return StructuredGenerationResult(
        data=PhaseRendererOutput(
            phase=CasePhase(
                phase_number=1,
                time_offset="T+0",
                narrative="A 42-year-old woman presents with thunderclap headache.",
                vitals=VitalSigns(hr=92, bp_systolic=168, bp_diastolic=95, rr=20, spo2=98.0, temp_c=37.4, gcs=14),
                lab_results=[],
                imaging_results=[],
                clinical_update=None,
                decisions=[
                    PhaseDecision(
                        action="Order CT head", is_optimal=True, quality="optimal",
                        reasoning="Most sensitive", clinical_outcome="SAH confirmed",
                    ),
                    PhaseDecision(
                        action="Discharge", is_optimal=False, quality="catastrophic",
                        reasoning="Misdiagnosis", clinical_outcome="Rebleed at home",
                    ),
                ],
                phase_outcome=PhaseOutcome(
                    optimal_next_phase=2, patient_status="stable",
                    narrative_transition="CT ordered stat.",
                ),
            ),
        ),
        input_tokens=800, output_tokens=600, model="test-model",
    )


@pytest.mark.asyncio
async def test_phase_renderer_renders_phase():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_render_result()

    renderer = PhaseRendererAgent(provider=provider)
    result = await renderer.render(
        blueprint_json='{"diagnosis":"SAH"}',
        phase_json='{"phase_number":1}',
        difficulty="resident",
        context="SAH content",
        lab_panel_context="CBC reference ranges",
    )

    assert result.data.phase.phase_number == 1
    assert len(result.data.phase.decisions) == 2
    assert result.data.phase.vitals.hr == 92
    provider.generate_structured.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_phase_renderer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement PhaseRendererAgent**

```python
# src/casecrawler/generation/phase_renderer.py
from __future__ import annotations

from pydantic import BaseModel

from casecrawler.generation.prompts import PHASE_RENDERER_SYSTEM, build_phase_renderer_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult
from casecrawler.models.phase import CasePhase


class PhaseRendererOutput(BaseModel):
    phase: CasePhase


class PhaseRendererAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def render(
        self,
        blueprint_json: str,
        phase_json: str,
        difficulty: str,
        context: str,
        lab_panel_context: str,
    ) -> StructuredGenerationResult:
        prompt = build_phase_renderer_prompt(
            blueprint_json=blueprint_json,
            phase_json=phase_json,
            difficulty=difficulty,
            context=context,
            lab_panel_context=lab_panel_context,
        )
        return await self._provider.generate_structured(
            prompt, PhaseRendererOutput, system=PHASE_RENDERER_SYSTEM,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_phase_renderer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/generation/phase_renderer.py tests/test_phase_renderer.py
git commit -m "feat: add PhaseRendererAgent for rendering individual case phases"
```

---

### Task 10: Consistency Checker Agent

**Files:**
- Create: `src/casecrawler/generation/consistency_checker.py`
- Test: `tests/test_consistency_checker.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_consistency_checker.py
from unittest.mock import AsyncMock

import pytest

from casecrawler.generation.consistency_checker import (
    ConsistencyCheckerAgent,
    ConsistencyCheckerOutput,
    ConsistencyIssue,
)
from casecrawler.llm.base import StructuredGenerationResult


def _mock_no_issues():
    return StructuredGenerationResult(
        data=ConsistencyCheckerOutput(issues=[]),
        input_tokens=500, output_tokens=50, model="test-model",
    )


def _mock_with_issues():
    return StructuredGenerationResult(
        data=ConsistencyCheckerOutput(
            issues=[
                ConsistencyIssue(
                    phase_number=3,
                    field="vitals.hr",
                    issue="HR drops from 120 to 68 without treatment",
                    suggested_fix="Set HR to 108 in phase 3 or add treatment in phase 2",
                ),
            ],
        ),
        input_tokens=500, output_tokens=100, model="test-model",
    )


@pytest.mark.asyncio
async def test_consistency_checker_no_issues():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_no_issues()

    checker = ConsistencyCheckerAgent(provider=provider)
    result = await checker.check(phases_json='[{"phase_number": 1}]')

    assert result.data.issues == []


@pytest.mark.asyncio
async def test_consistency_checker_finds_issues():
    provider = AsyncMock()
    provider.generate_structured.return_value = _mock_with_issues()

    checker = ConsistencyCheckerAgent(provider=provider)
    result = await checker.check(phases_json='[{"phase_number": 1}, {"phase_number": 2}]')

    assert len(result.data.issues) == 1
    assert result.data.issues[0].phase_number == 3
    assert "HR" in result.data.issues[0].issue
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_consistency_checker.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ConsistencyCheckerAgent**

```python
# src/casecrawler/generation/consistency_checker.py
from __future__ import annotations

from pydantic import BaseModel

from casecrawler.generation.prompts import CONSISTENCY_CHECKER_SYSTEM, build_consistency_checker_prompt
from casecrawler.llm.base import BaseLLMProvider, StructuredGenerationResult


class ConsistencyIssue(BaseModel):
    phase_number: int
    field: str
    issue: str
    suggested_fix: str


class ConsistencyCheckerOutput(BaseModel):
    issues: list[ConsistencyIssue]


class ConsistencyCheckerAgent:
    def __init__(self, provider: BaseLLMProvider) -> None:
        self._provider = provider

    async def check(self, phases_json: str) -> StructuredGenerationResult:
        prompt = build_consistency_checker_prompt(phases_json)
        return await self._provider.generate_structured(
            prompt, ConsistencyCheckerOutput, system=CONSISTENCY_CHECKER_SYSTEM,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_consistency_checker.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/generation/consistency_checker.py tests/test_consistency_checker.py
git commit -m "feat: add ConsistencyCheckerAgent for cross-phase coherence validation"
```

---

### Task 11: Multi-Step Pipeline Orchestration

**Files:**
- Create: `src/casecrawler/generation/multi_step_pipeline.py`
- Test: `tests/test_multi_step_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_multi_step_pipeline.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from casecrawler.generation.blueprint_reviewer import BlueprintReviewResult
from casecrawler.generation.case_planner import CasePlannerOutput
from casecrawler.generation.consistency_checker import ConsistencyCheckerOutput
from casecrawler.generation.multi_step_pipeline import MultiStepPipeline
from casecrawler.generation.phase_renderer import PhaseRendererOutput
from casecrawler.llm.base import StructuredGenerationResult
from casecrawler.models.blueprint import BranchPoint, CaseBlueprint, PhaseBlueprint
from casecrawler.models.case import (
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)
from casecrawler.models.diagnostics import VitalSigns
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome


def _blueprint():
    return CaseBlueprint(
        diagnosis="SAH",
        clinical_arc="SAH via CT, surgical management",
        phase_count=3,
        phases=[
            PhaseBlueprint(
                phase_number=i, time_offset=f"T+{i*30}min",
                clinical_context=f"Phase {i}", available_diagnostics=[], pending_diagnostics=[],
                decision_type="order_workup", correct_action=f"Action {i}",
                key_reasoning=f"Reasoning {i}",
            )
            for i in range(1, 4)
        ],
        branching_points=[],
        expected_complications=["vasospasm"],
    )


def _phase(n: int):
    return CasePhase(
        phase_number=n, time_offset=f"T+{n*30}min",
        narrative=f"Phase {n} narrative",
        vitals=VitalSigns(hr=80+n, bp_systolic=120, bp_diastolic=80, rr=16, spo2=98.0, temp_c=37.0),
        lab_results=[], imaging_results=[],
        decisions=[
            PhaseDecision(
                action=f"Action {n}", is_optimal=True, quality="optimal",
                reasoning="Correct", clinical_outcome="Good outcome",
            ),
        ],
        phase_outcome=PhaseOutcome(
            optimal_next_phase=n+1 if n < 3 else None,
            patient_status="stable",
            narrative_transition=f"Transition from phase {n}",
        ),
    )


def _mock_planner():
    return StructuredGenerationResult(
        data=CasePlannerOutput(
            blueprint=_blueprint(),
            patient=Patient(age=42, sex="female", demographics="No significant PMH"),
            specialty=["neurosurgery"],
        ),
        input_tokens=500, output_tokens=400, model="test-model",
    )


def _mock_blueprint_review_approved():
    return StructuredGenerationResult(
        data=BlueprintReviewResult(approved=True, notes=[]),
        input_tokens=300, output_tokens=50, model="test-model",
    )


def _mock_phase_render(n: int):
    return StructuredGenerationResult(
        data=PhaseRendererOutput(phase=_phase(n)),
        input_tokens=800, output_tokens=600, model="test-model",
    )


def _mock_consistency_ok():
    return StructuredGenerationResult(
        data=ConsistencyCheckerOutput(issues=[]),
        input_tokens=500, output_tokens=50, model="test-model",
    )


def _mock_clinical_review_approved():
    return StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92,
            approved=True, notes=[],
        ),
        input_tokens=800, output_tokens=100, model="test-model",
    )


@pytest.fixture
def mock_retriever():
    r = MagicMock()
    r.retrieve.return_value = [
        {"chunk_id": "c1", "text": "SAH content", "credibility": "guideline",
         "credibility_rank": 0, "source_document_id": "pubmed:1", "source": "pubmed",
         "score": 0.9, "specialty": "", "doi": "", "url": ""},
    ]
    r.format_context.return_value = "[Source 1] (guideline, pubmed)\nSAH content"
    return r


@pytest.mark.asyncio
async def test_multi_step_pipeline_generates_case(mock_retriever):
    provider = AsyncMock()
    # Call sequence: planner, blueprint_review, render×3, consistency, clinical_review
    provider.generate_structured.side_effect = [
        _mock_planner(),
        _mock_blueprint_review_approved(),
        _mock_phase_render(1),
        _mock_phase_render(2),
        _mock_phase_render(3),
        _mock_consistency_ok(),
        _mock_clinical_review_approved(),
    ]

    pipeline = MultiStepPipeline(
        provider=provider, retriever=mock_retriever,
        max_retries=3, review_threshold=0.7,
    )
    result = await pipeline.generate_one(topic="SAH", difficulty="resident")

    assert result is not None
    assert isinstance(result, GeneratedCase)
    assert result.is_multi_step()
    assert len(result.phases) == 3
    assert result.blueprint is not None
    assert result.blueprint.diagnosis == "SAH"
    assert result.review.approved is True


@pytest.mark.asyncio
async def test_multi_step_pipeline_retries_on_blueprint_rejection(mock_retriever):
    provider = AsyncMock()
    # First attempt: planner → blueprint rejected → retry planner → approved → render → consistency → review
    provider.generate_structured.side_effect = [
        _mock_planner(),
        StructuredGenerationResult(
            data=BlueprintReviewResult(approved=False, notes=["Too few phases"]),
            input_tokens=300, output_tokens=80, model="test-model",
        ),
        _mock_planner(),
        _mock_blueprint_review_approved(),
        _mock_phase_render(1),
        _mock_phase_render(2),
        _mock_phase_render(3),
        _mock_consistency_ok(),
        _mock_clinical_review_approved(),
    ]

    pipeline = MultiStepPipeline(
        provider=provider, retriever=mock_retriever,
        max_retries=3, review_threshold=0.7,
    )
    result = await pipeline.generate_one(topic="SAH", difficulty="resident")

    assert result is not None
    assert result.is_multi_step()


@pytest.mark.asyncio
async def test_multi_step_pipeline_returns_none_after_max_retries(mock_retriever):
    provider = AsyncMock()
    # All clinical reviews reject
    rejected_review = StructuredGenerationResult(
        data=ReviewResult(
            accuracy_score=0.5, pedagogy_score=0.9, bias_score=0.92,
            approved=False, notes=["Inaccurate"],
        ),
        input_tokens=800, output_tokens=100, model="test-model",
    )

    def side_effects():
        for _ in range(3):
            yield _mock_planner()
            yield _mock_blueprint_review_approved()
            yield _mock_phase_render(1)
            yield _mock_phase_render(2)
            yield _mock_phase_render(3)
            yield _mock_consistency_ok()
            yield rejected_review

    provider.generate_structured.side_effect = list(side_effects())

    pipeline = MultiStepPipeline(
        provider=provider, retriever=mock_retriever,
        max_retries=3, review_threshold=0.7,
    )
    result = await pipeline.generate_one(topic="SAH", difficulty="resident")

    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_multi_step_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement MultiStepPipeline**

```python
# src/casecrawler/generation/multi_step_pipeline.py
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
from casecrawler.models.case import DifficultyLevel, GeneratedCase, GroundTruth


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
            # Stage 2: Plan
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


```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_multi_step_pipeline.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run all existing tests to verify nothing broke**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/casecrawler/generation/multi_step_pipeline.py tests/test_multi_step_pipeline.py
git commit -m "feat: add MultiStepPipeline with plan-then-generate orchestration"
```

---

### Task 12: Export Data Models

**Files:**
- Create: `src/casecrawler/models/export.py`
- Create: `src/casecrawler/export/__init__.py`
- Test: `tests/test_export_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_export_models.py
from casecrawler.models.export import Action, Conversation, Episode, Observation, TimeStep, Turn


def test_observation():
    obs = Observation(
        narrative="Patient presents with headache",
        vitals={"hr": 92, "bp_systolic": 168},
        new_lab_results=[],
        new_imaging_results=[],
        pending_orders=["CBC"],
        time_elapsed="T+0",
    )
    assert obs.narrative == "Patient presents with headache"
    assert obs.pending_orders == ["CBC"]


def test_action():
    a = Action(id="a1", description="Order CT head", quality="optimal")
    assert a.quality == "optimal"


def test_timestep():
    ts = TimeStep(
        step_number=1,
        observation=Observation(
            narrative="Presentation",
            vitals=None,
            new_lab_results=[],
            new_imaging_results=[],
            pending_orders=[],
            time_elapsed="T+0",
        ),
        action_space=[
            Action(id="a1", description="Order CT head", quality="optimal"),
            Action(id="a2", description="Discharge", quality="catastrophic"),
        ],
        optimal_action="a1",
        reward_table={"a1": 1.0, "a2": -1.0},
    )
    assert ts.reward_table["a1"] == 1.0


def test_episode():
    ep = Episode(
        case_id="test-1",
        difficulty="resident",
        specialty=["neurosurgery"],
        steps=[
            TimeStep(
                step_number=1,
                observation=Observation(
                    narrative="Presentation", vitals=None,
                    new_lab_results=[], new_imaging_results=[],
                    pending_orders=[], time_elapsed="T+0",
                ),
                action_space=[Action(id="a1", description="CT head", quality="optimal")],
                optimal_action="a1",
                reward_table={"a1": 1.0},
            ),
        ],
    )
    assert len(ep.steps) == 1


def test_conversation():
    conv = Conversation(
        case_id="test-1",
        system_prompt="You are a physician.",
        turns=[
            Turn(role="system", content="Patient presents with headache."),
            Turn(role="assistant", content="I would order a CT head."),
        ],
    )
    assert len(conv.turns) == 2
    assert conv.turns[0].role == "system"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_export_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create export package init**

```python
# src/casecrawler/export/__init__.py
```

- [ ] **Step 4: Implement export models**

```python
# src/casecrawler/models/export.py
from __future__ import annotations

from pydantic import BaseModel


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


class TimeStep(BaseModel):
    step_number: int
    observation: Observation
    action_space: list[Action]
    optimal_action: str
    reward_table: dict[str, float]


class Episode(BaseModel):
    case_id: str
    difficulty: str
    specialty: list[str]
    steps: list[TimeStep]


class Turn(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    case_id: str
    system_prompt: str
    turns: list[Turn]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_export_models.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/casecrawler/models/export.py src/casecrawler/export/__init__.py tests/test_export_models.py
git commit -m "feat: add export data models (Episode, Conversation) for RL and SFT formats"
```

---

### Task 13: RL Episode Exporter

**Files:**
- Create: `src/casecrawler/export/rl_exporter.py`
- Test: `tests/test_rl_exporter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rl_exporter.py
from casecrawler.export.rl_exporter import DEFAULT_REWARD_MAP, export_rl_episode
from casecrawler.models.blueprint import CaseBlueprint, PhaseBlueprint
from casecrawler.models.case import (
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)
from casecrawler.models.diagnostics import LabResult, LabValue, VitalSigns
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome


def _make_multi_step_case():
    blueprint = CaseBlueprint(
        diagnosis="SAH",
        clinical_arc="SAH via CT",
        phase_count=2,
        phases=[
            PhaseBlueprint(
                phase_number=1, time_offset="T+0", clinical_context="ED",
                available_diagnostics=[], pending_diagnostics=[],
                decision_type="order_workup", correct_action="CT head",
                key_reasoning="Most sensitive",
            ),
            PhaseBlueprint(
                phase_number=2, time_offset="T+45min", clinical_context="CT back",
                available_diagnostics=["CT"], pending_diagnostics=[],
                decision_type="start_treatment", correct_action="Consult neuro",
                key_reasoning="Confirmed SAH",
            ),
        ],
        branching_points=[],
        expected_complications=[],
    )
    phases = [
        CasePhase(
            phase_number=1, time_offset="T+0",
            narrative="Woman with thunderclap headache.",
            vitals=VitalSigns(hr=92, bp_systolic=168, bp_diastolic=95, rr=20, spo2=98.0, temp_c=37.4),
            lab_results=[],
            imaging_results=[],
            decisions=[
                PhaseDecision(
                    action="Order CT head", is_optimal=True, quality="optimal",
                    reasoning="Sensitive", clinical_outcome="SAH confirmed",
                ),
                PhaseDecision(
                    action="Order MRI", is_optimal=False, quality="suboptimal",
                    reasoning="Slower", clinical_outcome="Delayed diagnosis",
                    time_cost="2h delay",
                ),
                PhaseDecision(
                    action="Discharge", is_optimal=False, quality="catastrophic",
                    reasoning="Misdiagnosis", clinical_outcome="Rebleed",
                ),
            ],
            phase_outcome=PhaseOutcome(optimal_next_phase=2, patient_status="stable", narrative_transition="CT ordered"),
        ),
        CasePhase(
            phase_number=2, time_offset="T+45min",
            narrative="CT shows SAH.",
            vitals=VitalSigns(hr=88, bp_systolic=155, bp_diastolic=90, rr=18, spo2=98.0, temp_c=37.2),
            lab_results=[
                LabResult(
                    panel="CBC",
                    values=[LabValue(name="WBC", value=9.8, unit="K/uL", reference_low=4.5, reference_high=11.0, flag=None)],
                    timestamp="T+40min",
                ),
            ],
            imaging_results=[],
            decisions=[
                PhaseDecision(
                    action="Consult neurosurgery", is_optimal=True, quality="optimal",
                    reasoning="Confirmed SAH", clinical_outcome="Surgical eval",
                ),
            ],
            phase_outcome=PhaseOutcome(optimal_next_phase=None, patient_status="stable", narrative_transition="Admitted"),
        ),
    ]
    return GeneratedCase(
        case_id="rl-test-1",
        topic="SAH",
        difficulty=DifficultyLevel.RESIDENT,
        specialty=["neurosurgery"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        blueprint=blueprint,
        phases=phases,
        vignette="Woman with thunderclap headache.\n\nCT shows SAH.",
        decision_prompt="What would you do next?",
        ground_truth=GroundTruth(
            diagnosis="SAH", optimal_next_step="CT head",
            rationale="Most sensitive", key_findings=[],
        ),
        decision_tree=[],
        complications=[],
        review=ReviewResult(accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92, approved=True, notes=[]),
        sources=[],
        metadata={"multi_step": True},
    )


def test_export_rl_episode_basic():
    case = _make_multi_step_case()
    episode = export_rl_episode(case)

    assert episode.case_id == "rl-test-1"
    assert episode.difficulty == "resident"
    assert len(episode.steps) == 2


def test_export_rl_episode_rewards():
    case = _make_multi_step_case()
    episode = export_rl_episode(case)

    step1 = episode.steps[0]
    assert len(step1.action_space) == 3
    # Find the optimal action
    optimal = next(a for a in step1.action_space if a.quality == "optimal")
    assert step1.reward_table[optimal.id] == DEFAULT_REWARD_MAP["optimal"]
    # Find catastrophic
    catastrophic = next(a for a in step1.action_space if a.quality == "catastrophic")
    assert step1.reward_table[catastrophic.id] == DEFAULT_REWARD_MAP["catastrophic"]


def test_export_rl_episode_observations_include_diagnostics():
    case = _make_multi_step_case()
    episode = export_rl_episode(case)

    step2 = episode.steps[1]
    assert step2.observation.vitals is not None
    assert len(step2.observation.new_lab_results) == 1


def test_export_rl_episode_custom_rewards():
    case = _make_multi_step_case()
    custom = {"optimal": 10.0, "acceptable": 5.0, "suboptimal": 0.0, "harmful": -5.0, "catastrophic": -10.0}
    episode = export_rl_episode(case, reward_map=custom)

    step1 = episode.steps[0]
    optimal = next(a for a in step1.action_space if a.quality == "optimal")
    assert step1.reward_table[optimal.id] == 10.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_rl_exporter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement RL exporter**

```python
# src/casecrawler/export/rl_exporter.py
from __future__ import annotations

from casecrawler.models.case import GeneratedCase
from casecrawler.models.export import Action, Episode, Observation, TimeStep

DEFAULT_REWARD_MAP: dict[str, float] = {
    "optimal": 1.0,
    "acceptable": 0.6,
    "suboptimal": 0.2,
    "harmful": -0.5,
    "catastrophic": -1.0,
}


def export_rl_episode(
    case: GeneratedCase,
    reward_map: dict[str, float] | None = None,
) -> Episode:
    rewards = reward_map or DEFAULT_REWARD_MAP
    steps: list[TimeStep] = []

    for phase in case.phases:
        vitals_dict = phase.vitals.model_dump() if phase.vitals else None
        lab_dicts = [lr.model_dump() for lr in phase.lab_results]
        imaging_dicts = [ir.model_dump() for ir in phase.imaging_results]

        actions: list[Action] = []
        reward_table: dict[str, float] = {}
        optimal_action_id = ""

        for i, decision in enumerate(phase.decisions):
            action_id = f"p{phase.phase_number}_a{i}"
            actions.append(Action(
                id=action_id,
                description=decision.action,
                quality=decision.quality,
            ))
            reward_table[action_id] = rewards.get(decision.quality, 0.0)
            if decision.is_optimal:
                optimal_action_id = action_id

        pending = []
        if phase.phase_outcome and phase.phase_outcome.narrative_transition:
            # Extract pending orders from context if available
            pass

        steps.append(TimeStep(
            step_number=phase.phase_number,
            observation=Observation(
                narrative=phase.narrative,
                vitals=vitals_dict,
                new_lab_results=lab_dicts,
                new_imaging_results=imaging_dicts,
                pending_orders=pending,
                time_elapsed=phase.time_offset,
            ),
            action_space=actions,
            optimal_action=optimal_action_id,
            reward_table=reward_table,
        ))

    return Episode(
        case_id=case.case_id,
        difficulty=case.difficulty.value,
        specialty=case.specialty,
        steps=steps,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_rl_exporter.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/export/rl_exporter.py tests/test_rl_exporter.py
git commit -m "feat: add RL episode exporter with configurable reward mapping"
```

---

### Task 14: SFT Conversation Exporter

**Files:**
- Create: `src/casecrawler/export/sft_exporter.py`
- Test: `tests/test_sft_exporter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sft_exporter.py
from casecrawler.export.sft_exporter import export_sft_conversation
from casecrawler.models.blueprint import CaseBlueprint, PhaseBlueprint
from casecrawler.models.case import (
    DifficultyLevel,
    GeneratedCase,
    GroundTruth,
    Patient,
    ReviewResult,
)
from casecrawler.models.diagnostics import VitalSigns
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome


def _make_case():
    blueprint = CaseBlueprint(
        diagnosis="SAH", clinical_arc="SAH via CT", phase_count=2,
        phases=[
            PhaseBlueprint(
                phase_number=1, time_offset="T+0", clinical_context="ED",
                available_diagnostics=[], pending_diagnostics=[],
                decision_type="order_workup", correct_action="CT head",
                key_reasoning="Most sensitive initial test for SAH",
            ),
            PhaseBlueprint(
                phase_number=2, time_offset="T+45min", clinical_context="CT back",
                available_diagnostics=["CT"], pending_diagnostics=[],
                decision_type="start_treatment", correct_action="Consult neurosurgery",
                key_reasoning="CT confirms SAH needs surgical evaluation",
            ),
        ],
        branching_points=[], expected_complications=[],
    )
    phases = [
        CasePhase(
            phase_number=1, time_offset="T+0",
            narrative="A 42-year-old woman with thunderclap headache.",
            vitals=VitalSigns(hr=92, bp_systolic=168, bp_diastolic=95, rr=20, spo2=98.0, temp_c=37.4),
            lab_results=[], imaging_results=[],
            decisions=[
                PhaseDecision(
                    action="Order CT head", is_optimal=True, quality="optimal",
                    reasoning="Most sensitive", clinical_outcome="SAH confirmed",
                ),
                PhaseDecision(
                    action="Discharge", is_optimal=False, quality="catastrophic",
                    reasoning="Misdiagnosis", clinical_outcome="Rebleed",
                ),
            ],
            phase_outcome=PhaseOutcome(optimal_next_phase=2, patient_status="stable", narrative_transition="CT ordered"),
        ),
        CasePhase(
            phase_number=2, time_offset="T+45min",
            narrative="CT shows diffuse SAH.",
            vitals=VitalSigns(hr=88, bp_systolic=155, bp_diastolic=90, rr=18, spo2=98.0, temp_c=37.2),
            lab_results=[], imaging_results=[],
            decisions=[
                PhaseDecision(
                    action="Consult neurosurgery", is_optimal=True, quality="optimal",
                    reasoning="Confirmed SAH", clinical_outcome="Surgical eval",
                ),
            ],
            phase_outcome=PhaseOutcome(optimal_next_phase=None, patient_status="stable", narrative_transition="Admitted"),
        ),
    ]
    return GeneratedCase(
        case_id="sft-test-1", topic="SAH", difficulty=DifficultyLevel.RESIDENT,
        specialty=["neurosurgery"],
        patient=Patient(age=42, sex="female", demographics="Healthy"),
        blueprint=blueprint, phases=phases,
        vignette="...", decision_prompt="What would you do next?",
        ground_truth=GroundTruth(diagnosis="SAH", optimal_next_step="CT head", rationale="", key_findings=[]),
        decision_tree=[], complications=[],
        review=ReviewResult(accuracy_score=0.95, pedagogy_score=0.9, bias_score=0.92, approved=True, notes=[]),
        sources=[], metadata={},
    )


def test_sft_conversation_structure():
    case = _make_case()
    conv = export_sft_conversation(case)

    assert conv.case_id == "sft-test-1"
    assert "physician" in conv.system_prompt.lower() or "clinician" in conv.system_prompt.lower()
    # Should have system-assistant pairs: 2 phases × 2 turns = 4 turns
    assert len(conv.turns) == 4
    assert conv.turns[0].role == "system"
    assert conv.turns[1].role == "assistant"
    assert conv.turns[2].role == "system"
    assert conv.turns[3].role == "assistant"


def test_sft_conversation_system_turns_contain_narrative():
    case = _make_case()
    conv = export_sft_conversation(case)

    assert "thunderclap headache" in conv.turns[0].content
    assert "CT shows" in conv.turns[2].content


def test_sft_conversation_assistant_turns_contain_reasoning():
    case = _make_case()
    conv = export_sft_conversation(case)

    # Assistant turns should reference the optimal action and reasoning
    assert "CT head" in conv.turns[1].content or "CT" in conv.turns[1].content
    assert "neurosurgery" in conv.turns[3].content.lower()


def test_sft_wrong_path_variant():
    case = _make_case()
    conv = export_sft_conversation(case, include_wrong_path=True)

    # Wrong path conversation should exist and have the wrong action in assistant turns
    # It takes the first non-optimal action at phase 1
    assert conv.case_id == "sft-test-1"
    # The assistant turn for phase 1 should contain the wrong action
    assert "Discharge" in conv.turns[1].content or "catastrophic" in conv.turns[1].content.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_sft_exporter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement SFT exporter**

```python
# src/casecrawler/export/sft_exporter.py
from __future__ import annotations

from casecrawler.models.case import GeneratedCase
from casecrawler.models.export import Conversation, Turn
from casecrawler.models.phase import CasePhase

SFT_SYSTEM_PROMPT = (
    "You are a physician evaluating a patient. At each step, you will receive "
    "clinical information and must decide on the next action. Explain your "
    "clinical reasoning before stating your decision."
)


def _build_system_turn(phase: CasePhase) -> str:
    parts = [phase.narrative]
    if phase.vitals:
        v = phase.vitals
        parts.append(
            f"\nVitals: HR {v.hr}, BP {v.bp_systolic}/{v.bp_diastolic}, "
            f"RR {v.rr}, SpO2 {v.spo2}%, Temp {v.temp_c}°C"
            + (f", GCS {v.gcs}" if v.gcs is not None else "")
        )
    for lr in phase.lab_results:
        values_str = ", ".join(f"{lv.name} {lv.value} {lv.unit}" for lv in lr.values)
        parts.append(f"\n{lr.panel} ({lr.timestamp}): {values_str}")
    for ir in phase.imaging_results:
        findings_str = "; ".join(f"{f.structure}: {f.observation}" for f in ir.findings)
        parts.append(f"\n{ir.modality} {ir.body_region} ({ir.timestamp}): {findings_str}. Impression: {ir.impression}")
    if phase.clinical_update:
        parts.append(f"\nUpdate: {phase.clinical_update}")
    return "".join(parts)


def _build_assistant_turn(phase: CasePhase, use_optimal: bool = True) -> str:
    if use_optimal:
        decision = next((d for d in phase.decisions if d.is_optimal), phase.decisions[0])
    else:
        decision = next((d for d in phase.decisions if not d.is_optimal), phase.decisions[-1])

    return (
        f"Based on the clinical presentation, I would {decision.action.lower()}. "
        f"Reasoning: {decision.reasoning}. "
        f"Expected outcome: {decision.clinical_outcome}."
    )


def export_sft_conversation(
    case: GeneratedCase,
    include_wrong_path: bool = False,
) -> Conversation:
    turns: list[Turn] = []
    use_optimal = not include_wrong_path

    for phase in case.phases:
        turns.append(Turn(role="system", content=_build_system_turn(phase)))
        turns.append(Turn(role="assistant", content=_build_assistant_turn(phase, use_optimal=use_optimal)))

    return Conversation(
        case_id=case.case_id,
        system_prompt=SFT_SYSTEM_PROMPT,
        turns=turns,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest tests/test_sft_exporter.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/export/sft_exporter.py tests/test_sft_exporter.py
git commit -m "feat: add SFT conversation exporter with wrong-path variant support"
```

---

### Task 15: CLI Integration — `--multi-step` Flag and `export` Command

**Files:**
- Modify: `src/casecrawler/cli.py`
- Modify: `src/casecrawler/api/routes/generate.py`

- [ ] **Step 1: Add `--multi-step` flag to CLI generate command**

In `src/casecrawler/cli.py`, find the `generate` command and add a `--multi-step` option:

```python
@click.option("--multi-step", "multi_step", is_flag=True, help="Generate multi-step cases with structured diagnostics")
```

Add it to the function signature and conditionally use `MultiStepPipeline`:

```python
# Inside the generate function, after provider and retriever setup:
if multi_step:
    from casecrawler.generation.multi_step_pipeline import MultiStepPipeline
    pipeline = MultiStepPipeline(
        provider=provider, retriever=retriever,
        max_retries=cfg.generation.max_retries,
        review_threshold=cfg.generation.review_threshold,
    )
else:
    pipeline = GenerationPipeline(
        provider=provider, retriever=retriever,
        max_retries=cfg.generation.max_retries,
        review_threshold=cfg.generation.review_threshold,
    )
```

- [ ] **Step 2: Add `export` command to CLI**

Add a new command to `src/casecrawler/cli.py`:

```python
@cli.command()
@click.argument("output_path")
@click.option("--format", "export_format", type=click.Choice(["rl", "sft", "both"]), default="both")
@click.option("--difficulty", default=None, help="Filter by difficulty")
@click.option("--topic", default=None, help="Filter by topic")
@click.option("--min-accuracy", default=None, type=float, help="Minimum accuracy score")
@click.option("--include-wrong-paths", is_flag=True, help="Include wrong-path SFT variants")
def export(
    output_path: str,
    export_format: str,
    difficulty: str | None,
    topic: str | None,
    min_accuracy: float | None,
    include_wrong_paths: bool,
) -> None:
    """Export multi-step cases to training data formats."""
    import json

    from casecrawler.export.rl_exporter import export_rl_episode
    from casecrawler.export.sft_exporter import export_sft_conversation
    from casecrawler.storage.case_store import CaseStore

    cfg = get_config()
    store = CaseStore()
    cases = store.list_cases(topic=topic, difficulty=difficulty, min_accuracy=min_accuracy, limit=10000)
    multi_step_cases = [c for c in cases if c.is_multi_step()]

    if not multi_step_cases:
        click.echo("No multi-step cases found matching filters.")
        return

    exported = 0
    with open(output_path, "w") as f:
        for case in multi_step_cases:
            if export_format in ("rl", "both"):
                episode = export_rl_episode(case)
                f.write(json.dumps({"type": "rl_episode", **episode.model_dump()}) + "\n")
                exported += 1
            if export_format in ("sft", "both"):
                conv = export_sft_conversation(case)
                f.write(json.dumps({"type": "sft_conversation", **conv.model_dump()}) + "\n")
                exported += 1
                if include_wrong_paths:
                    wrong = export_sft_conversation(case, include_wrong_path=True)
                    f.write(json.dumps({"type": "sft_wrong_path", **wrong.model_dump()}) + "\n")
                    exported += 1

    click.echo(f"Exported {exported} records from {len(multi_step_cases)} cases to {output_path}")
```

- [ ] **Step 3: Add `multi_step` field to API GenerateRequest**

In `src/casecrawler/api/routes/generate.py`, add to `GenerateRequest`:

```python
    multi_step: bool = False
```

And update `run_generation` to use `MultiStepPipeline` when `multi_step=True`:

```python
# At the start of run_generation, after existing setup:
if multi_step:
    from casecrawler.generation.multi_step_pipeline import MultiStepPipeline
    pipeline = MultiStepPipeline(
        provider=provider, retriever=retriever,
        max_retries=cfg.generation.max_retries,
        review_threshold=cfg.generation.review_threshold,
    )
else:
    pipeline = GenerationPipeline(...)
```

- [ ] **Step 4: Run full test suite to verify nothing broke**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/casecrawler/cli.py src/casecrawler/api/routes/generate.py
git commit -m "feat: add --multi-step flag to generate and export CLI command"
```

---

### Task 16: Full Integration Verification

**Files:** No new files — verification only.

- [ ] **Step 1: Run the complete test suite**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m pytest -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Verify import chain works**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -c "from casecrawler.generation.multi_step_pipeline import MultiStepPipeline; from casecrawler.export.rl_exporter import export_rl_episode; from casecrawler.export.sft_exporter import export_sft_conversation; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Verify CLI commands are registered**

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m casecrawler.cli generate --help`
Expected: Output includes `--multi-step` flag

Run: `cd /Users/colin/conductor/workspaces/case-crawler/montevideo && python -m casecrawler.cli export --help`
Expected: Output includes `--format`, `--difficulty`, `--include-wrong-paths` options

- [ ] **Step 4: Commit any remaining fixes**

```bash
git add -A
git commit -m "chore: integration verification complete"
```
