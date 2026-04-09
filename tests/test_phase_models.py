from casecrawler.models.diagnostics import ImagingFinding, ImagingResult, LabResult, LabValue, VitalSigns
from casecrawler.models.phase import CasePhase, PhaseDecision, PhaseOutcome

def test_phase_decision_optimal():
    pd = PhaseDecision(action="Order non-contrast CT head", is_optimal=True, quality="optimal", reasoning="Most sensitive initial test for SAH", clinical_outcome="CT shows hyperdense material in basal cisterns", time_cost=None, leads_to_phase=None)
    assert pd.is_optimal is True
    assert pd.quality == "optimal"

def test_phase_decision_suboptimal():
    pd = PhaseDecision(action="Order MRI brain", is_optimal=False, quality="suboptimal", reasoning="MRI can detect SAH but takes longer and is less available in ED", clinical_outcome="Delayed diagnosis by ~2 hours, MRI eventually shows SAH", time_cost="delays diagnosis by ~2h", leads_to_phase=None)
    assert pd.quality == "suboptimal"
    assert pd.time_cost is not None

def test_phase_decision_catastrophic():
    pd = PhaseDecision(action="Discharge with migraine diagnosis", is_optimal=False, quality="catastrophic", reasoning="Misidentified as primary headache", clinical_outcome="Patient rebleeds at home within 24 hours", time_cost=None, leads_to_phase=None)
    assert pd.quality == "catastrophic"

def test_phase_outcome():
    po = PhaseOutcome(optimal_next_phase=2, patient_status="stable", narrative_transition="CT ordered stat. Patient resting in resus bay.")
    assert po.optimal_next_phase == 2
    assert po.patient_status == "stable"

def test_phase_outcome_terminal():
    po = PhaseOutcome(optimal_next_phase=None, patient_status="critical", narrative_transition="Patient discharged against medical advice. Found unresponsive at home 6 hours later.")
    assert po.optimal_next_phase is None

def test_case_phase_full():
    phase = CasePhase(phase_number=1, time_offset="T+0", narrative="A 42-year-old woman presents to the ED with sudden onset severe headache.", vitals=VitalSigns(hr=92, bp_systolic=168, bp_diastolic=95, rr=20, spo2=98.0, temp_c=37.4, gcs=14), lab_results=[], imaging_results=[], clinical_update=None, decisions=[PhaseDecision(action="Order non-contrast CT head", is_optimal=True, quality="optimal", reasoning="Most sensitive", clinical_outcome="SAH confirmed", time_cost=None, leads_to_phase=None), PhaseDecision(action="Discharge with migraine diagnosis", is_optimal=False, quality="catastrophic", reasoning="Misdiagnosis", clinical_outcome="Rebleed at home", time_cost=None, leads_to_phase=None)], phase_outcome=PhaseOutcome(optimal_next_phase=2, patient_status="stable", narrative_transition="CT ordered stat."))
    assert phase.phase_number == 1
    assert len(phase.decisions) == 2
    assert phase.vitals.hr == 92
    assert phase.phase_outcome.optimal_next_phase == 2

def test_case_phase_with_diagnostics():
    phase = CasePhase(phase_number=2, time_offset="T+45min", narrative="CT results are back. Labs still pending.", vitals=VitalSigns(hr=88, bp_systolic=155, bp_diastolic=90, rr=18, spo2=98.0, temp_c=37.2, gcs=14), lab_results=[LabResult(panel="CBC", values=[LabValue(name="WBC", value=9.8, unit="K/uL", reference_low=4.5, reference_high=11.0, flag=None), LabValue(name="Hgb", value=13.2, unit="g/dL", reference_low=12.0, reference_high=16.0, flag=None), LabValue(name="Plt", value=245.0, unit="K/uL", reference_low=150.0, reference_high=400.0, flag=None)], timestamp="T+40min")], imaging_results=[ImagingResult(modality="CT", body_region="head", indication="r/o SAH", findings=[ImagingFinding(structure="basal cisterns", observation="hyperdense material", severity="diffuse", laterality="bilateral")], impression="Acute subarachnoid hemorrhage, Fisher grade 3", timestamp="T+35min")], clinical_update="Nurse reports patient is photophobic and increasingly drowsy.", decisions=[PhaseDecision(action="Consult neurosurgery stat", is_optimal=True, quality="optimal", reasoning="CT-confirmed SAH requires urgent neurosurgical evaluation", clinical_outcome="Neurosurgery evaluates within 30 minutes", time_cost=None, leads_to_phase=None)], phase_outcome=PhaseOutcome(optimal_next_phase=3, patient_status="stable", narrative_transition="Neurosurgery consulted. CTA ordered to identify aneurysm."))
    assert len(phase.lab_results) == 1
    assert len(phase.imaging_results) == 1
    assert phase.clinical_update is not None
