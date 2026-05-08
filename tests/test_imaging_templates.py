# tests/test_imaging_templates.py
from casecrawler.generation.imaging_templates import (
    IMAGING_TEMPLATES,
    ImagingTemplate,
    build_imaging_report,
    get_imaging_template,
    infer_imaging_labels,
)

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


def test_infer_imaging_labels_from_prompt_synonyms():
    labels = infer_imaging_labels(
        "portable chest x-ray with right lower lobe opacity and effusion",
        modality="XR",
    )

    assert [label.display for label in labels] == ["Opacity", "Pleural effusion"]


def test_infer_imaging_labels_ignores_negated_findings():
    labels = infer_imaging_labels(
        "portable chest x-ray with no pneumothorax or pleural effusion",
        modality="XR",
    )

    assert [label.display for label in labels] == [
        "No acute cardiopulmonary abnormality"
    ]


def test_infer_imaging_labels_keeps_explicit_absence_labels():
    labels = infer_imaging_labels(
        "Noncontrast head CT with no acute intracranial hemorrhage",
        modality="CT",
    )

    assert [label.display for label in labels] == ["No acute intracranial hemorrhage"]


def test_infer_imaging_labels_for_abdominal_and_neuro_prompts():
    labels = infer_imaging_labels(
        "CT abdomen with dilated appendix, appendiceal wall thickening, fat stranding",
        modality="CT",
    )
    assert [label.display for label in labels] == ["Appendicitis", "Fat stranding"]

    labels = infer_imaging_labels(
        "CT abdomen with striated nephrogram and perinephric stranding pyelonephritis",
        modality="CT",
    )
    assert [label.display for label in labels] == [
        "Fat stranding",
        "Perinephric stranding",
        "Pyelonephritis",
    ]

    labels = infer_imaging_labels(
        "Noncontrast head CT no acute hemorrhage, postictal seizure evaluation",
        modality="CT",
    )
    assert [label.display for label in labels] == [
        "No acute intracranial hemorrhage",
        "Postictal seizure evaluation",
    ]


def test_build_imaging_report_mentions_labels_and_modality():
    labels = infer_imaging_labels("portable chest x-ray pulmonary edema", modality="XR")

    report = build_imaging_report(
        prompt="portable chest x-ray pulmonary edema",
        modality="XR",
        body_region="chest",
        labels=labels,
    )

    assert "Synthetic XR chest radiology report" in report
    assert "Pulmonary edema" in report
    assert "Impression:" in report
