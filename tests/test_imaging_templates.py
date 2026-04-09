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
