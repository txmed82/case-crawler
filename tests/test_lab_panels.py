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
    expected = ["CBC", "BMP", "CMP", "coags", "ABG", "CSF", "troponin", "lipase", "d_dimer", "BNP", "UA", "LFTs", "thyroid", "iron_studies"]
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
    bmp = LAB_PANELS["BMP"]
    k = next(c for c in bmp.components if c.name == "K")
    assert k.critical_low is not None
    assert k.critical_high is not None

def test_cmp_contains_bmp_components():
    bmp_names = {c.name for c in LAB_PANELS["BMP"].components}
    cmp_names = {c.name for c in LAB_PANELS["CMP"].components}
    assert bmp_names.issubset(cmp_names)
