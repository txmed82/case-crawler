from casecrawler.models.diagnostics import (
    ImagingFinding,
    ImagingResult,
    LabResult,
    LabValue,
    VitalSigns,
)


def test_vital_signs_basic():
    vs = VitalSigns(
        hr=88,
        bp_systolic=142,
        bp_diastolic=88,
        rr=18,
        spo2=97.0,
        temp_c=37.2,
        gcs=15,
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
        name="WBC",
        value=18.5,
        unit="K/uL",
        reference_low=4.5,
        reference_high=11.0,
        flag="H",
    )
    assert lv.flag == "H"
    assert lv.value > lv.reference_high


def test_lab_value_no_flag():
    lv = LabValue(
        name="Na",
        value=140.0,
        unit="mEq/L",
        reference_low=136.0,
        reference_high=145.0,
        flag=None,
    )
    assert lv.flag is None


def test_lab_result_panel():
    lr = LabResult(
        panel="CBC",
        values=[
            LabValue(
                name="WBC",
                value=7.2,
                unit="K/uL",
                reference_low=4.5,
                reference_high=11.0,
                flag=None,
            ),
            LabValue(
                name="Hgb",
                value=13.5,
                unit="g/dL",
                reference_low=12.0,
                reference_high=16.0,
                flag=None,
            ),
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
