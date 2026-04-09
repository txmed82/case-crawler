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
    "troponin": LabPanel(name="troponin", components=[LabComponent(name="Troponin I", unit="ng/mL", reference_low=0.0, reference_high=0.04, critical_high=0.4, precision=3)]),
    "lipase": LabPanel(name="lipase", components=[LabComponent(name="Lipase", unit="U/L", reference_low=0.0, reference_high=160.0, precision=0)]),
    "d_dimer": LabPanel(name="d_dimer", components=[LabComponent(name="D-dimer", unit="ng/mL", reference_low=0.0, reference_high=500.0, precision=0)]),
    "BNP": LabPanel(name="BNP", components=[LabComponent(name="BNP", unit="pg/mL", reference_low=0.0, reference_high=100.0, precision=0)]),
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
