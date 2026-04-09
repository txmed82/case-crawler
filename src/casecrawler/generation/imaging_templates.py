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
    "CT": ImagingTemplate(modality="CT", valid_body_regions=["head", "chest", "abdomen", "pelvis", "spine", "neck", "extremity"], terminology={"density": ["hyperdense", "hypodense", "isodense"], "enhancement": ["enhancing", "non-enhancing", "rim-enhancing"], "morphology": ["mass", "lesion", "collection", "effusion", "hemorrhage", "calcification"], "distribution": ["focal", "diffuse", "multifocal", "segmental"]}, report_format="findings → impression"),
    "MRI": ImagingTemplate(modality="MRI", valid_body_regions=["brain", "spine", "abdomen", "pelvis", "extremity", "chest", "neck"], terminology={"signal": ["hyperintense", "hypointense", "isointense"], "sequences": ["T1-weighted", "T2-weighted", "FLAIR", "DWI", "ADC", "post-contrast"], "findings": ["restricted diffusion", "enhancement", "edema", "mass effect", "herniation"], "morphology": ["mass", "lesion", "collection", "effusion"]}, report_format="findings → impression"),
    "XR": ImagingTemplate(modality="XR", valid_body_regions=["chest", "abdomen", "extremity", "spine", "pelvis"], terminology={"density": ["opacity", "lucency", "radiopaque", "radiolucent"], "findings": ["consolidation", "infiltrate", "effusion", "pneumothorax", "cardiomegaly", "fracture", "dislocation"], "distribution": ["focal", "diffuse", "bilateral", "unilateral", "lobar", "patchy"]}, report_format="findings → impression"),
    "US": ImagingTemplate(modality="US", valid_body_regions=["abdomen", "pelvis", "neck", "extremity", "chest", "cardiac"], terminology={"echogenicity": ["hyperechoic", "hypoechoic", "anechoic", "isoechoic", "heterogeneous"], "findings": ["mass", "collection", "free fluid", "thrombus", "calculus", "dilation"], "flow": ["hyperemic", "avascular", "reduced flow", "absent flow", "reversal of flow"]}, report_format="findings → impression"),
    "CTA": ImagingTemplate(modality="CTA", valid_body_regions=["head", "neck", "chest", "abdomen", "extremity"], terminology={"vascular": ["aneurysm", "stenosis", "occlusion", "dissection", "filling defect", "extravasation"], "density": ["hyperdense", "hypodense"], "morphology": ["saccular", "fusiform", "irregular", "smooth"]}, report_format="findings → impression"),
}
