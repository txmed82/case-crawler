from __future__ import annotations

import re

from pydantic import BaseModel

from casecrawler.models.synthetic import Code


class ImagingTemplate(BaseModel):
    modality: str
    valid_body_regions: list[str]
    terminology: dict[str, list[str]]
    report_format: str

def get_imaging_template(modality: str) -> ImagingTemplate | None:
    return IMAGING_TEMPLATES.get(modality)


def infer_imaging_labels(prompt: str, modality: str) -> list[Code]:
    normalized = _normalize(prompt)
    labels: list[Code] = []
    for label, terms in _LABEL_TERMS.items():
        if any(_contains_term(normalized, term) for term in terms):
            labels.append(
                Code(
                    system="https://casecrawler.dev/synthetic-radiology-labels",
                    code=_slug(label),
                    display=label,
                )
            )
    if not labels and "XR" == modality.upper() and "chest" in normalized:
        labels.append(
            Code(
                system="https://casecrawler.dev/synthetic-radiology-labels",
                code="no_acute_cardiopulmonary_abnormality",
                display="No acute cardiopulmonary abnormality",
            )
        )
    return labels


def build_imaging_report(
    *,
    prompt: str,
    modality: str,
    body_region: str,
    labels: list[Code],
) -> str:
    label_text = ", ".join(label.display for label in labels) or "No focal abnormality"
    template = get_imaging_template(modality.upper())
    format_hint = template.report_format if template else "findings -> impression"
    return (
        f"Synthetic {modality} {body_region} radiology report ({format_hint}). "
        f"Findings: {label_text}. Prompt context: {prompt}. "
        f"Impression: {label_text}."
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("-", " ").replace("_", " "))


def _contains_term(normalized_text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}\b", normalized_text) is not None


def _slug(value: str) -> str:
    return re.sub(r"\W+", "_", value.lower()).strip("_")


_LABEL_TERMS: dict[str, tuple[str, ...]] = {
    "Appendicitis": (
        "appendicitis",
        "dilated appendix",
        "appendiceal wall thickening",
    ),
    "Atelectasis": ("atelectasis", "volume loss", "linear opacity"),
    "Cardiomegaly": ("cardiomegaly", "enlarged heart", "enlarged cardiac silhouette"),
    "Consolidation": ("consolidation", "airspace disease"),
    "Fat stranding": (
        "fat stranding",
        "inflammatory stranding",
        "perinephric stranding",
    ),
    "Fracture": ("fracture",),
    "Hyperinflation": ("hyperinflation", "hyperinflated lungs"),
    "No acute intracranial hemorrhage": (
        "no acute intracranial hemorrhage",
        "no acute hemorrhage",
    ),
    "Opacity": ("opacity", "infiltrate"),
    "Perinephric stranding": ("perinephric stranding",),
    "Peripancreatic inflammation": (
        "peripancreatic inflammation",
        "peripancreatic edema",
        "pancreatitis",
    ),
    "Pleural effusion": ("pleural effusion", "effusion"),
    "Postictal seizure evaluation": ("postictal", "seizure evaluation"),
    "Pneumonia": ("pneumonia",),
    "Pneumothorax": ("pneumothorax",),
    "Pulmonary edema": ("pulmonary edema", "interstitial edema", "edema"),
    "Pyelonephritis": ("pyelonephritis", "striated nephrogram"),
}

IMAGING_TEMPLATES: dict[str, ImagingTemplate] = {
    "CT": ImagingTemplate(modality="CT", valid_body_regions=["head", "chest", "abdomen", "pelvis", "spine", "neck", "extremity"], terminology={"density": ["hyperdense", "hypodense", "isodense"], "enhancement": ["enhancing", "non-enhancing", "rim-enhancing"], "morphology": ["mass", "lesion", "collection", "effusion", "hemorrhage", "calcification"], "distribution": ["focal", "diffuse", "multifocal", "segmental"]}, report_format="findings → impression"),
    "MRI": ImagingTemplate(modality="MRI", valid_body_regions=["brain", "spine", "abdomen", "pelvis", "extremity", "chest", "neck"], terminology={"signal": ["hyperintense", "hypointense", "isointense"], "sequences": ["T1-weighted", "T2-weighted", "FLAIR", "DWI", "ADC", "post-contrast"], "findings": ["restricted diffusion", "enhancement", "edema", "mass effect", "herniation"], "morphology": ["mass", "lesion", "collection", "effusion"]}, report_format="findings → impression"),
    "XR": ImagingTemplate(modality="XR", valid_body_regions=["chest", "abdomen", "extremity", "spine", "pelvis"], terminology={"density": ["opacity", "lucency", "radiopaque", "radiolucent"], "findings": ["consolidation", "infiltrate", "effusion", "pneumothorax", "cardiomegaly", "fracture", "dislocation"], "distribution": ["focal", "diffuse", "bilateral", "unilateral", "lobar", "patchy"]}, report_format="findings → impression"),
    "US": ImagingTemplate(modality="US", valid_body_regions=["abdomen", "pelvis", "neck", "extremity", "chest", "cardiac"], terminology={"echogenicity": ["hyperechoic", "hypoechoic", "anechoic", "isoechoic", "heterogeneous"], "findings": ["mass", "collection", "free fluid", "thrombus", "calculus", "dilation"], "flow": ["hyperemic", "avascular", "reduced flow", "absent flow", "reversal of flow"]}, report_format="findings → impression"),
    "CTA": ImagingTemplate(modality="CTA", valid_body_regions=["head", "neck", "chest", "abdomen", "extremity"], terminology={"vascular": ["aneurysm", "stenosis", "occlusion", "dissection", "filling defect", "extravasation"], "density": ["hyperdense", "hypodense"], "morphology": ["saccular", "fusiform", "irregular", "smooth"]}, report_format="findings → impression"),
}
