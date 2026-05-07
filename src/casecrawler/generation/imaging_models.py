from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImagingModelProfile:
    name: str
    model_id: str
    modality: str
    body_region: str
    prompt_prefix: str = ""
    default_negative_prompt: str | None = (
        "patient identifiers, text overlays, signatures, watermarks"
    )
    license: str | None = None
    notes: str = ""

    def render_prompt(self, finding_prompt: str) -> str:
        finding_prompt = finding_prompt.strip()
        if not self.prompt_prefix:
            return finding_prompt
        return f"{self.prompt_prefix.strip()} {finding_prompt}".strip()

    def is_compatible(self, modality: str, body_region: str) -> bool:
        profile_modality = self.modality.lower()
        profile_region = self.body_region.lower()
        requested_modality = modality.lower()
        requested_region = body_region.lower()
        modality_ok = profile_modality in {"medical_image", "multimodal", "any"}
        modality_ok = modality_ok or profile_modality == requested_modality
        region_ok = profile_region in {"unspecified", "multiregion", "any"}
        region_ok = region_ok or profile_region == requested_region
        return modality_ok and region_ok


IMAGING_MODEL_PROFILES: dict[str, ImagingModelProfile] = {
    "prompt2medimage": ImagingModelProfile(
        name="prompt2medimage",
        model_id="Nihirc/Prompt2MedImage",
        modality="medical_image",
        body_region="unspecified",
        prompt_prefix="high quality synthetic medical image:",
        license="wtfpl",
        notes="General medical image diffusion model fine-tuned on ROCO.",
    ),
    "mimic_cxr_editing": ImagingModelProfile(
        name="mimic_cxr_editing",
        model_id="IrohXu/stable-diffusion-mimic-cxr-v0.1",
        modality="XR",
        body_region="chest",
        prompt_prefix="frontal chest x-ray, radiology image:",
        license=None,
        notes="MIMIC-CXR fine-tune documented as better suited for x-ray editing.",
    ),
    "stable_diffusion_chest_xray": ImagingModelProfile(
        name="stable_diffusion_chest_xray",
        model_id="danyalmalik/stable-diffusion-chest-xray",
        modality="XR",
        body_region="chest",
        prompt_prefix="synthetic frontal chest x-ray:",
        license="creativeml-openrail-m",
        notes="Open chest X-ray Stable Diffusion profile from Hugging Face.",
    ),
    "cxr_normal_dreambooth": ImagingModelProfile(
        name="cxr_normal_dreambooth",
        model_id="chimbiwide/cxr-normal-dreambooth",
        modality="XR",
        body_region="chest",
        prompt_prefix="A chest xray of healthy normal lungs, clear lung fields.",
        license="openrail++",
        notes="DreamBooth chest x-ray profile for normal lung fields.",
    ),
    "cxr_pneumonia_dreambooth": ImagingModelProfile(
        name="cxr_pneumonia_dreambooth",
        model_id="chimbiwide/cxr-pneumonia-dreambooth",
        modality="XR",
        body_region="chest",
        prompt_prefix="A chest xray showing pneumonia infection, lung opacity.",
        license="openrail++",
        notes="DreamBooth chest x-ray profile for pneumonia-like opacity.",
    ),
    "symptom_xray_lora": ImagingModelProfile(
        name="symptom_xray_lora",
        model_id="Osama03/Medical-X-ray-image-generation-stable-diffusion",
        modality="XR",
        body_region="multiregion",
        prompt_prefix="medical x-ray scan:",
        license="openrail",
        notes="LoRA-based symptom-to-medical-image model.",
    ),
    "roentgen_v2_gated": ImagingModelProfile(
        name="roentgen_v2_gated",
        model_id="stanfordmimi/RoentGen-v2",
        modality="XR",
        body_region="chest",
        prompt_prefix="frontal chest x-ray:",
        license="restricted",
        notes="Gated model requiring MIMIC-CXR credentialing and accepted terms.",
    ),
}


def list_imaging_model_profiles() -> list[ImagingModelProfile]:
    return list(IMAGING_MODEL_PROFILES.values())


def resolve_imaging_model_profile(name: str | None) -> ImagingModelProfile | None:
    if name is None:
        return None
    try:
        return IMAGING_MODEL_PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(IMAGING_MODEL_PROFILES))
        raise ValueError(
            f"Unknown imaging model profile '{name}'. Available profiles: {available}."
        ) from exc
