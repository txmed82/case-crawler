from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImagingModelProfile:
    name: str
    model_id: str
    modality: str
    body_region: str
    adapter_type: str = "diffusers"
    prompt_prefix: str = ""
    default_negative_prompt: str | None = (
        "patient identifiers, text overlays, signatures, watermarks"
    )
    license: str | None = None
    gated: bool = False
    use_policy: str = "review_license_before_use"
    command_template: list[str] = field(default_factory=list)
    input_contract: dict[str, object] = field(default_factory=dict)
    output_contract: dict[str, object] = field(default_factory=dict)
    validation_requirements: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.command_template:
            object.__setattr__(
                self,
                "command_template",
                [
                    "casecrawler",
                    "generate-dataset",
                    "<topic>",
                    "--imaging-backend",
                    self.adapter_type,
                    "--imaging-model-profile",
                    self.name,
                ],
            )
        if not self.input_contract:
            object.__setattr__(
                self,
                "input_contract",
                {
                    "backend": self.adapter_type,
                    "inputs": [
                        "prompt",
                        "negative_prompt",
                        "modality",
                        "body_region",
                    ],
                    "prompt_policy": "profile.render_prompt(prompt)",
                    "negative_prompt_default": self.default_negative_prompt,
                    "output_dir": "synthetic.image_output_dir",
                },
            )
        if not self.output_contract:
            object.__setattr__(
                self,
                "output_contract",
                {
                    "artifact": "ImagingAsset",
                    "file_format": "png",
                    "fields": [
                        "image_id",
                        "modality",
                        "body_region",
                        "prompt",
                        "file_path",
                        "report_text",
                        "labels",
                        "generation_backend",
                    ],
                },
            )
        if not self.validation_requirements:
            object.__setattr__(
                self,
                "validation_requirements",
                [
                    "image_file_signature",
                    "image_dimensions_min_32x32",
                    "radiology_label_evidence",
                    "privacy_screen",
                    "image_text_alignment_if_configured",
                ],
            )

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
        use_policy="open_model_review_outputs_before_release",
        notes="General medical image diffusion model fine-tuned on ROCO.",
    ),
    "medisyn": ImagingModelProfile(
        name="medisyn",
        model_id="hiesingerlab/MediSyn",
        modality="medical_image",
        body_region="multiregion",
        prompt_prefix="synthetic medical image:",
        license="cc-by-nc-nd-4.0",
        use_policy="non_commercial_no_derivatives_review_before_release",
        notes=(
            "Generalist text-guided latent diffusion model for public-domain "
            "medical image synthesis across multiple specialties and image types."
        ),
    ),
    "chexgenbench_sana_e20": ImagingModelProfile(
        name="chexgenbench_sana_e20",
        model_id="raman07/CheXGenBench-Models-Sana-e20",
        modality="XR",
        body_region="chest",
        prompt_prefix="synthetic frontal chest radiograph:",
        license=None,
        use_policy="model_card_missing_review_terms_and_validate_privacy_utility",
        notes=(
            "CheXGenBench Sana 0.6B chest-radiograph model profile; model page "
            "currently has sparse licensing details, so release packages should "
            "review terms and run privacy, fidelity, and clinical utility checks."
        ),
    ),
    "mimic_cxr_editing": ImagingModelProfile(
        name="mimic_cxr_editing",
        model_id="IrohXu/stable-diffusion-mimic-cxr-v0.1",
        modality="XR",
        body_region="chest",
        prompt_prefix="frontal chest x-ray, radiology image:",
        license=None,
        use_policy="license_unspecified_review_before_use",
        notes="MIMIC-CXR fine-tune documented as better suited for x-ray editing.",
    ),
    "stable_diffusion_chest_xray": ImagingModelProfile(
        name="stable_diffusion_chest_xray",
        model_id="danyalmalik/stable-diffusion-chest-xray",
        modality="XR",
        body_region="chest",
        prompt_prefix="synthetic frontal chest x-ray:",
        license="creativeml-openrail-m",
        use_policy="openrail_review_outputs_before_release",
        notes="Open chest X-ray Stable Diffusion profile from Hugging Face.",
    ),
    "cxr_normal_dreambooth": ImagingModelProfile(
        name="cxr_normal_dreambooth",
        model_id="chimbiwide/cxr-normal-dreambooth",
        modality="XR",
        body_region="chest",
        prompt_prefix="A chest xray of healthy normal lungs, clear lung fields.",
        license="openrail++",
        use_policy="openrail_review_outputs_before_release",
        notes="DreamBooth chest x-ray profile for normal lung fields.",
    ),
    "cxr_pneumonia_dreambooth": ImagingModelProfile(
        name="cxr_pneumonia_dreambooth",
        model_id="chimbiwide/cxr-pneumonia-dreambooth",
        modality="XR",
        body_region="chest",
        prompt_prefix="A chest xray showing pneumonia infection, lung opacity.",
        license="openrail++",
        use_policy="openrail_review_outputs_before_release",
        notes="DreamBooth chest x-ray profile for pneumonia-like opacity.",
    ),
    "symptom_xray_lora": ImagingModelProfile(
        name="symptom_xray_lora",
        model_id="Osama03/Medical-X-ray-image-generation-stable-diffusion",
        modality="XR",
        body_region="multiregion",
        prompt_prefix="medical x-ray scan:",
        license="openrail",
        use_policy="openrail_review_outputs_before_release",
        notes="LoRA-based symptom-to-medical-image model.",
    ),
    "roentgen_v2_gated": ImagingModelProfile(
        name="roentgen_v2_gated",
        model_id="stanfordmimi/RoentGen-v2",
        modality="XR",
        body_region="chest",
        prompt_prefix="frontal chest x-ray:",
        license="restricted",
        gated=True,
        use_policy="credentialed_mimic_cxr_terms_required",
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
