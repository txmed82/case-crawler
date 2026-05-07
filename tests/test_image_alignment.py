from casecrawler.models.synthetic import ImagingAsset
from casecrawler.validation.image_alignment import (
    BiomedCLIPImageValidator,
    ImageAlignmentValidator,
)


def _asset(file_path: str | None = "image.png") -> ImagingAsset:
    return ImagingAsset(
        image_id="img-1",
        modality="XR",
        body_region="chest",
        prompt="portable chest x-ray pulmonary edema",
        file_path=file_path,
        report_text="portable chest radiograph with pulmonary edema",
        generation_backend="unit-test",
    )


def test_lexical_image_alignment_scores_prompt_report_overlap():
    score = ImageAlignmentValidator().score(_asset())

    assert 0 < score <= 1


def test_biomedclip_validator_uses_injected_scorer():
    calls = []

    def scorer(image_path: str, report_text: str) -> float:
        calls.append((image_path, report_text))
        return 0.82

    score = BiomedCLIPImageValidator(scorer=scorer).score(_asset("xray.png"))

    assert score == 0.82
    assert calls == [("xray.png", "portable chest radiograph with pulmonary edema")]


def test_biomedclip_validator_clamps_scores_and_requires_file_path():
    validator = BiomedCLIPImageValidator(scorer=lambda _path, _text: 2.0)

    assert validator.score(_asset("xray.png")) == 1.0
    assert validator.score(_asset(None)) == 0.0


def test_biomedclip_validator_requires_imaging_extra(monkeypatch):
    def fake_require_package(import_name: str, extra: str):
        raise RuntimeError(f"Install casecrawler[{extra}] to use this backend.")

    monkeypatch.setattr(
        "casecrawler.validation.image_alignment.require_package",
        fake_require_package,
    )

    try:
        BiomedCLIPImageValidator().score(_asset("xray.png"))
    except RuntimeError as exc:
        assert "casecrawler[imaging]" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing imaging extra.")
