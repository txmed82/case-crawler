import struct
import zlib

from casecrawler.models.synthetic import Code, ImagingAsset
from casecrawler.validation.image_alignment import (
    BiomedCLIPImageValidator,
    ImageAlignmentValidator,
    MedGemmaImageTextValidator,
    validate_image_file_asset,
    validate_radiology_label_consistency,
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


def test_medgemma_validator_uses_injected_json_analyzer():
    calls = []

    def analyzer(image_path: str, report_text: str) -> str:
        calls.append((image_path, report_text))
        return '{"score": 0.73, "rationale": "report matches edema"}'

    score = MedGemmaImageTextValidator(analyzer=analyzer).score(_asset("xray.png"))

    assert score == 0.73
    assert calls == [("xray.png", "portable chest radiograph with pulmonary edema")]


def test_medgemma_validator_accepts_numeric_and_dict_analyzer_outputs():
    assert MedGemmaImageTextValidator(analyzer=lambda _path, _text: 0.81).score(
        _asset("xray.png")
    ) == 0.81
    assert MedGemmaImageTextValidator(
        analyzer=lambda _path, _text: {"score": 1.7}
    ).score(_asset("xray.png")) == 1.0


def test_medgemma_validator_handles_unparseable_output_and_requires_file_path():
    validator = MedGemmaImageTextValidator(analyzer=lambda _path, _text: "unclear")

    assert validator.score(_asset("xray.png")) == 0.0
    assert validator.score(_asset(None)) == 0.0


def test_medgemma_validator_requires_hf_extra(monkeypatch):
    def fake_require_package(import_name: str, extra: str):
        raise RuntimeError(f"Install casecrawler[{extra}] to use this backend.")

    monkeypatch.setattr(
        "casecrawler.validation.image_alignment.require_package",
        fake_require_package,
    )

    try:
        MedGemmaImageTextValidator().score(_asset("xray.png"))
    except RuntimeError as exc:
        assert "casecrawler[hf]" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing hf extra.")


def test_radiology_label_consistency_accepts_supported_labels():
    asset = _asset()
    asset = asset.model_copy(
        update={
            "labels": [
                Code(
                    system="synthetic",
                    code="pulmonary_edema",
                    display="Pulmonary edema",
                )
            ]
        }
    )

    issues = validate_radiology_label_consistency(asset)

    assert issues == []


def test_radiology_label_consistency_flags_missing_label_evidence():
    asset = _asset().model_copy(
        update={
            "labels": [
                Code(
                    system="synthetic",
                    code="pneumothorax",
                    display="Pneumothorax",
                )
            ]
        }
    )

    issues = validate_radiology_label_consistency(asset)

    assert len(issues) == 1
    assert "Pneumothorax" in issues[0]


def test_radiology_label_consistency_flags_negated_label():
    asset = _asset().model_copy(
        update={
            "report_text": "Portable chest radiograph without pneumothorax.",
            "labels": [
                Code(
                    system="synthetic",
                    code="pneumothorax",
                    display="Pneumothorax",
                )
            ],
        }
    )

    issues = validate_radiology_label_consistency(asset)

    assert len(issues) == 1
    assert "negated" in issues[0]


def test_validate_image_file_asset_accepts_supported_image_signature(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png_bytes(width=64, height=64))

    issues = validate_image_file_asset(_asset(str(image_path)))

    assert issues == []


def test_validate_image_file_asset_rejects_tiny_raster_image(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png_bytes(width=16, height=16))

    issues = validate_image_file_asset(_asset(str(image_path)))

    assert any(issue.field == "imaging.img-1.dimensions" for issue in issues)
    assert any("at least 32x32" in issue.message for issue in issues)


def test_validate_image_file_asset_rejects_invalid_image_signature(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not a png")

    issues = validate_image_file_asset(_asset(str(image_path)))

    assert any(issue.field == "imaging.img-1.file_signature" for issue in issues)


def _png_bytes(*, width: int, height: int) -> bytes:
    raw = b"".join(b"\x00" + (b"\x80" * width) for _ in range(height))
    chunks = [
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw)),
        _png_chunk(b"IEND", b""),
    ]
    return b"".join(chunks)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )
