from casecrawler.generation.imaging_generator import ImagingGenerator


def test_imaging_placeholder_does_not_require_diffusers(tmp_path):
    asset = ImagingGenerator().generate_placeholder(
        str(tmp_path),
        "portable chest x-ray with pulmonary edema",
    )

    assert asset.generation_backend == "placeholder"
    assert asset.modality == "XR"

