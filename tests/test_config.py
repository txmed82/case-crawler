from pathlib import Path

from casecrawler.config import load_config
from casecrawler.models.dataset import ExportFormat
from casecrawler.models.config import AppConfig


def test_load_default_config():
    config = load_config(config_path="/nonexistent/path.yaml")
    assert isinstance(config, AppConfig)
    assert config.ingestion.default_limit_per_source == 20
    assert config.chunking.default_chunk_size == 500
    assert config.embedding.model == "all-MiniLM-L6-v2"
    assert config.synthetic.validation_threshold == 0.8
    assert config.synthetic.image_output_dir == "./data/images"
    assert config.synthetic.imaging_model_profile is None
    assert config.synthetic.diffusers_model_id == "stabilityai/stable-diffusion-2-1"
    assert config.synthetic.time_series_backend == "deterministic"
    assert config.synthetic.export_formats == list(ExportFormat)
    assert not hasattr(config, "generation")


def test_load_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "ingestion:\n  default_limit_per_source: 50\nchunking:\n  default_chunk_size: 300\n"
    )
    config = load_config(config_path=str(config_file))
    assert config.ingestion.default_limit_per_source == 50
    assert config.chunking.default_chunk_size == 300
    # defaults still work for unset values
    assert config.embedding.model == "all-MiniLM-L6-v2"


def test_load_synthetic_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    image_output_dir = tmp_path / "images"
    synthea_executable = tmp_path / "synthea" / "run_synthea"
    config_file.write_text(
        "synthetic:\n"
        "  default_complexity: rare\n"
        "  validation_threshold: 0.9\n"
        "  imaging_model_profile: cxr_pneumonia_dreambooth\n"
        "  diffusers_model_id: test/medical-image-model\n"
        "  time_series_backend: external\n"
        "  time_series_model_profile: timediff\n"
        "  time_series_command:\n"
        "    - timediff-sample\n"
        f"  synthea_executable: {synthea_executable}\n"
        f"  image_output_dir: {image_output_dir}\n"
        "  max_api_generation_count: 10\n"
        "  max_api_returned_records: 3\n"
    )
    config = load_config(config_path=str(config_file))

    assert config.synthetic.default_complexity == "rare"
    assert config.synthetic.validation_threshold == 0.9
    assert config.synthetic.imaging_model_profile == "cxr_pneumonia_dreambooth"
    assert config.synthetic.diffusers_model_id == "test/medical-image-model"
    assert config.synthetic.time_series_backend == "external"
    assert config.synthetic.time_series_model_profile == "timediff"
    assert config.synthetic.time_series_command == ["timediff-sample"]
    assert config.synthetic.synthea_executable == str(synthea_executable)
    assert config.synthetic.image_output_dir == str(image_output_dir)
    assert config.synthetic.max_api_generation_count == 10
    assert config.synthetic.max_api_returned_records == 3


def test_example_config_exposes_all_export_profiles():
    config_path = Path(__file__).resolve().parents[1] / "config.example.yaml"
    config = load_config(config_path=str(config_path))

    assert config.synthetic.export_formats == list(ExportFormat)
    assert ExportFormat.NOTE_FACT_SFT_JSONL in config.synthetic.export_formats
    assert ExportFormat.FHIR_NDJSON in config.synthetic.export_formats
    assert ExportFormat.PARQUET in config.synthetic.export_formats
