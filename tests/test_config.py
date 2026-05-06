from casecrawler.config import load_config
from casecrawler.models.config import AppConfig


def test_load_default_config():
    config = load_config(config_path="/nonexistent/path.yaml")
    assert isinstance(config, AppConfig)
    assert config.ingestion.default_limit_per_source == 20
    assert config.chunking.default_chunk_size == 500
    assert config.embedding.model == "all-MiniLM-L6-v2"
    assert config.synthetic.validation_threshold == 0.8
    assert config.synthetic.image_output_dir == "./data/images"


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
        f"  synthea_executable: {synthea_executable}\n"
        f"  image_output_dir: {image_output_dir}\n"
        "  max_api_generation_count: 10\n"
        "  max_api_returned_records: 3\n"
    )
    config = load_config(config_path=str(config_file))

    assert config.synthetic.default_complexity == "rare"
    assert config.synthetic.validation_threshold == 0.9
    assert config.synthetic.synthea_executable == str(synthea_executable)
    assert config.synthetic.image_output_dir == str(image_output_dir)
    assert config.synthetic.max_api_generation_count == 10
    assert config.synthetic.max_api_returned_records == 3
