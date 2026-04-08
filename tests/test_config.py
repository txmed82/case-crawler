from casecrawler.config import load_config
from casecrawler.models.config import AppConfig


def test_load_default_config():
    config = load_config(config_path="/nonexistent/path.yaml")
    assert isinstance(config, AppConfig)
    assert config.ingestion.default_limit_per_source == 20
    assert config.chunking.default_chunk_size == 500
    assert config.embedding.model == "all-MiniLM-L6-v2"


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
