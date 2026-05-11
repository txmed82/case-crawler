"""Pin the example configs in ``examples/configs/`` to ``AppConfig``.

Docs that link to ``yaml`` files have a habit of drifting. Each config
under ``examples/configs/`` is a copy-paste starter; if the schema
moves underneath them, the README claim "all three pass AppConfig
validation as written" stops being true. Catching that here is
cheaper than catching it in a user's first run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from casecrawler.models.config import AppConfig

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "configs"


@pytest.mark.parametrize(
    "config_filename",
    [
        "minimal-rag.yaml",
        "minimal-imaging.yaml",
        "dpo-export.yaml",
    ],
)
def test_example_config_loads_into_app_config(config_filename: str):
    config_path = EXAMPLES_DIR / config_filename
    assert config_path.exists(), f"missing example config: {config_path}"
    with config_path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    AppConfig(**raw)
