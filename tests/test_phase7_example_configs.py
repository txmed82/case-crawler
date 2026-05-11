"""Pin the example configs in ``examples/configs/`` to ``AppConfig``.

Docs that link to ``yaml`` files have a habit of drifting. Each config
under ``examples/configs/`` is a copy-paste starter; if the schema
moves underneath them, the README claim "all three pass AppConfig
validation as written" stops being true. Catching that here is
cheaper than catching it in a user's first run.

The parametrize list is discovered from disk at collection time, so a
new starter dropped into ``examples/configs/`` is exercised
automatically — no need to remember to add the filename here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from casecrawler.models.config import AppConfig

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples" / "configs"
_DISCOVERED_CONFIGS: tuple[str, ...] = tuple(
    sorted(p.name for p in EXAMPLES_DIR.glob("*.yaml"))
)


def test_example_configs_directory_is_non_empty():
    """Belt-and-braces: if someone accidentally moves or renames
    ``examples/configs/``, the parametrize below would produce zero
    test cases and silently pass. Pin a minimum here."""

    assert _DISCOVERED_CONFIGS, (
        f"No example configs discovered under {EXAMPLES_DIR}. "
        "Phase 7 shipped three starter configs; if you removed them, "
        "delete this test instead of leaving a hollow assertion."
    )


@pytest.mark.parametrize("config_filename", _DISCOVERED_CONFIGS)
def test_example_config_loads_into_app_config(config_filename: str):
    config_path = EXAMPLES_DIR / config_filename
    with config_path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    AppConfig(**raw)
