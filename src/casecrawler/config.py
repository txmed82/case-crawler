from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from casecrawler.models.config import AppConfig

_config: AppConfig | None = None


def load_config(config_path: str | None = None) -> AppConfig:
    global _config

    load_dotenv()

    if config_path is None:
        candidates = ["config.yaml", "config.yml"]
        for candidate in candidates:
            if Path(candidate).exists():
                config_path = candidate
                break

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        _config = AppConfig(**raw)
    else:
        _config = AppConfig()

    return _config


def get_config() -> AppConfig:
    if _config is None:
        return load_config()
    return _config


def get_env(key: str) -> str | None:
    return os.environ.get(key)
