#!/usr/bin/env python
"""Shared helpers for loading MESH neuron configuration files."""

import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "neurons.json"
CONFIG_ENV_VAR = "MESH_NEURON_CONFIG"


def resolve_config_path(config_path=None):
    """Resolve the active neuron-config path from args or environment."""
    raw_path = config_path or os.environ.get(CONFIG_ENV_VAR)
    path = Path(raw_path).expanduser() if raw_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def load_config(config_path=None):
    """Load and return the active neuron configuration and its path."""
    path = resolve_config_path(config_path)
    with path.open('r', encoding='utf-8') as handle:
        cfg = json.load(handle)
    return cfg, path


def config_is_default(config_path=None):
    """Return True when the active config is the repo default."""
    return resolve_config_path(config_path) == DEFAULT_CONFIG_PATH.resolve()