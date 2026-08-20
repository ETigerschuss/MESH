#!/usr/bin/env python
"""Shared helpers for loading MESH neuron configuration files.

COORDINATE CONVENTIONS (read before touching any geometry/EM code)
------------------------------------------------------------------
All spatial data in this pipeline is stored in **nanometres**.

* Neuroglancer / FlyWire voxel coordinates use a **4 x 4 x 40 nm** voxel
  (x, y, z). To convert a voxel you read in Neuroglancer to nm, multiply by
  (4, 4, 40). This is the convention for the confirmed-GJ coordinate and every
  EM-snapshot filename tag (``vx../vy../vz..``). NB: it is *not* 4x4x80 — that
  mismatch caused an early mislocalisation; always use 40 nm in z.
* EM volume (bossDB FAFB v14, mip 1): **8 nm/px** in x,y and **40 nm** per z
  slice. A 512 px crop = 4096 nm = 4.096 um. Scalebar: 1 um = 125 px.
* Segmentation volume (flywire_v141_m783, mip 0): **16 nm/px** x,y, 40 nm z
  (upsampled 2x onto the EM grid for overlays).
* Contact/overlap areas are in **um^2**. Two numbers coexist: the geometric
  *mesh* area (proximity-based, over-counts ~30%) and the seg-adjacency
  *confined* area (``overlap_validation.json``) — the confined value is the
  authoritative one for the paper.
"""

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