# MESH — Neuron EM Overlap Viewer Pipeline

Interactive pipeline for investigating potential gap junctions between MOT/MOS and VS/HS neurons in the *Drosophila* brain, using the FlyWire EM connectomics dataset.

## Overview

Analyses pairwise spatial overlaps between 18 neurons in the *Drosophila* optic lobe and provides an interactive 3D + EM viewer for manual proofreading of candidate gap-junction sites.

**Neurons (configured in `neurons.json`):**
- **4 VS neurons** — Vertical System: VS1-VS4, L/R
- **6 HS neurons** — Horizontal System: HSN, HSE, HSS, L/R
- **4 Motor neurons** — MOT_L/R, MOS_L/R

**Key features:**
- Pairwise overlap analysis at configurable distance thresholds
- **Spatial clustering** of overlap faces — disconnected contact regions (>10 um apart) become separate viewer entries
- Interactive 3D HTML viewer (Plotly) with EM snapshot panel + Z-stack navigation
- Per-slice EM images centered on the actual overlap location (not averaged centroids)
- Overlap deletion, area recalculation, auto-advance navigation
- 2D skeleton projection plots

## Pipeline Scripts

Run all scripts in order with **`python run_all.py`**, or run individually:

| # | Script | Description |
|---|--------|-------------|
| 1 | **`overlap_analysis.py`** | Core analysis: downloads 18 neuron meshes from FlyWire, computes all pairwise overlaps at 0.1 um threshold, identifies contact patches (Top 1-6 per pair) and synaptic connections. Saves results to `comprehensive_overlap_results_YYYY-MM-DD/`. Produces: `all_results_combined.csv`, `synapses.csv`, `geometric_data/contact_faces.csv`, `geometric_data/contact_vertices.csv`, overlap matrices, and 3D Plotly figures. Runtime: ~2 h first run (mesh download), ~30 min cached. |
| 2 | **`generate_skeleton_plots.py`** | Generates 2D PNG projection plots of neuron skeletons with overlap faces highlighted (red, `#FF0030`). One plot per neuron pair with overlap. No legends. |
| 3 | **`generate_em_stacks.py`** | Downloads EM snapshots with coloured segmentation overlays from CloudVolume. **Spatially clusters** overlap faces per pair (10 um threshold via `scipy.cluster.hierarchy`) so disconnected contact regions get separate indices and per-slice centroids. Also downloads contact-patch Z-stacks (+/-20 slices) and synapse Z-stacks (+/-20 slices). Writes `overlap_em_meta.json` (consumed by the viewer). |
| 4 | **`skeleton_em_viewer.py`** | Generates the final self-contained HTML viewer (`skeleton_em_viewer.html`). 3D Plotly scene with neuron meshes, overlap faces (Mesh3d), contacts, synapses + right-side EM panel with Z-slider, delete buttons, area matrix, auto-advance. Reads `overlap_em_meta.json` and images from `em_snaps/`. |

### Supporting files

| File | Description |
|------|-------------|
| **`neurons.json`** | Central neuron configuration — FlyWire IDs, names, groups, RGB colours. All scripts read from this single file (no hardcoded neuron lists). |
| **`run_all.py`** | Orchestrator — runs all 4 pipeline scripts in sequence, auto-loads FlyWire token from `cave-secret.json`. |
| **`requirements.txt`** | Python dependencies. |

## Color Palette

| Group | Color | Hex |
|-------|-------|-----|
| MOT | Purple | `#5E3C99` |
| MOS | Green | `#4D9221` |
| VS | Orange | `#D14900` |
| HS | Magenta | `#C51B7D` |

## Requirements

### Python

Python 3.10+ (tested with 3.13). Install dependencies:

```bash
pip install -r requirements.txt
```

Core packages: `numpy`, `pandas`, `plotly`, `Pillow`, `navis`, `fafbseg`, `cloud-volume`, `trimesh`, `matplotlib`, `scipy`, `tqdm`

### FlyWire CAVE Token

Required for mesh download and EM data access. Set **before** running:

```bash
# Linux / macOS
export FLYWIRE_TOKEN="your_token_here"

# Windows PowerShell
$Env:FLYWIRE_TOKEN = "your_token_here"
```

Alternatively, place the token in `~/.cloudvolume/secrets/cave-secret.json`:
```json
{"token": "your_token_here"}
```

`run_all.py` will auto-load it from there.

Obtain a token from the [FlyWire CAVE portal](https://global.daf-apis.com/info/).

### Internet Connection

First run downloads ~1.4 GB of neuron meshes + ~3,400 overlap EM snapshots + contact/synapse stacks. Subsequent runs use cached data and skip existing images.

## Installation

```bash
git clone https://github.com/ETigerschuss/MESH.git
cd MESH

# (Optional) create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## Usage

### Full pipeline

```bash
python run_all.py
```

### Individual scripts

```bash
python overlap_analysis.py        # Step 1: mesh download + overlap analysis
python generate_skeleton_plots.py # Step 2: 2D skeleton plots
python generate_em_stacks.py      # Step 3: EM snapshot download (+ spatial clustering)
python skeleton_em_viewer.py      # Step 4: generate HTML viewer
```

Then open `comprehensive_overlap_results_YYYY-MM-DD/skeleton_em_viewer.html` in a browser.

## Viewer Features

### Left Panel — Neuron Controls
- Toggle visibility per neuron (mesh, contacts, synapses, overlaps)
- Overlap area matrix with clickable cells

### Center Panel — 3D Scene
- Neuron meshes + overlap face triangles (Mesh3d)
- Contacts (red circles), synapses (yellow markers)
- Click any overlap face -> jumps to EM panel for that cluster
- Diamond-shaped 3D position indicator

### Right Panel — EM Snapshots
- Segmentation overlay (source + target neuron colours)
- Z-slider with per-cluster valid slices
- Delete Slice / Delete All buttons (removes overlaps from area calculation)
- Auto-advance to next overlap after deletion
- Previous / Next navigation across all visible items

## Project Structure

```
MESH/
├── neurons.json                  # Neuron IDs, names, groups, colours
├── run_all.py                    # Pipeline orchestrator
├── overlap_analysis.py           # Step 1: overlap analysis
├── generate_skeleton_plots.py    # Step 2: 2D skeleton plots
├── generate_em_stacks.py         # Step 3: EM snapshot download + clustering
├── skeleton_em_viewer.py         # Step 4: HTML viewer generator
├── requirements.txt              # Python dependencies
├── .gitignore                    # Excludes large data files
└── README.md                     # This file

Generated outputs (not in repository):
comprehensive_overlap_results_YYYY-MM-DD/
├── skeleton_em_viewer.html       # Interactive viewer (open in browser)
├── all_results_combined.csv      # Contact patch data
├── synapses.csv                  # Synapse data
├── overlap_em_meta.json          # Overlap cluster metadata (55 entries)
├── geometric_data/               # contact_faces.csv, contact_vertices.csv
├── em_snaps/                     # ~21,000 PNG snapshots
├── neuron_meshes/                # OBJ files (~1.4 GB)
├── individual_patches/           # Per-pair patch CSVs
└── skeleton_plots/               # 2D skeleton PNGs
```

## Dataset Information

- **FlyWire dataset:** FAFB v141 (flywire_783)
- **EM volume:** `https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14` (MIP 1, 8x8x40 nm)
- **Segmentation:** `precomputed://gs://flywire_v141_m783` (MIP 0, 16x16x40 nm)
- **Z-stack depth:** +/-20 slices (40 nm/slice = +/-800 nm total)
- **Snapshot size:** 512x512 px (4,096x4,096 nm)
- **Spatial clustering threshold:** 10 um (faces farther apart become separate overlap entries)

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "FlyWire token missing" | Set `FLYWIRE_TOKEN` env var or place token in `~/.cloudvolume/secrets/cave-secret.json` |
| "No module named X" | `pip install -r requirements.txt` |
| CloudVolume connection errors | Check internet + Google Cloud Storage access; `pip install --upgrade cloud-volume` |
| Viewer shows no EM snapshots | Run `python generate_em_stacks.py` first, or check `em_snaps/` folder exists |
| HTML file too large to open | Normal — viewer is ~18 MB; use Chrome/Firefox |

## Citation

- **FlyWire dataset:** Dorkenwald et al. (2023) *Nature*
- **FAFB dataset:** Zheng et al. (2018) *Cell*
