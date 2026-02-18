# FlyWire Neuron Overlap Analysis Toolkit

Interactive 3D visualization toolkit for analyzing neuron overlaps in the *Drosophila* visual system using the FlyWire connectomics dataset.

## Overview

This toolkit analyzes pairwise spatial overlaps between 20 neurons in the *Drosophila* optic lobe:
- **8 VS neurons** (Vertical System: VS1–VS4, L/R)
- **6 HS neurons** (Horizontal System: HSN, HSE, HSS, L/R)
- **4 Motor neurons** (MOT, MOS, L/R)
- **2 BIPS neurons** (BIPS, L/R)

**Key Features:**
- Pairwise overlap analysis at configurable distance thresholds
- Interactive 3D HTML viewer with EM snapshots and Z-stack navigation
- Contact patch and synapse identification
- Skeleton/volume overlap plots
- Neuron mesh export (OBJ format)

## Color Palette

All scripts use a consistent color scheme:

| Group | Color | Hex |
|-------|-------|-----|
| MOS   | Green   | `#4D9221` |
| MOT   | Purple  | `#5E3C99` |
| VS    | Orange  | `#D14900` |
| HS    | Magenta | `#C51B7D` |
| BIPS  | Black   | `#000000` |

## Requirements

### Python Dependencies

```bash
pip install -r requirements.txt
```

Core packages: `numpy`, `pandas`, `plotly`, `Pillow`, `navis`, `fafbseg`, `cloud-volume`, `trimesh`, `matplotlib`, `scipy`, `tqdm`

### FlyWire Data Access

A FlyWire CAVE token is required for downloading neuron data. Set it as an environment variable **before** running the analysis:

```bash
# Linux / macOS
export FLYWIRE_TOKEN="your_token_here"

# Windows PowerShell
$Env:FLYWIRE_TOKEN = "your_token_here"

# Windows CMD
set FLYWIRE_TOKEN=your_token_here
```

You can obtain a token from the [FlyWire CAVE portal](https://global.daf-apis.com/info/).

### Internet Connection

First run downloads ~1.4 GB of neuron meshes and EM data. Subsequent runs use cached data.

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/flywire-overlap-analysis.git
cd flywire-overlap-analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_toolkit.py
```

## Usage

### 1. Main Analysis

```bash
export FLYWIRE_TOKEN="your_token_here"
python overlap_analysis.py
```

This runs the full pipeline:
1. Downloads 20 neuron meshes from FlyWire (FAFB v141)
2. Analyzes pairwise overlaps at 0.1 um threshold
3. Identifies contact patches and synaptic connections
4. Exports meshes and generates interactive visualizations

**Runtime:** ~2 hours (first run, depending on network speed)

**Output:** `comprehensive_overlap_results_YYYY-MM-DD/` directory

### 2. Interactive EM Viewer

```bash
python em_viewer.py
```

Generates `em_viewer.html` with:
- 3D point-cloud viewer with neuron meshes, contacts, synapses
- EM snapshot panel with segmentation overlay
- **Z-stack navigation** (+-20 slices, +-800nm depth per contact/synapse)
- Auto-downloads EM snapshots from CloudVolume if missing

### 3. Skeleton Overlap Plots

```bash
python generate_skeleton_plots.py
```

Generates 2D PNG plots of neuron skeletons with overlap regions highlighted.

### 4. Skeleton EM Viewer

```bash
python skeleton_em_viewer.py
```

Alternative viewer using navis skeleton rendering instead of point clouds.

### 5. Batch EM Stack Generation (Optional)

```bash
python generate_em_stacks.py
```

Pre-downloads all EM snapshots with Z-stacks. Optional -- the EM viewer auto-downloads on first run.

## Configuration

### Results Directory

All scripts auto-detect the latest `comprehensive_overlap_results_*` directory. Override with:

```bash
export MESH_RESULTS_DIR="my_custom_results_dir"
python em_viewer.py
```

### Neuron Selection

Edit the `NEURON_IDS` dictionary at the top of each script to add/remove neurons.

## Interactive Viewer Features

### Left Panel: Neuron Controls
- Toggle visibility for meshes, contacts, and synapses per neuron

### Center Panel: 3D Mesh Viewer
- Interactive rotation, zoom, pan
- Click contact (red circles) or synapse (yellow) markers
- Selected point highlights in 3D view

### Right Panel: EM Snapshot Viewer
- Segmentation overlay with neuron-colored regions
- **Z-stack slider** (+-20 slices, +-800nm depth)
- Previous/Next item navigation filtered to visible neurons
- Resizable panels via draggable borders

## Project Structure

```
flywire-overlap-analysis/
├── overlap_analysis.py          # Main analysis pipeline
├── em_viewer.py                 # Interactive EM viewer generator
├── skeleton_em_viewer.py        # Skeleton-based EM viewer
├── generate_skeleton_plots.py   # 2D skeleton overlap plots
├── generate_em_stacks.py        # Batch EM snapshot generator
├── test_toolkit.py              # Installation verification
├── test_em_viewer.py            # Quick viewer test
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── .gitignore                   # Git exclusions

Generated outputs (not in repository):
comprehensive_overlap_results_YYYY-MM-DD/
├── em_viewer.html               # Interactive 3D viewer
├── skeleton_em_viewer.html      # Skeleton-based viewer
├── all_results_combined.csv     # Contact patch data
├── synapses.csv                 # Synapse data
├── neuron_meshes/               # 20 OBJ files (~1.4 GB)
├── em_snaps/                    # PNG snapshots with Z-stacks
├── geometric_data/              # Contact face triangles
├── individual_patches/          # Per-pair patch data
└── [additional plots and matrices]
```

## Dataset Information

- **FlyWire dataset:** FAFB v141 (flywire_783)
- **EM volume:** `https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14` (MIP 1, 8x8x40 nm)
- **Segmentation:** `precomputed://gs://flywire_v141_m783` (MIP 0, 16x16x40 nm)
- **Z-stack depth:** +-20 slices (40 nm/slice = +-800 nm total)
- **Snapshot size:** 512x512 pixels (4096x4096 nm)

## Testing

```bash
# Verify installation and dependencies
python test_toolkit.py

# Test EM viewer generation (requires existing results)
python test_em_viewer.py
```

## Troubleshooting

### "FlyWire token missing"
Set the `FLYWIRE_TOKEN` environment variable before running. See FlyWire Data Access above.

### "No module named X"
```bash
pip install -r requirements.txt
```

### CloudVolume connection errors
- Check internet connection
- Verify Google Cloud Storage access (needed for segmentation volume)
- Try: `pip install --upgrade cloud-volume`

### Viewer shows no EM snapshots
- Delete `em_snaps/` folder and re-run `python em_viewer.py` to re-download
- Check CloudVolume is installed: `python -c "from cloudvolume import CloudVolume; print('OK')"`

### Large file sizes
- `em_viewer.html` can be 100+ MB (embedded EM images)
- Do NOT commit `comprehensive_overlap_results_*/` to git
- `.gitignore` is configured to exclude large files

## Citation

If you use this toolkit, please cite:

- **FlyWire dataset:** Dorkenwald et al. (2023) *Nature*
- **FAFB dataset:** Zheng et al. (2018) *Cell*

## License

[Add your chosen license -- MIT recommended]

## Authors

[Add your name and affiliation]
