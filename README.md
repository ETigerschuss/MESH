# FlyWire Overlap Analysis Toolkit

Interactive 3D visualization toolkit for analyzing neuron overlaps in the Drosophila visual system using the FlyWire connectomics dataset.

## Overview

This toolkit analyzes pairwise spatial overlaps between 26 neurons in the Drosophila optic lobe:
- **16 VS neurons** (Vertical System: VS1-VS8, L/R)
- **6 HS neurons** (Horizontal System: HSN, HSE, HSS, L/R)  
- **4 Motor neurons** (MOT, MOS, L/R)

**Key Features:**
- Analyzes overlaps at multiple distance thresholds (1, 2, 5, 10 μm)
- Generates interactive 3D mesh viewer with EM snapshots
- Identifies contact patches and synaptic connections
- Exports neuron meshes and comprehensive statistics

## Output

The analysis generates:
- **Interactive 3D HTML viewer** with EM snapshot navigation
- Contact patch data (200+ patches across 26 neurons)
- Synapse data (67 MOT/MOS synapses)
- Neuron meshes (26 OBJ files)
- EM snapshots with segmentation overlay (15,000+ images)
- Overlap matrices and statistical plots

## Requirements

### Python Dependencies
```
numpy
pandas
plotly
Pillow
navis
fafbseg
cloud-volume
trimesh
opencv-python
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Data Access
- FlyWire CAVE token (for neuron download)
- Internet connection for first run (downloads ~1.4 GB meshes)

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/flywire-overlap-analysis.git
cd flywire-overlap-analysis

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_toolkit.py
```

## Usage

### Main Analysis (Recommended)

Run the complete analysis pipeline:

```bash
python overlap_analysis.py
```

**What it does:**
1. Loads 26 neurons from FlyWire dataset (FAFB v141)
2. Analyzes pairwise overlaps at 4 distance thresholds
3. Exports meshes to `comprehensive_overlap_results/neuron_meshes/`
4. Automatically generates interactive EM viewer

**Runtime:** 30-60 minutes (depending on network speed)

**Output:** `comprehensive_overlap_results/` directory with all results

### EM Viewer (Standalone)

If you already have results, regenerate the viewer:

```bash
python em_viewer.py
```

Generates: `comprehensive_overlap_results/em_viewer.html`

### Generate EM Stacks (Optional)

Batch-generate all EM snapshots with Z-stacks:

```bash
python generate_em_stacks.py
```

This is optional - the viewer generates snapshots on-the-fly if missing.

## Interactive Viewer Features

The generated `em_viewer.html` includes:

### Left Panel: Neuron Controls
- Toggle visibility for meshes, contacts, and synapses
- Color-coded by neuron type (VS: teal, HS: orange, MN: purple)

### Center Panel: 3D Mesh Viewer
- Interactive rotation, zoom, pan
- Click contact (red circles) or synapse (yellow) points
- Selected point highlights in 3D view

### Right Panel: EM Snapshot Viewer
- Segmentation overlay on EM images
- Z-stack navigation (±20 slices, ±800nm depth)
- Previous/Next navigation filtered to visible neurons
- Resizable panels via draggable borders

## Project Structure

```
flywire-overlap-analysis/
├── overlap_analysis.py          # Main analysis script
├── em_viewer.py                  # Interactive viewer generator
├── generate_em_stacks.py         # Optional EM batch generator
├── test_toolkit.py               # Installation verification
├── test_em_viewer.py             # Quick viewer test
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git exclusions

Generated outputs (not in repository):
comprehensive_overlap_results/
├── em_viewer.html                # Interactive 3D viewer (~100 MB)
├── all_results_combined.csv      # Contact patch data
├── synapses.csv                  # Synapse data
├── neuron_meshes/                # 26 OBJ files (~1.4 GB)
├── em_snaps/                     # 15,000+ PNG files (~2 GB)
└── [additional plots and matrices]
```

## Dataset Information

- **FlyWire dataset:** FAFB v141 (flywire_783)
- **EM resolution:** 8×8×40nm (MIP 1)
- **Segmentation resolution:** 16×16×40nm (MIP 0)
- **Z-stack depth:** ±20 slices (40nm/slice, ±800nm total)

## Testing

### Verify Installation
```bash
python test_toolkit.py
```

Checks:
- All core files present
- Python syntax valid
- Dependencies installed
- Results directory structure

### Test EM Viewer
```bash
python test_em_viewer.py
```

Requires existing results. Tests viewer generation in ~30 seconds.

## Troubleshooting

### "No module named X"
Install dependencies: `pip install -r requirements.txt`

### "FlyWire token required"
Set your CAVE token in the script (line 27) or as environment variable

### Viewer generation fails
- Check that results exist in `comprehensive_overlap_results/`
- Verify CSV files and meshes are present
- Run `python test_em_viewer.py` for detailed diagnostics

### Files too large for GitHub
- Do NOT commit `comprehensive_overlap_results/` directory
- `.gitignore` is configured to exclude large files
- Only scripts and documentation should be tracked

## Citation

If you use this toolkit in your research, please cite:

- **FlyWire dataset:** Dorkenwald et al. (2023) Nature
- **FAFB dataset:** Zheng et al. (2018) Cell

## License

[Add your chosen license here - MIT recommended]

## Authors

[Add your name and affiliation]

## Acknowledgments

- FlyWire community for the connectomics dataset
- [Your lab/institution]

## Contact

[Your contact information or lab website]

---

**Last updated:** November 2025
