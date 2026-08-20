# MESH — Neuron EM Overlap Viewer Pipeline

Interactive pipeline for investigating potential gap junctions and synaptic connectivity between motor neurons (MOT/MOS) and visual interneurons (VS/HS/BIPS/H2) in the *Drosophila* brain, using the FlyWire EM connectomics dataset (FAFB v141, segmentation version mat783).

## Overview

Analyses pairwise spatial overlaps between **22 neurons** in the *Drosophila* optic lobe and provides an interactive 3D + EM viewer for manual proofreading of candidate gap-junction sites, combined with an integrated biophysical circuit simulator.

**Neurons (configured in `neurons.json`):**
- **8 VS neurons** — Vertical System LPTCs: VS1-VS4, left + right
- **6 HS neurons** — Horizontal System LPTCs: HSN, HSE, HSS, left + right
- **4 Motor neurons** — MOT_L/R (neck rotation), MOS_L/R (smooth eye movements)

**Key features:**
- Pairwise overlap analysis at configurable distance thresholds
- **Spatial clustering** of overlap faces — disconnected contact regions (>10 um apart) become separate viewer entries
- Overlap area is estimated from the summed surface area of mesh faces whose centroids fall within the distance threshold
- Interactive 3D HTML viewer (Plotly) with EM snapshot panel + Z-stack navigation
- Per-slice EM images centered on the actual overlap location (not averaged centroids)
- Overlap deletion, area recalculation, auto-advance navigation
- EM contrast control, snapshot download, and editable Tier 1 / Tier 2 simulation parameters
- 2D skeleton projection plots

## Pipeline Scripts

Run all scripts in order with **`python run_all.py`**, or run individually:

| # | Script | Description |
|---|--------|-------------|
| 1 | **`overlap_analysis.py`** | Core analysis: downloads 22 neuron meshes from FlyWire, computes all pairwise overlaps at 0.1 µm threshold, identifies contact patches (Top 10 per pair) and synaptic connections. Saves results to `comprehensive_overlap_results_YYYY-MM-DD/`. Produces: `all_results_combined.csv`, `synapses.csv`, `geometric_data/contact_faces.csv`, `geometric_data/contact_vertices.csv`, overlap matrices, and 3D Plotly figures. Runtime: ~2 h first run (mesh download), ~30 min cached. |
| 2 | **`generate_skeleton_plots.py`** | Generates 2D PNG projection plots of neuron skeletons with overlap faces highlighted. One plot per neuron pair with overlap, plus 9 summary scenarios (ALL, MOT-only, MOS-only, hemispheres). |
| 3 | **`generate_em_stacks.py`** | Downloads EM snapshots with coloured segmentation overlays from CloudVolume. **Spatially clusters** overlap faces per pair (10 µm threshold via `scipy.cluster.hierarchy`) so disconnected contact regions get separate indices and per-slice centroids. Also downloads contact-patch Z-stacks (±20 slices) and synapse Z-stacks (±20 slices). Writes `overlap_em_meta.json` (consumed by the viewer). Has timeout-and-retry logic (60 s / 3 attempts / exponential back-off). |
| 4 | **`skeleton_em_viewer.py`** | Generates the final self-contained HTML viewer. 3D Plotly scene with neuron meshes, overlap faces (Mesh3d), contacts, synapses + right-side EM panel with Z-slider, delete buttons, area matrix, auto-advance. Integrates the full Tier 1 + Tier 2 biophysical circuit model. Reads `overlap_em_meta.json` and images from `em_snaps/`. |

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
| BIPS | Teal | `#00796B` |
| H2 | Blue | `#1565C0` |

Colors are defined per neuron in `neurons.json` and are shared between the 3D viewer mesh colors and the Tier 1 circuit simulation trace colors.

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

### Quick setup from GitHub

```bash
git clone https://github.com/ETigerschuss/MESH.git
cd MESH

# Recommended: create an isolated virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate.bat       # Windows (Command Prompt)
# or:
venv\Scripts\Activate.ps1       # Windows (PowerShell)

pip install -r requirements.txt
```

### Obtain a FlyWire CAVE token

```bash
# Visit https://global.daf-apis.com/info/ to get your token, then:

# Option A — environment variable (must be set each session):
export FLYWIRE_TOKEN="your_token_here"   # Linux/macOS
$Env:FLYWIRE_TOKEN = "your_token_here"  # Windows PowerShell

# Option B — persistent secret file (loaded automatically by run_all.py):
mkdir -p ~/.cloudvolume/secrets
echo '{"token": "your_token_here"}' > ~/.cloudvolume/secrets/cave-secret.json
```

### Run the pipeline

```bash
python run_all.py          # Full pipeline (Steps 1-4 in sequence)
```

This will:
1. Download neuron meshes from FlyWire (~1.4 GB, first run only)
2. Compute pairwise overlaps and save results
3. Download EM snapshot stacks (~3,400 images)
4. Generate the interactive HTML viewer

Then open `comprehensive_overlap_results_YYYY-MM-DD/skeleton_em_viewer.html` in Chrome or Firefox.

> **First run time:** ~2-3 hours (mesh + EM download). Subsequent runs: ~5-15 minutes (all cached).

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

### Overlap Area Calculation

Overlap area is computed in Step 1 from the triangle mesh geometry, not from image pixels.

1. For each neuron pair, the script builds a `cKDTree` over one mesh and queries the other mesh for faces whose centroids lie within the chosen distance threshold.
2. Every overlapping triangle contributes its geometric face area.
3. The total overlap area for a pair is the sum of all accepted triangle-face areas, reported in µm².
4. Top patches are then ranked by connected area, and those per-face records are written to `geometric_data/contact_faces.csv`.

This means the matrix values are surface-area estimates on the FlyWire meshes. They are sensitive to mesh quality, proofreading state, and the chosen overlap threshold.

### Tier-1 Circuit Simulation (Biophysical Model)
The viewer includes an integrated **neural circuit model** of the first-order optic flow pathway (VS, HS, BIPS, MOT, MOS neurons). Access it via the **Circuit Model (Tier 1)** tab in the matrix popup.

#### Model Architecture

| Cell type | Count | Model type | Description |
|-----------|-------|------------|-------------|
| VS1-VS4 (L+R) | 8 | Graded LPTC | Vertical-system wide-field motion detectors |
| HSN, HSE, HSS (L+R) | 6 | Graded LPTC | Horizontal-system wide-field motion detectors |
| MOT_L/R | 2 | HH spiking MN | Neck/torque motor neuron, ~120 Hz rest rate |
| MOS_L/R | 2 | HH spiking MN | Smooth-pursuit motor neuron, ~100 Hz rest rate |
| BIPS_L/R | 2 | HH spiking MN | GABAergic interneuron projecting to HS |
| H2_L/R | 2 | (passive) | H2 tangential cell — currently no active model |

**Connectivity** (from FlyWire mat783 synaptic cleft segmentation, cleft area ≥ 50 voxels):

```
LPTC chain GJs:   VS1↔VS2↔VS3↔VS4  (each side, axo-axonal)
                  HSN↔HSE↔HSS       (each side, axo-axonal)
LPTC→MN GJs:      VS1-4 → MOS (bidirectional gap junctions, axon↔dendrite)
                  HSN/E/S → MOS (bidirectional gap junctions)
                  HSN/E/S → MOT (bidirectional gap junctions; VS does NOT connect to MOT)
Chemical synapses: See RAW_COUNTS matrix in viewer Connectivity tab
                  HS→MOT/MOS (excitatory, ACh), HS→BIPS (excitatory)
                  BIPS→HSN/E/S (inhibitory, GABA, Erev = -80 mV)
                  VS2/3→MOS (excitatory, chemical)
                  MOT↔MOS (excitatory, MN↔MN feedback, 3-6 contacts)
```

#### Cell Models

**LPTC (graded, non-spiking):**

The membrane voltage follows:

$$C_m \frac{dV}{dt} = -I_T - I_K - I_L + I_\text{input}$$

where:
- $I_T = g_{VT} \cdot m_{\infty,Ca}(V)^3 \cdot h_{Ca}(t) \cdot (V - V_{Ca})$ — T-type Ca²⁺ (optional)
- $I_K = g_K \cdot n(t)^4 \cdot (V - V_K)$ — delayed-rectifier K⁺
- $I_L = (g_L + 1/R_{in}) \cdot (V - V_L)$ — leak + standing conductance

Resting potential is set by $V_L$ (leak reversal). Input resistance $R_{in}$ controls voltage gain.

**MN — Motor Neuron (Hodgkin-Huxley spiking):**

$$C_m \frac{dV}{dt} = -I_T - I_{Na} - I_K - I_{NaP} - I_L + I_\text{input}$$

where:
- $I_{Na} = g_{Na} \cdot m(t)^3 \cdot h(t) \cdot (V - V_{Na})$ — transient Na⁺ (spike upstroke)
- $I_K = g_K \cdot n(t)^4 \cdot (V - V_K)$ — delayed rectifier K⁺ (repolarisation / AHP)
- $I_{NaP} = g_{NaP} \cdot m_{\infty,NaP}(V) \cdot (V - V_{Na})$ — **persistent Na⁺** (primary tonic driver)
- $I_L = (g_L + 1/R_{in}) \cdot (V - V_L)$ — leak

Gate kinetics follow the original Hodgkin-Huxley (1952) formulation. $I_{NaP}$ does not inactivate, providing a sustained sub-threshold depolarising current that sustains tonic firing at 100-120 Hz without external drive.

**Key tuning parameters for MN firing rate:**

| Parameter | Role | MOT calibrated | MOS calibrated |
|-----------|------|---------------|---------------|
| `gNaP` | Primary rate control — higher gNaP → higher tonic rate | 1.748 nS | 1.510 nS |
| `VL` | Leak reversal — more depolarised → higher rate | -58.81 mV | -62.76 mV |
| `Ibias` | Secondary additive bias current | 2.84 nA | 2.10 nA |
| `gK` | AHP depth — higher gK → longer ISI → lower rate | 52.68 nS | 45.60 nS |

> **Note:** To change which cell fires faster, adjust `gNaP` differentially (primary lever). `Ibias` shifts the rate but cannot overcome a large gNaP difference. The auto-calibrate tool performs a bisection search over `gNaP` to hit a user-defined target firing rate.

**Gap junctions (bidirectional, LP-filtered):**

$$\frac{dV_f}{dt} = \frac{(V_B - V_A) - V_f}{\tau}$$
$$I_{A \leftarrow B} = G \cdot V_f, \quad I_{B \leftarrow A} = -G \cdot V_f$$

Time constant $\tau = C/G$. The filter prevents individual MN spikes from creating large artefactual transients in connected LPTCs. Default $\tau \approx 8$ ms for LPTC↔MN junctions.

**Graded chemical synapses (LPTC pre-synaptic):**

$$I_{syn} = -g_{max} \cdot \text{clamp}\!\left(\frac{V_{pre} - V_{thresh}}{V_{scale}}, 0, 1\right) \cdot (V_{post} - E_{rev})$$

Default thresholds: $V_{thresh} = -40$ mV, $V_{scale} = 20$ mV. $E_{rev} = 0$ mV (excitatory) or $-80$ mV (GABA, inhibitory).

**Alpha-function synapses (MN pre-synaptic, spike-triggered):**

On spike detection (upward $V_{pre}$ crossing of 0 mV):
$$\Delta \dot{g} += g_{max}/\tau_{syn}$$

Evolves as: $\dot{g}(t) = (g_{max}/\tau) \cdot e^{-t/\tau}$, giving a fast conductance transient with default $\tau_{syn} = 5$ ms.

#### Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dt` | 0.01 ms | Integration step (forward Euler) |
| `Cm` | 1.0 nF | Membrane capacitance |
| `V_Na` | +50 mV | Na⁺ reversal potential |
| `V_K` | -77 mV | K⁺ reversal potential |
| `V_Ca` | +120 mV | T-type Ca²⁺ reversal potential |
| Pre-roll | 500 ms | Hidden settling time before t=0 |
| `G_lptc` | 0.05 nS | Within-chain LPTC gap junction conductance |
| `G_VS↔MOS` | 0.1 nS | VS→MOS gap junction conductance |
| `G_HS↔MOS` | 0.1 nS | HS→MOS gap junction conductance |
| `G_HS↔MOT` | 0.1 nS | HS→MOT gap junction conductance |

#### Key Features
1. **Neuron Deletion (circuit lesion tool):**
   - Click a neuron in the wiring diagram to deactivate it
   - Deleted neurons are excluded from all synaptic and GJ transmission
   - Use to test redundancy: "What happens if HSE_L is removed?"

2. **Auto-Calibration:**
   - Click **Auto-Calibrate** to fit gNaP (then Ibias) to user-defined rest Hz targets
   - Calibration report shows achieved rate vs target for MOT and MOS

3. **Parameter Editing:**
   - The Tier 1 panel exposes editable intrinsic parameters for MOS, MOT, VS, and HS cells
   - Gap-junction and chemical-synapse gains can be changed without rerunning the Python pipeline
   - Tier 2 exposes its own axial-conductance and compartment-capacitance controls

4. **Visual Output (6 plots):**
   - LPTC voltages: left and right eye VS/HS traces
   - MN voltages: MOT/MOS with stimulus window marked
   - Pseudopupil time series: spike rate vs time, left vs right pseudopupil
   - Polar plot: response direction tuning

Methodologically, the pseudopupil polar plots are built from the simulated MOS and MOT membrane voltages in three steps. First, the viewer converts each motor-neuron voltage trace into a robust firing-rate time series using a sliding spike-rate estimator. Second, it computes a pre-stimulus baseline over the user-defined baseline window and converts the response into movement drive using baseline-subtracted MOS and MOT rates, a small dead-zone, and saturating pull/release nonlinearities; MOS contributes the horizontal component and MOT the vertical component, with an extra cooperative term when both channels rise or fall together. Third, those instantaneous x/y components are accumulated into a pseudopupil trajectory, and the polar arrows report the net direction from the change between the early and late portions of that trajectory, separately for the MOS component, the MOT component, and the combined net movement.

#### Usage
1. Open the **Matrix** button → **Circuit Model (Tier 1)** tab
2. Use default parameters, or click **Preset: MOT~120 / MOS~100** for calibrated values
3. Adjust intrinsic, synaptic, or coupling parameters as needed
4. Select stimulus targets (e.g., "VS Left (1-4)") and set amplitude/timing
5. Click **▶ Run**
6. (Optional) Click neurons in the wiring diagram to test lesions, then re-run

### Center Panel — 3D Scene
- Neuron meshes + overlap face triangles (Mesh3d)
- Contacts (red circles), synapses (yellow markers)
- Click any overlap face -> jumps to EM panel for that cluster
- Diamond-shaped 3D position indicator

### Right Panel — EM Snapshots
- Segmentation overlay (source + target neuron colours)
- Z-slider with per-cluster valid slices
- **Contrast slider** — interactively changes EM image contrast in the browser for easier membrane tracing
- **Delete Slice** — Remove false positives by deleting individual Z-slices
   - Area is **automatically recalculated** from the remaining valid slices in that spatial cluster
   - The current viewer uses proportional rescaling: `remaining area = original cluster area × (remaining slices / original slices)`
   - Pair totals are updated by summing across all spatial sub-clusters for that neuron pair
  - Supported for both **overlap and contact** types
  - Auto-advances to next valid slice after deletion
- **Delete All** — Eliminate entire overlap pair (all Z-slices at once)
  - Marks pair as eliminated in area matrix
  - Useful for confirmed false positives
  - Can be undone by regenerating HTML (deletions are tracked in overlaps only, not persisted)
- Previous / Next navigation across all visible items
- Download EM — Export current snapshot with coordinates in filename and metadata panel
   - The PNG export includes the current slice metadata and touching-cell labels burned into the image footer
   - The export path still needs one cleanup pass so the downloadable data/annotation file is unambiguously expressed in FlyWire global coordinates

#### Deletion Workflow for Proofreading
1. Open EM panel for an overlap pair
2. Scroll through Z-slices to identify false positives (scanning artifacts, retracted branches, etc.)
3. Click **Delete Slice** for slices that don't represent real synapses
4. When a pair is confirmed as false positive, click **Delete All** to remove it
5. Area matrix updates in **real-time** — total overlap area is recalculated
6. Download audit JSON (available in viewer) to save deletion history

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
- **Chemical synapse source:** FlyWire / CAVE synapse tables accessed through `fafbseg.flywire.synapses.get_synapses` on mat783-aligned data
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

## References And Provenance

### Connectomics Dataset And Infrastructure

- **FlyWire community reconstruction / data provenance:** Dorkenwald S. et al. (2023). *FlyWire: online community for whole-brain connectomics*. Nature.
- **FAFB EM volume:** Zheng Z. et al. (2018). *A Complete Electron Microscopy Volume of the Brain of Adult Drosophila melanogaster*. Cell.

### Chemical Synapses In This Repository

- The `synapses.csv` table is pulled from FlyWire/CAVE through `fafbseg.flywire.synapses.get_synapses`, not manually curated inside this repository.
- In practice, cite the FlyWire and FAFB papers above for the dataset provenance of those chemical synapse coordinates and counts.
- The current Tier 1 `RAW_COUNTS` matrix is a hard-coded snapshot derived from the FlyWire mat783 connectivity used during viewer generation; if that table is updated in the future, the README should be updated with the exact release identifier.

### Tier 1 Biophysical Model Provenance

- **Hodgkin-Huxley gating equations:** Hodgkin A. L. and Huxley A. F. (1952). *A quantitative description of membrane current and its application to conduction and excitation in nerve*. The Journal of Physiology.
- **Important caveat:** the Tier 1 model is not a direct copy of one published parameter table. The equations are literature-based, but many numeric values in this repository were tuned manually and with the built-in auto-calibration tool so MOT and MOS reproduce the target resting firing rates used in this project.
- The graded LPTC abstraction, electrical coupling, and direction-selective fly motion-pathway framing are motivated by the classic fly tangential-cell literature; where an exact parameter provenance is still unresolved, this README now states that explicitly rather than implying a publication source that is not yet pinned down.

### What Still Needs Tightening

- The downloadable annotation/data export should be normalized to explicit FlyWire global coordinates.
- The Tier 1 section would benefit from a future pass that maps each fitted intrinsic parameter to either a literature source or an explicit "project-fit" label.

## Known Limitations

1. **Tier 1 model is single-compartment:** MOT/MOS are modelled with one electrical compartment. Dendritic filtering and axonal spike initiation are not captured. Multi-compartment modelling is available in Tier 2.
2. **Gap junctions are inferred from anatomy:** GJ sites are placed at the largest overlap region for each known pair. Functional confirmation (dye coupling, physiology) has not been performed for all pairs.
3. **BIPS and H2 lack specific parameters:** BIPS uses MOT HH defaults; H2 is currently passive (no active conductances). Both should be updated as electrophysiological data become available.
4. **EM proofreading deletions are session-local:** Deletions made in the viewer are not saved back to the JSON pipeline output. Export the `viewer_annotations.json` from the viewer and run `_patch_em_stacks.py` to apply them permanently.
5. **Forward Euler integration:** dt = 0.01 ms is adequate for this model but numerical drift may accumulate in very long runs (> 10 s). Increase dt only if simulation time allows — action potentials require dt ≤ 0.025 ms.

## Extending the Pipeline

To add new neurons:
1. Add an entry to `neurons.json` with FlyWire ID, `color_hex`, `group`, and `hemibrain_id`
2. Add the neuron name to `viewer_neurons` and `synapse_groups` (if synapses should be loaded)
3. Add pairing rules in `pairing_rules` if needed
4. Re-run `python run_all.py` — the pipeline will skip cached meshes and only download the new neuron

To change the overlap distance threshold:
- Edit `THRESHOLDS_MICRONS` in `overlap_analysis.py`
