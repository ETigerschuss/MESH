"""
Skeleton EM Viewer — Interactive Circuit Reconstruction & Circuit Model
=========================================================================

A self-contained, interactive 3D + 2D viewer for electron-microscopy (EM) circuit
reconstruction with integrated neural circuit simulation model.  One Python run
produces a single, standalone HTML file that embeds all data, Plotly, and
JavaScript — no server required.

KEY FEATURES
------------
1. **3D Visualization:**
   - Neuron meshes (colored by neuron group)
   - Overlap faces (contact regions between neurons)
   - Contact points (synaptic sites)
   - Interactive rotation, pan, zoom

2. **2D EM Snapshots:**
   - Segmentation overlays (colored by neurons)
     - Browser-side contrast adjustment for easier membrane tracing
   - Z-stack navigation (±20 slices)
   - **DELETION:** Remove false positives by deleting individual Z-slices
         → Area is automatically recalculated from remaining slices by proportional rescaling of each cluster's original area
     - **EXPORT:** Download snapshots with coordinates in filename & metadata
         → Current export metadata still needs one cleanup pass to guarantee FlyWire-global coordinate wording everywhere

3. **Overlap Area Proofreading:**
   - Interactive overlap matrix (clickable cells)
   - Mark false positives for elimination
   - Real-time area recalculation
   - Audit trail of deletions

4. **Tier-1 Circuit Simulation (Experimental):**
   - Hodgkin-Huxley-like spiking model for MOT/MOS motor neurons
   - Compartmental LPTCs (VS/HS wide-field neurons)
   - Gap junctions + chemical synapses
    - Editable intrinsic, synaptic, and coupling parameters from the browser UI
   - **NEURON DELETION:** Test circuit robustness by removing individual neurons
     → Deleted neurons automatically excluded from synaptic transmission & coupling
   - Bilateral pseudopupil output (eye motion tuning)
   - Auto-calibration for resting firing rates

SCIENTIFIC PROVENANCE
---------------------
Dataset provenance follows the FlyWire reconstruction on FAFB v141. The chemical
synapse table is read from FlyWire/CAVE via ``fafbseg.flywire.synapses``. The
Tier 1 conductance equations are based on Hodgkin-Huxley-style membrane dynamics,
but the numeric defaults in this repository are a hybrid of literature-inspired
values and project-specific fitting / auto-calibration, not a one-to-one copy of
a single published parameter table.

STANDALONE USAGE
----------------
Run from the project root::

        python skeleton_em_viewer.py

The output file is written to::

        <latest comprehensive_overlap_results_*>/skeleton_em_viewer.html

Override the results directory with an environment variable::

        set MESH_RESULTS_DIR=C:\\path\\to\\my_results
        python skeleton_em_viewer.py

REQUIRED INPUTS (inside RESULTS_DIR)
--------------------------------------
* ``all_results_combined.csv``       — contact-patch table (Has_Contact rows)
* ``synapses.csv``                   — chemical synapse coordinates
* ``geometric_data/contact_vertices.csv``  — overlap-region midpoints
* ``geometric_data/contact_faces.csv``     — overlap triangle meshes (optional)
* ``overlap_em_meta.json``           — EM snapshot metadata per pair (optional)
* ``contact_cluster_map.json``       — patch→cluster mapping (optional)
* ``neuron_meshes/<id>.obj``         — neuron surface meshes (optional)
* ``em_snaps/``                      — PNG snapshots at each contact/overlap

CONFIGURATION (active neuron config)
------------------------------------
All neuron identities, colors, synapse groups, and pipeline parameters are
read from the active config profile (``neurons.json`` by default, or
``MESH_NEURON_CONFIG`` when set). Keys used:

* ``neurons``          — dict of name → {id, color_hex, group, ...}
* ``viewer_neurons``   — ordered list of neuron names to show in the sidebar
* ``synapse_groups``   — set of group names whose synapses are loaded
* ``top_patches``      — how many Top-N patch columns to read (default 10)
* ``face_decimation_nm``  — vertex merging distance for mesh decimation (default 80)

VIEWER FEATURES
---------------
* 3D Plotly viewer — meshes, contacts (red), synapses (yellow/blue), overlaps
* EM panel — click any 3D point to open its EM snapshot; navigate Z-stack
* Deletion workflow — flag spurious contacts/overlap slices; export JSON audit log
* Gap-junction annotation — mark putative GJ sites and export for post-processing
* Heatmap matrix — 4-panel overlap-area matrix (full / L-R mean / group / bidir)
* Tier 1 circuit model — single-compartment HH simulation of LPTC–MN circuit
* Tier 2 circuit model — multi-compartment simulation with axial coupling

ARCHITECTURE
------------
Python side (this file):
    load_*()        — data loaders, each reading one CSV / JSON source
    build_figure()  — constructs the Plotly figure with all trace types
    generate_html() — template substitution, serialises all data to JSON
    main()          — orchestrates the above and writes the output file

JavaScript (embedded in HTML_TEMPLATE):
    State + 3D viewer interaction (~700 lines)
    EM viewer + Z-stack navigation + deletion (~400 lines)
    Heatmap modal + connectivity matrix (~400 lines)
    Tier 1 biophysical circuit model (~600 lines)
    Tier 2 multi-compartment model (~600 lines)
"""

import os
import json
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh

from mesh_config import load_config

# ── Config from active profile ────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_cfg, CONFIG_PATH = load_config()

NEURON_CFG = _cfg['neurons']
NEURON_IDS = {info['id']: name for name, info in NEURON_CFG.items()}
NEURON_COLORS = {name: info['color_hex'] for name, info in NEURON_CFG.items()}
VIEWER_NEURONS = _cfg.get('viewer_neurons', sorted(NEURON_CFG.keys()))

_synapse_groups = set(_cfg.get('synapse_groups', []))
SYNAPSE_NEURONS = [n for n, info in NEURON_CFG.items() if info['group'] in _synapse_groups]

# Inhibitory pairs used for synapse color/label classification.
INH_PAIRS = frozenset([
    ('VS1_L', 'VS2_L'),
    ('VS1_R', 'VS2_R'),
    ('VS1_R', 'VS3_R'),
    ('VS2_L', 'VS3_L'),
    ('VS3_L', 'VS4_L'),
    ('BIPS_L', 'HSN_L'),
    ('BIPS_L', 'HSE_L'),
    ('BIPS_R', 'HSN_R'),
    ('BIPS_R', 'HSE_R'),
    ('BIPS_R', 'HSS_L'),
    ('BIPS_R', 'BIPS_L'),
])


def _default_results_dir():
    """Find the most recent results directory created by the analysis pipeline.

    Searches SCRIPT_DIR for directories matching
    ``comprehensive_overlap_results_*`` and returns the lexicographically latest
    (i.e. most recently timestamped) one.  Falls back to a bare
    ``comprehensive_overlap_results`` name if none exist.

    Override at runtime with environment variable ``MESH_RESULTS_DIR``.
    """
    candidates = [d for d in os.listdir(SCRIPT_DIR)
                  if os.path.isdir(os.path.join(SCRIPT_DIR, d))
                  and d.startswith('comprehensive_overlap_results_')]
    if candidates:
        return os.path.join(SCRIPT_DIR, sorted(candidates)[-1])
    return os.path.join(SCRIPT_DIR, 'comprehensive_overlap_results')


RESULTS_DIR = os.environ.get('MESH_RESULTS_DIR', _default_results_dir())


# ── HTML Template ─────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EM Overlay Viewer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0; padding: 0;
            font-family: Arial, sans-serif;
            background: #1a1a1a; color: white;
            overflow: hidden;
        }
        .container { display: flex; height: 100vh; width: 100vw; }

        /* ── Sidebar ──────────────────────────────── */
        .sidebar {
            width: 160px; min-width: 100px; max-width: 300px;
            background: #2a2a2a; padding: 10px;
            overflow-y: auto; border-right: 1px solid #444;
        }
        .sidebar h3 {
            margin: 0 0 10px 0; font-size: 14px; color: #FFD400;
            border-bottom: 1px solid #444; padding-bottom: 5px;
        }
        .neuron-group {
            margin-bottom: 8px; padding: 6px;
            background: #1f1f1f; border-radius: 4px;
        }
        .neuron-name {
            font-weight: bold; font-size: 12px; display: block; margin-bottom: 4px;
        }
        .neuron-controls {
            display: flex; flex-direction: column; gap: 2px; font-size: 11px;
        }
        .neuron-controls label {
            display: flex; align-items: center; gap: 4px;
            cursor: pointer; padding: 2px 0;
        }

        /* ── Resizers ─────────────────────────────── */
        .resizer {
            width: 5px; background: #444; cursor: col-resize;
        }
        .resizer:hover { background: #666; }

        /* ── Center column ────────────────────────── */
        .center-col {
            flex: 1; min-width: 300px;
            display: flex; flex-direction: column;
            background: #0a0a0a;
        }
        .controls {
            background: #222; padding: 8px 15px;
            border-bottom: 1px solid #444;
        }
        .plot-container { flex: 1; position: relative; min-height: 0; }

        /* ── Overlap table below the 3D viewer ──── */
        .table-container {
            display: none; /* legacy - unused */
        }

        /* ── Modal overlay for heatmap ─────────────── */
        .modal-overlay {
            display: none; position: fixed; top: 0; left: 0;
            width: 100vw; height: 100vh;
            background: rgba(0,0,0,0.7); z-index: 1000;
            align-items: center; justify-content: center;
        }
        .modal-overlay.active { display: flex; }
        .modal-content {
            background: #ffffff; border: 1px solid #cfcfcf;
            border-radius: 8px; max-width: 90vw; max-height: 85vh;
            overflow: auto; padding: 16px; position: relative;
            color: #222;
        }
        .modal-content h3 { color: #1f1f1f; margin: 0 0 12px 0; }
        .modal-close {
            position: absolute; top: 8px; right: 12px;
            background: none; border: none; color: #666;
            font-size: 22px; cursor: pointer;
        }
        .modal-close:hover { color: #111; }

        /* ── Right EM panel ────────────────────────── */
        .em-panel {
            width: 600px; min-width: 400px; max-width: 1000px;
            background: #2a2a2a; display: flex; flex-direction: column;
            border-left: 1px solid #444;
        }
        .em-header {
            background: #1f1f1f; padding: 10px 15px;
            border-bottom: 1px solid #444;
        }
        #emTitle {
            font-weight: bold; color: #FFD400; font-size: 13px;
            display: block; margin-bottom: 4px;
        }
        #emLocation { color: #888; font-size: 11px; }
        .em-display {
            flex: 1; position: relative; display: flex;
            align-items: center; justify-content: center;
            background: #0a0a0a; padding: 10px;
            min-height: 0; overflow: hidden;
        }
        #emImage {
            width: 100%; height: 100%;
            object-fit: contain; display: block;
        }
        #emPlaceholder { color: #555; font-size: 13px; text-align: center; }
        .em-controls { background: #1f1f1f; border-top: 1px solid #444; }
        .control-row {
            padding: 8px 15px; display: flex; align-items: center;
            gap: 10px; border-bottom: 1px solid #333;
        }
        .control-row:last-child { border-bottom: none; }
        .control-row button {
            padding: 6px 12px; cursor: pointer;
            background: #444; color: white; border: 1px solid #666;
            border-radius: 3px; font-size: 11px; white-space: nowrap;
        }
        .control-row button:hover { background: #555; }
        .control-row button:disabled { opacity: 0.5; cursor: not-allowed; }
        .control-row input[type="range"] { flex: 1; min-width: 0; }
        .control-label { font-size: 11px; color: #999; min-width: 50px; }
        .control-value {
            font-size: 11px; color: #FFD400;
            font-weight: bold; text-align: center;
        }
        .info-text { color: #999; font-size: 12px; }
        #btnDeleteSlice, #btnDeleteAll { background: #8B0000; border-color: #B00; }
        #btnDeleteSlice:hover, #btnDeleteAll:hover { background: #B22222; }
        #btnDeleteAll { background: #4A0000; border-color: #900; }
        #btnDeleteAll:hover { background: #6B0000; }
        .deleted-banner {
            background: #8B0000; color: #fff; text-align: center;
            padding: 6px; font-size: 12px; font-weight: bold;
            display: none;
        }
        #btnExport { background: #2E7D32; border-color: #4CAF50; }
        #btnExport:hover { background: #388E3C; }
        #btnMarkGJ { background: #1B5E20; border-color: #00E676; }
        #btnMarkGJ:hover { background: #2E7D32; }
        #btnMarkGJ.active { background: #00E676; border-color: #00E676; color: #000; }
        #btnRemoveGJ { background: #4A0000; border-color: #900; }
        #btnRemoveGJ:hover { background: #6B0000; }

        /* ── Modal tab bar ─────────────────────── */
        .modal-tabs { display: flex; gap: 0; margin-bottom: 12px; border-bottom: 2px solid #d0d0d0; }
        .modal-tab {
            padding: 6px 16px; cursor: pointer; background: #f3f3f3; color: #555;
            border: 1px solid #d0d0d0; border-bottom: none; border-radius: 6px 6px 0 0;
            font-size: 12px; font-weight: bold;
        }
        .modal-tab:hover { background: #e9e9e9; color: #222; }
        .modal-tab.active { background: #ffffff; color: #111; border-bottom: 2px solid #ffffff; margin-bottom: -2px; }
        .modal-tab-content { display: none; }
        .modal-tab-content.active { display: block; }
        .gj-table { border-collapse: collapse; font-size: 11px; width: 100%; }
        .gj-table th { background: #f0f0f0; color: #222; padding: 4px 8px; border: 1px solid #d0d0d0; text-align: left; font-size: 10px; }
        .gj-table td { padding: 4px 8px; border: 1px solid #d0d0d0; font-size: 10px; color: #222; }
        .gj-table tr:hover { background: #f8f8f8; }
        .gj-table .gj-delete { cursor: pointer; color: #B00; }
        .gj-table .gj-delete:hover { color: #F44; }

        /* ── Heatmap matrix ─────────────────────── */
        .heatmap-container {
            max-height: 350px; min-height: 120px;
            overflow: auto; background: #ffffff;
            border-top: 2px solid #d0d0d0;
            padding: 8px;
        }
        .heatmap-container table {
            border-collapse: collapse;
            font-size: 11px;
        }
        .heatmap-container th {
            background: #f0f0f0; color: #222;
            padding: 3px 6px; text-align: center;
            border: 1px solid #d0d0d0;
            position: sticky; top: 0; z-index: 2;
            font-size: 10px;
        }
        .heatmap-container th.row-header {
            position: sticky; left: 0; z-index: 3;
            background: #f0f0f0;
        }
        .heatmap-container th.corner {
            position: sticky; left: 0; top: 0; z-index: 4;
            background: #f0f0f0;
        }
        .heatmap-container td {
            padding: 2px 4px; text-align: center;
            border: 1px solid #d0d0d0; cursor: pointer;
            font-size: 10px; font-family: monospace;
            min-width: 46px;
        }
        .heatmap-container td:hover { outline: 2px solid #1976d2; }
        .heatmap-container td.diagonal { background: #f7f7f7; cursor: default; }
        .heatmap-container td.no-data { color: #9a9a9a; }
    </style>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
    <div class="container">
        <!-- Left: Neuron Controls -->
        <div class="sidebar">
            <h3>Neurons</h3>
            {CHECKBOXES_HTML}
        </div>
        <div class="resizer" id="resizer1"></div>

        <!-- Center: 3D Viewer -->
        <div class="center-col">
            <div class="controls">
                <span id="infoText" class="info-text">Select neurons, then click contacts (red), synapses (yellow), or enable overlaps</span>
            </div>
            <div class="plot-container">
                {PLOT_DIV}
            </div>
        </div>
        <div class="resizer" id="resizer2"></div>

        <!-- Right: EM Panel -->
        <div class="em-panel">
            <div class="em-header">
                <span id="emTitle">EM Snapshot</span>
                <span id="emLocation"></span>
            </div>
            <div class="deleted-banner" id="deletedBanner">&#10007; DELETED &#8212; this contact has been removed</div>
            <div class="em-display">
                <img id="emImage" style="display:none;" alt="EM Snapshot">
                <span id="emPlaceholder">Click a contact or synapse to view EM</span>
            </div>
            <div class="em-controls">
                <div class="control-row">
                    <button id="btnPrevItem">&lt; Prev</button>
                    <span id="itemInfo" class="control-value" style="flex: 1;"></span>
                    <button id="btnNextItem">Next &gt;</button>
                </div>
                <div class="control-row">
                    <button id="btnPrevZ">&lt;</button>
                    <span class="control-label">Z-Stack:</span>
                    <input type="range" id="zSlider" min="-20" max="20" value="0" step="1">
                    <span id="zValue" class="control-value" style="min-width: 80px;">0</span>
                    <button id="btnNextZ">&gt;</button>
                </div>
                <div class="control-row" style="gap:8px;flex-wrap:wrap;">
                    <span class="control-label">EM Contrast:</span>
                    <input type="range" id="emOpacitySlider" min="50" max="250" value="100" step="5" style="width:120px;">
                    <span id="emOpacityValue" class="control-value" style="min-width:45px;">100%</span>
                    <button id="btnDownloadEM" title="Download current EM image with coordinates and touching cells">&#128247; Download EM</button>
                </div>
                <div class="control-row" style="justify-content: center; gap: 12px; flex-wrap: wrap;">
                    <span id="zNote" style="color: #888; font-size: 10px;">&#177;800nm depth range</span>
                    <button id="btnDeleteSlice" title="Remove this single slice (contact or overlap Z-slice)">&#128465; Delete Slice</button>
                    <button id="btnDeleteAll" title="Remove entire overlap pair (all slices)">&#128465; Delete All</button>
                    <button id="btnMarkGJ" title="Mark current location as putative gap junction">&#9889; Putative Gap-Junc</button>
                    <button id="btnRemoveGJ" title="Remove gap junction at current location" style="display:none;">&#10006; Remove GJ</button>
                    <button id="btnMatrix" title="Show overlap area heatmap matrix" style="background:#1565C0;border-color:#42A5F5;">&#9638; Matrix</button>
                    <button id="btnExport" title="Export list of deleted contacts">&#128190; Export</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Matrix / Gap-Junction Modal -->
    <div class="modal-overlay" id="matrixModal">
        <div class="modal-content">
            <button class="modal-close" id="matrixClose">&times;</button>
            <h3 id="modalTitle">Overlap Area Matrix (&micro;m&sup2;)</h3>
            <div class="modal-tabs">
                <div class="modal-tab active" data-tab="overlaps">Overlap Areas</div>
                <div class="modal-tab" data-tab="gapjunctions">Gap Junctions</div>
                <div class="modal-tab" data-tab="connectivity">Connectivity</div>
                <div class="modal-tab" data-tab="circuit">Circuit Model (Tier 1)</div>
                <div class="modal-tab" data-tab="mc">Multi-Compartment (Tier 2)</div>
            </div>
            <div class="modal-tab-content active" id="tabOverlaps">
                <div style="display:flex;flex-direction:column;gap:10px;">
                    <div>
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                            <b style="color:#222;font-size:11px;">Full Matrix (all 22 neurons)</b>
                            <input type="range" id="heatSliderFull" min="0.1" max="100" value="100" step="0.1"
                                   style="width:140px;" title="Adjust max heat intensity">
                            <span id="heatSliderFullVal" style="color:#555;font-size:10px;min-width:60px;">max: 100%</span>
                        </div>
                        <div class="heatmap-container" id="heatmapContainer"></div>
                    </div>
                    <div>
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                            <b style="color:#222;font-size:11px;">L/R Pair Mean</b>
                            <input type="range" id="heatSliderPair" min="0.1" max="100" value="100" step="0.1"
                                   style="width:140px;" title="Adjust max heat intensity">
                            <span id="heatSliderPairVal" style="color:#555;font-size:10px;min-width:60px;">max: 100%</span>
                        </div>
                        <div class="heatmap-container" id="heatmapPairContainer"></div>
                    </div>
                    <div>
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                            <b style="color:#222;font-size:11px;">Group Mean (L+R collapsed)</b>
                            <input type="range" id="heatSliderGroup" min="0.1" max="100" value="100" step="0.1"
                                   style="width:140px;" title="Adjust max heat intensity">
                            <span id="heatSliderGroupVal" style="color:#555;font-size:10px;min-width:60px;">max: 100%</span>
                        </div>
                        <div class="heatmap-container" id="heatmapGroupContainer"></div>
                    </div>
                    <div>
                        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                            <b style="color:#222;font-size:11px;">MOS / MOT Bidirectional Mean</b>
                            <input type="range" id="heatSliderBidir" min="0.1" max="100" value="100" step="0.1"
                                   style="width:140px;" title="Adjust max heat intensity">
                            <span id="heatSliderBidirVal" style="color:#555;font-size:10px;min-width:60px;">max: 100%</span>
                        </div>
                        <div class="heatmap-container" id="heatmapBidirContainer"></div>
                    </div>
                </div>
            </div>
            <div class="modal-tab-content" id="tabGapJunctions">
                <div id="gjContainer" style="padding:8px;"></div>
            </div>
            <div class="modal-tab-content" id="tabConnectivity">
                <div class="heatmap-container" id="connectivityContainer"></div>
            </div>
            <div class="modal-tab-content" id="tabCircuit">
                <div id="circuitContainer" style="padding:8px;"></div>
            </div>
            <div class="modal-tab-content" id="tabMC">
                <div id="mcContainer" style="padding:8px;"></div>
            </div>
        </div>
    </div>

    <script>
        // ─────────────────────────────────────────────────────────────────────
        // EMBEDDED DATA
        // All data is serialised by generate_html() at build time and injected
        // as JSON literals into these constants.  Nothing is fetched at runtime
        // except the em_snaps/*.png images (relative paths).
        // ─────────────────────────────────────────────────────────────────────
        const snapshotMap   = {SNAPSHOT_JSON};
        const snapshotZMap  = {SNAPSHOT_ZMAP_JSON};
        const contactClusterMap = {CLUSTER_MAP_JSON};
        const neuronNames   = {NEURON_NAMES_JSON};
        const traceInfo     = {TRACE_INFO_JSON};
        let   contactList   = {CONTACT_LIST_JSON};
        const synapseList   = {SYNAPSE_LIST_JSON};
        const overlapList   = {OVERLAP_LIST_JSON};
        let   overlapTable  = {OVERLAP_TABLE_JSON};
        const overlapPairs  = {OVERLAP_PAIRS_JSON};
        const overlapPairFaces = {OVERLAP_PAIR_FACES_JSON};
        const neuronColors  = {NEURON_COLORS_JSON};
        const deletedItems = [];  // track deleted contacts + overlap slices

        // ── DOM refs ────────────────────────────────────────────────
        const plotDiv       = document.getElementById('plotly3d');
        const emImage       = document.getElementById('emImage');
        const emPlaceholder = document.getElementById('emPlaceholder');
        const emTitle       = document.getElementById('emTitle');
        const emLocation    = document.getElementById('emLocation');
        const infoText      = document.getElementById('infoText');
        const itemInfo      = document.getElementById('itemInfo');
        const zSlider       = document.getElementById('zSlider');
        const zValue        = document.getElementById('zValue');
        const zNote         = document.getElementById('zNote');
        const emOpacitySlider = document.getElementById('emOpacitySlider');
        const emOpacityValue  = document.getElementById('emOpacityValue');
        const btnDownloadEM   = document.getElementById('btnDownloadEM');
        const btnDeleteSlice = document.getElementById('btnDeleteSlice');
        const btnDeleteAll  = document.getElementById('btnDeleteAll');
        const btnExport     = document.getElementById('btnExport');
        const btnMatrix     = document.getElementById('btnMatrix');
        const deletedBanner = document.getElementById('deletedBanner');
        const heatmapDiv    = document.getElementById('heatmapContainer');
        const matrixModal   = document.getElementById('matrixModal');
        const btnMarkGJ     = document.getElementById('btnMarkGJ');
        const btnRemoveGJ   = document.getElementById('btnRemoveGJ');
        const gjContainer   = document.getElementById('gjContainer');
        const modalTitle    = document.getElementById('modalTitle');
        const connectivityContainer = document.getElementById('connectivityContainer');
        const circuitContainer = document.getElementById('circuitContainer');

        const sidebar       = document.querySelector('.sidebar');
        const centerCol     = document.querySelector('.center-col');
        const emPanel       = document.querySelector('.em-panel');

        // ── State ───────────────────────────────────────────────────
        let currentKind = null, currentIdx = null, currentZ = 0;
        let currentList = [], currentListIndex = -1;
        let mouseDownX = 0, mouseDownY = 0;
        const deletedIdxSet = new Set();           // contact idxs
        const deletedOverlapSlices = new Set();     // 'pairIdx:zOffset' keys
        const deletedOverlapPairs = new Set();      // fully eliminated overlap idxs
        // Track current 3D position for indicator
        let curItemX = 0, curItemY = 0, curItemZ = 0, curItemZnm = 0;
        // Track current item's pair for gap junction marking
        let currentSource = '', currentTarget = '';
        // Gap junction state
        const gapJunctions = [];  // [{x, y, z, source, target, kind, idx, zOffset, timestamp}]
        let gjTraceIdx = traceInfo['_gap_junctions'];

        // ── Pre-populate gap junctions from overlap data ────────────
        // ─────────────────────────────────────────────────────────────────────
        // GAP-JUNCTION PRE-POPULATION
        // Seeds the gapJunctions array with anatomically known GJ pairs based on
        // electrophysiology and the Tier 2 circuit model.  For each known pair
        // the biggest overlap region (by area) is used as the GJ centroid.
        // Users can add/remove further sites interactively.
        //   LPTC chain (axo-axonal): VS1↔VS2↔VS3↔VS4, HSN↔HSE↔HSS
        //   LPTC↔MN (axon↔dendrite): VS/HS → MOS,  HS → MOT
        // ─────────────────────────────────────────────────────────────────────
        (function prePopulateGJs() {
            const knownGJPairs = [];
            ['L','R'].forEach(s => {
                // VS chain (axo-axonal)
                for (let k = 1; k <= 3; k++)
                    knownGJPairs.push({a:'VS'+k+'_'+s, b:'VS'+(k+1)+'_'+s, type:'axo-axonal (LPTC chain)'});
                // HS chain (axo-axonal)
                knownGJPairs.push({a:'HSN_'+s, b:'HSE_'+s, type:'axo-axonal (LPTC chain)'});
                knownGJPairs.push({a:'HSE_'+s, b:'HSS_'+s, type:'axo-axonal (LPTC chain)'});
                // VS ↔ MOS (LPTC axon ↔ MN dendrite)
                for (let k = 1; k <= 4; k++)
                    knownGJPairs.push({a:'VS'+k+'_'+s, b:'MOS_'+s, type:'axon\u2194dendrite (LPTC\u2194MN)'});
                // HS ↔ MOS (LPTC axon ↔ MN dendrite)
                ['HSN','HSE','HSS'].forEach(h => {
                    knownGJPairs.push({a:h+'_'+s, b:'MOS_'+s, type:'axon\u2194dendrite (LPTC\u2194MN)'});
                });
                // HS ↔ MOT (LPTC axon ↔ MN dendrite)
                ['HSN','HSE','HSS'].forEach(h => {
                    knownGJPairs.push({a:h+'_'+s, b:'MOT_'+s, type:'axon\u2194dendrite (LPTC\u2194MN)'});
                });
            });

            // For each known GJ pair, find the biggest overlap region
            knownGJPairs.forEach(pair => {
                // Search overlapList for matching pair (either direction)
                const matches = overlapList.filter(ov =>
                    (ov.source === pair.a && ov.target === pair.b) ||
                    (ov.source === pair.b && ov.target === pair.a));
                if (matches.length === 0) return;
                // Pick the one with biggest area
                let best = matches[0];
                matches.forEach(m => { if (m.area_um2 > best.area_um2) best = m; });
                // Place GJ at center of that overlap
                gapJunctions.push({
                    x: best.x, y: best.y, z: best.z,
                    source: pair.a, target: pair.b,
                    kind: pair.type,
                    idx: best.idx,
                    zOffset: 0,
                    timestamp: 'preset',
                });
            });
            if (gapJunctions.length > 0)
                console.log('[GJ] Pre-populated ' + gapJunctions.length + ' gap junctions from overlap data');
        })();

        // ── WebGL context loss recovery ─────────────────────────────
        plotDiv.addEventListener('webglcontextlost', function(e) {
            console.warn('WebGL context lost — will attempt recovery');
            e.preventDefault();
        });
        plotDiv.addEventListener('webglcontextrestored', function() {
            console.log('WebGL context restored — replotting');
            try { Plotly.redraw(plotDiv); } catch(e) { console.error('Redraw fail:', e); }
        });

        // ── Plotly safety — direct data mutation instead of Plotly.restyle ──
        // ─────────────────────────────────────────────────────────────────────
        // PLOTLY RENDERING WORKAROUND
        // Calling Plotly.restyle() on gl3d (WebGL) traces can hang the browser:
        // the returned Promise never resolves, or triggers a full scene recompute
        // that takes seconds.  Instead we mutate plotDiv.data[idx] in-place and
        // batch a single Plotly.redraw() via requestAnimationFrame so at most one
        // redraw fires per animation frame regardless of how many trace updates
        // were triggered.
        // ─────────────────────────────────────────────────────────────────────
        let redrawScheduled = false;
        function scheduleRedraw() {
            if (redrawScheduled) return;
            redrawScheduled = true;
            requestAnimationFrame(() => {
                redrawScheduled = false;
                try { Plotly.redraw(plotDiv); }
                catch(e) { console.warn('redraw error:', e.message); }
            });
        }
        function _setNested(obj, key, value) {
            // Handle dot-notation keys like 'marker.color' → obj.marker.color
            const parts = key.split('.');
            for (let k = 0; k < parts.length - 1; k++) {
                if (obj[parts[k]] === undefined) obj[parts[k]] = {};
                obj = obj[parts[k]];
            }
            obj[parts[parts.length - 1]] = value;
        }
        function safeRestyle(div, update, indices) {
            // Mutate trace data in-place, then schedule a single redraw
            try {
                const idxArr = Array.isArray(indices) ? indices : [indices];
                idxArr.forEach((trIdx, i) => {
                    const trace = plotDiv.data[trIdx];
                    if (!trace) return;
                    for (const key in update) {
                        const val = update[key];
                        let v;
                        if (Array.isArray(val) && val.length === idxArr.length) {
                            v = val[i];
                        } else if (Array.isArray(val) && val.length === 1 && idxArr.length === 1) {
                            v = val[0];
                        } else {
                            v = val;
                        }
                        _setNested(trace, key, v);
                    }
                });
                scheduleRedraw();
            } catch(e) { console.warn('safeRestyle error:', e.message); }
        }
        function safeRelayout(div, update) {
            try { Plotly.relayout(div, update); }
            catch(e) { console.warn('relayout error:', e.message); }
        }

        // ── Utility: debounce ───────────────────────────────────────────────
        // Returns a wrapper that delays invoking *fn* until *wait* ms after the
        // last call.  Used to collapse rapid-fire checkbox cascades and slider
        // drags into a single render pass, avoiding redundant full-scene redraws.
        function debounce(fn, wait) {
            let timer = null;
            return function() {
                const ctx = this, args = arguments;
                clearTimeout(timer);
                timer = setTimeout(function() { timer = null; fn.apply(ctx, args); }, wait);
            };
        }

        plotDiv.addEventListener('mousedown', e => {
            mouseDownX = e.clientX; mouseDownY = e.clientY;
        });

        // ── Helper: visible neurons ─────────────────────────────────
        function getVisibleNeurons() {
            return neuronNames.filter(n => {
                const cb = document.getElementById('mesh_' + n);
                return cb && cb.checked;
            });
        }

        function getVisibleItems(kind) {
            const itemList = kind === 'contact' ? contactList
                           : kind === 'synapse' ? synapseList
                           : overlapList;
            const seen = new Set();
            neuronNames.forEach(neuron => {
                if (kind === 'overlap') {
                    const allCb = document.getElementById('alloverlaps_' + neuron);
                    if (allCb && allCb.checked) {
                        itemList.filter(it => (it.source === neuron || it.target === neuron) && !it._eliminated)
                                .forEach(it => { if (it.idx >= 0) seen.add(it.idx); });
                    }
                    const curCb = document.getElementById('curoverlaps_' + neuron);
                    if (curCb && curCb.checked) {
                        const mv = getVisibleNeurons();
                        itemList.filter(it => {
                            const involves = it.source === neuron || it.target === neuron;
                            const other = it.source === neuron ? it.target : it.source;
                            return involves && mv.includes(other) && !it._eliminated;
                        }).forEach(it => { if (it.idx >= 0) seen.add(it.idx); });
                    }
                    // Putative GJ checkboxes are handled in gap-junction rendering only.
                } else {
                    const allCb = document.getElementById('all' + kind + 's_' + neuron);
                    if (allCb && allCb.checked) {
                        itemList.filter(it => it.source === neuron || it.target === neuron)
                                .forEach(it => seen.add(it.idx));
                    }
                    const curCb = document.getElementById('cur' + kind + 's_' + neuron);
                    if (curCb && curCb.checked) {
                        const mv = getVisibleNeurons();
                        itemList.filter(it => {
                            const involves = it.source === neuron || it.target === neuron;
                            const other = it.source === neuron ? it.target : it.source;
                            return involves && mv.includes(other);
                        }).forEach(it => seen.add(it.idx));
                    }
                }
            });
            return Array.from(seen).sort((a, b) => a - b);
        }

        // ── Per-neuron data caches ──────────────────────────────────
        const neuronContactData = {}, neuronSynapseData = {};
        function rebuildNeuronCaches() {
            neuronNames.forEach(n => {
                neuronContactData[n] = contactList.filter(
                    c => c.source === n || c.target === n);
                neuronSynapseData[n] = synapseList.filter(
                    s => s.source === n || s.target === n);
            });
        }
        rebuildNeuronCaches();

        // ── Rebuild trace data after filter/delete ──────────────────
        function rebuildTraceData(neuron, kind) {
            const allCb = document.getElementById('all' + kind + '_' + neuron);
            const curCb = document.getElementById('cur' + kind + '_' + neuron);
            const traceIdx = traceInfo[neuron + '_' + kind];
            if (traceIdx === undefined) return;
            const allData = kind === 'contacts'
                ? neuronContactData[neuron] : neuronSynapseData[neuron];
            let filtered = [];
            const allOn = allCb && allCb.checked;
            const curOn = curCb && curCb.checked;
            if (allOn) filtered = allData;
            else if (curOn) {
                const mv = getVisibleNeurons();
                filtered = allData.filter(item => {
                    const other = item.source === neuron ? item.target : item.source;
                    return mv.includes(other);
                });
            }
            const show = filtered.length > 0 && (allOn || curOn);
            const restyleObj = {
                x: [filtered.map(d => d.x)],
                y: [filtered.map(d => d.y)],
                z: [filtered.map(d => d.z)],
                visible: [show],
                customdata: [filtered.map(d => [d.x, d.y, d.z,
                    kind === 'contacts' ? 'contact' : 'synapse',
                    d.source, d.target, d.idx,
                    kind === 'contacts' ? (d.patch_num || 0)
                        : (d.isInh ? 'Inhibitory (GABA)' : 'Excitatory (ACh/Glut)')])]
            };
            if (kind === 'synapses') {
                restyleObj['marker.color'] = [filtered.map(d =>
                    d.isInh ? '#4488ff' : 'yellow')];
            }
            safeRestyle(plotDiv, restyleObj, [traceIdx]);
        }
        function recalcAllCurrentTraces() {
            neuronNames.forEach(n => {
                ['contacts', 'synapses'].forEach(kind => {
                    const cb = document.getElementById('cur' + kind + '_' + n);
                    if (cb && cb.checked) rebuildTraceData(n, kind);
                });
            });
        }

        // ── Overlap trace toggle ────────────────────────────────────
        function rebuildAllOverlapTrace(neuron) {
            const cb = document.getElementById('alloverlaps_' + neuron);
            const traceIdx = traceInfo[neuron + '_alloverlaps'];
            if (traceIdx === undefined) return;
            if (!cb || !cb.checked) {
                safeRestyle(plotDiv, { visible: [false] }, [traceIdx]);
                return;
            }
            // Rebuild from pairs data, filtering eliminated
            const pairs = overlapPairs[neuron] || [];
            const xs = [], ys = [], zs = [], cds = [];
            pairs.forEach(p => {
                if (!deletedOverlapPairs.has(p.idx)) {
                    for (let i = 0; i < p.x.length; i++) {
                        xs.push(p.x[i]); ys.push(p.y[i]); zs.push(p.z[i]);
                        cds.push([p.x[i], p.y[i], p.z[i],
                                  'overlap', p.source, p.target,
                                  p.idx, p.z_lo, p.z_hi]);
                    }
                }
            });
            safeRestyle(plotDiv, {
                x: [xs], y: [ys], z: [zs],
                customdata: [cds], visible: [xs.length > 0]
            }, [traceIdx]);
        }
        function rebuildCurrOverlapTrace(neuron) {
            const cb = document.getElementById('curoverlaps_' + neuron);
            const traceIdx = traceInfo[neuron + '_curoverlaps'];
            if (traceIdx === undefined) return;
            if (!cb || !cb.checked) {
                safeRestyle(plotDiv, { visible: [false] }, [traceIdx]);
                return;
            }
            const mv = getVisibleNeurons();
            const pairs = overlapPairs[neuron] || [];
            const xs = [], ys = [], zs = [], cds = [];
            pairs.forEach(p => {
                if (mv.includes(p.other) && !deletedOverlapPairs.has(p.idx)) {
                    for (let i = 0; i < p.x.length; i++) {
                        xs.push(p.x[i]); ys.push(p.y[i]); zs.push(p.z[i]);
                        cds.push([p.x[i], p.y[i], p.z[i],
                                  'overlap', p.source, p.target,
                                  p.idx, p.z_lo, p.z_hi]);
                    }
                }
            });
            safeRestyle(plotDiv, {
                x: [xs], y: [ys], z: [zs],
                customdata: [cds], visible: [xs.length > 0]
            }, [traceIdx]);
        }
        function recalcAllCurrOverlaps() {
            neuronNames.forEach(n => {
                const cb = document.getElementById('curoverlaps_' + n);
                if (cb && cb.checked) rebuildCurrOverlapTrace(n);
            });
        }

        // ── Putative GJ filtering (reusing overlap-face checkboxes) ──────────────
        function _isCurrentPair(gj) {
            if (!currentSource || !currentTarget) return false;
            return (gj.source === currentSource && gj.target === currentTarget)
                || (gj.source === currentTarget && gj.target === currentSource);
        }
        function _hideOverlapFaceTraces() {
            neuronNames.forEach(n => {
                const allIdx = traceInfo[n + '_alloverlapfaces'];
                const curIdx = traceInfo[n + '_curroverlapfaces'];
                if (allIdx !== undefined) safeRestyle(plotDiv, { visible: [false] }, [allIdx]);
                if (curIdx !== undefined) safeRestyle(plotDiv, { visible: [false] }, [curIdx]);
            });
        }
        function rebuildPutativeGJTrace() {
            if (gjTraceIdx === undefined) return;
            const allNeurons = neuronNames.filter(n => {
                const cb = document.getElementById('alloverlapfaces_' + n);
                return cb && cb.checked;
            });
            const curNeurons = neuronNames.filter(n => {
                const cb = document.getElementById('curroverlapfaces_' + n);
                return cb && cb.checked;
            });
            const anyOn = allNeurons.length > 0 || curNeurons.length > 0;
            const allSet = new Set(allNeurons), curSet = new Set(curNeurons);
            const xs = [], ys = [], zs = [], txt = [];
            if (anyOn) {
                gapJunctions.forEach((gj, i) => {
                    const involvesAll = allSet.has(gj.source) || allSet.has(gj.target);
                    const involvesCur = curSet.has(gj.source) || curSet.has(gj.target);
                    const include = involvesAll || (involvesCur && _isCurrentPair(gj));
                    if (!include) return;
                    xs.push(gj.x); ys.push(gj.y); zs.push(gj.z);
                    txt.push('GJ #' + (i + 1) + ': ' + gj.source + ' ↔ ' + gj.target);
                });
            }
            const tr = plotDiv.data[gjTraceIdx];
            if (!tr) return;
            tr.x = xs; tr.y = ys; tr.z = zs; tr.text = txt;
            tr.visible = anyOn && xs.length > 0;
            _hideOverlapFaceTraces();
            scheduleRedraw();
        }
        function rebuildAllOverlapFacesTrace(neuron) {
            rebuildPutativeGJTrace();
        }
        function rebuildCurrOverlapFacesTrace(neuron) {
            rebuildPutativeGJTrace();
        }
        function recalcAllCurrOverlapFaces() {
            rebuildPutativeGJTrace();
        }

        // ── Checkbox handlers ───────────────────────────────────────
        // Debounced wrappers for the three "rebuild every neuron" aggregates.
        // When the user rapidly toggles several checkboxes (e.g. "All Meshes"),
        // each individual change fires immediately for its own trace, but the
        // expensive full-scene recalculation is collapsed into one call 60 ms
        // after the burst ends.
        const _dRecalcCurrentTraces  = debounce(recalcAllCurrentTraces,  60);
        const _dRecalcCurrOverlaps   = debounce(recalcAllCurrOverlaps,   60);
        const _dRecalcCurrOvFaces    = debounce(recalcAllCurrOverlapFaces, 60);

        neuronNames.forEach(neuron => {
            const meshCb = document.getElementById('mesh_' + neuron);
            if (meshCb) meshCb.addEventListener('change', function() {
                const idx = traceInfo[neuron + '_mesh'];
                if (idx !== undefined)
                    safeRestyle(plotDiv, {visible: [this.checked]}, [idx]);
                // Defer the expensive "rebuild all neurons" passes so rapid
                // checkbox cascades coalesce into one redraw.
                _dRecalcCurrentTraces();
                _dRecalcCurrOverlaps();
                _dRecalcCurrOvFaces();
            });
            ['contacts', 'synapses'].forEach(kind => {
                const allCb = document.getElementById('all' + kind + '_' + neuron);
                if (allCb) allCb.addEventListener('change', function() {
                    rebuildTraceData(neuron, kind);
                    // When checked, populate EM viewer with only this neuron's items
                    if (this.checked && kind === 'contacts') {
                        const items = contactList.filter(
                            c => c.source === neuron || c.target === neuron);
                        if (items.length > 0) {
                            currentKind = 'contact';
                            currentList = items.map(c => c.idx);
                            currentListIndex = 0;
                            const first = items[0];
                            selectItem('contact', first.idx, first.x, first.y, first.z,
                                       first.source, first.target);
                            // Override currentList to only this neuron's contacts
                            currentList = items.map(c => c.idx);
                            currentListIndex = 0;
                            itemInfo.textContent = 'Contact 1/' + currentList.length
                                + ' (idx: ' + first.idx + ')  [' + neuron + ']';
                        }
                    }
                });
                const curCb = document.getElementById('cur' + kind + '_' + neuron);
                if (curCb) curCb.addEventListener('change', () => rebuildTraceData(neuron, kind));
            });
            const allOvCb = document.getElementById('alloverlaps_' + neuron);
            if (allOvCb) allOvCb.addEventListener('change', () => rebuildAllOverlapTrace(neuron));
            const curOvCb = document.getElementById('curoverlaps_' + neuron);
            if (curOvCb) curOvCb.addEventListener('change', () => rebuildCurrOverlapTrace(neuron));
            const allOvFCb = document.getElementById('alloverlapfaces_' + neuron);
            if (allOvFCb) allOvFCb.addEventListener('change', () => rebuildAllOverlapFacesTrace(neuron));
            const curOvFCb = document.getElementById('curroverlapfaces_' + neuron);
            if (curOvFCb) curOvFCb.addEventListener('change', () => rebuildCurrOverlapFacesTrace(neuron));
        });

        // Axis toggle
        const axesCb = document.getElementById('toggleAxes');
        if (axesCb) axesCb.addEventListener('change', function() {
            const s = this.checked;
            safeRelayout(plotDiv, {
                'scene.xaxis.title': s ? 'X (nm)' : '',
                'scene.yaxis.title': s ? 'Y (nm)' : '',
                'scene.zaxis.title': s ? 'Z (nm)' : '',
                'scene.xaxis.showticklabels': s,
                'scene.yaxis.showticklabels': s,
                'scene.zaxis.showticklabels': s
            });
        });

        // ── Bulk-load buttons ──────────────────────────────────────
        function _bulkToggle(cbPrefix, rebuildFn) {
            // Check if any are currently on
            const anyOn = neuronNames.some(n => {
                const cb = document.getElementById(cbPrefix + n);
                return cb && cb.checked;
            });
            const newState = !anyOn;
            neuronNames.forEach(n => {
                const cb = document.getElementById(cbPrefix + n);
                if (cb) {
                    cb.checked = newState;
                    rebuildFn(n);
                }
            });
        }
        document.getElementById('btnAllMesh').addEventListener('click', function() {
            const anyOn = neuronNames.some(n => {
                const cb = document.getElementById('mesh_' + n);
                return cb && cb.checked;
            });
            const newState = !anyOn;
            neuronNames.forEach(n => {
                const cb = document.getElementById('mesh_' + n);
                if (cb) {
                    cb.checked = newState;
                    const idx = traceInfo[n + '_mesh'];
                    if (idx !== undefined) safeRestyle(plotDiv, {visible: [newState]}, [idx]);
                }
            });
            _dRecalcCurrentTraces();
            _dRecalcCurrOverlaps();
            _dRecalcCurrOvFaces();
        });
        document.getElementById('btnAllOverlaps').addEventListener('click', function() {
            _bulkToggle('alloverlaps_', rebuildAllOverlapTrace);
        });
        document.getElementById('btnAllOvFaces').addEventListener('click', function() {
            _bulkToggle('alloverlapfaces_', rebuildAllOverlapFacesTrace);
        });
        document.getElementById('btnAllSynapses').addEventListener('click', function() {
            _bulkToggle('allsynapses_', function(n) { rebuildTraceData(n, 'synapses'); });
        });

        // ── 3D Position indicator ───────────────────────────────────
        // We update the _pos_indicator trace to show a sphere at current EM location
        const posTraceIdx = traceInfo['_pos_indicator'];
        function update3DIndicator(x, y, z) {
            if (posTraceIdx === undefined) return;
            curItemX = x; curItemY = y; curItemZ = z;
            safeRestyle(plotDiv, {
                x: [[x]], y: [[y]], z: [[z]], visible: [true]
            }, [posTraceIdx]);
        }
        function hide3DIndicator() {
            if (posTraceIdx === undefined) return;
            safeRestyle(plotDiv, { visible: [false] }, [posTraceIdx]);
        }

        // ── Click handler ───────────────────────────────────────────
        plotDiv.on('plotly_click', function(data) {
            if (!data.points || !data.points.length) return;
            const pt = data.points[0];
            if (data.event) {
                const dx = data.event.clientX - mouseDownX;
                const dy = data.event.clientY - mouseDownY;
                if (Math.sqrt(dx*dx + dy*dy) > 5) return;
            }
            const cd = pt.customdata;
            // Gap-junction trace click — navigate back to the item + Z-offset
            if (pt.curveNumber === gjTraceIdx && pt.pointNumber !== undefined) {
                const gj = gapJunctions[pt.pointNumber];
                if (gj) navigateToGJ(gj);
                return;
            }
            // Mesh3d (overlap faces) have no customdata — find nearest overlap pair
            if (!cd) {
                if (pt.x !== undefined && pt.y !== undefined && pt.z !== undefined) {
                    // Find closest overlap pair by centroid distance
                    let bestOv = null, bestDist = Infinity;
                    overlapList.forEach(ov => {
                        const d = Math.sqrt(
                            (pt.x - ov.x)**2 + (pt.y - ov.y)**2 + (pt.z - ov.z)**2);
                        if (d < bestDist) { bestDist = d; bestOv = ov; }
                    });
                    if (bestOv) {
                        selectItem('overlap', bestOv.idx, bestOv.x, bestOv.y, bestOv.z,
                                   bestOv.source, bestOv.target);
                        // Jump to the Z-slice nearest the actual click point
                        jumpToNearestZ(bestOv, pt.z);
                    }
                }
                return;
            }
            const [x, y, z, kind, source, target, idx] = cd;
            if (kind === 'mesh') return;
            selectItem(kind, idx, x, y, z, source, target);
            // For overlap scatter clicks, jump to the Z-slice nearest the vertex
            if (kind === 'overlap') {
                const ov = overlapList.find(o => o.idx === idx);
                if (ov) jumpToNearestZ(ov, z);
            }
        });

        // Navigate to a gap junction's exact item and Z-offset
        function navigateToGJ(gj) {
            // Re-select the original item
            const list = gj.kind === 'contact' ? contactList
                       : gj.kind === 'synapse' ? synapseList : overlapList;
            const item = list.find(i => i.idx === gj.idx);
            if (!item) return;
            selectItem(gj.kind, gj.idx, item.x, item.y, item.z, gj.source, gj.target);
            // Now navigate to the exact Z-offset where the GJ was marked
            if (gj.zOffset !== 0 && gj.zOffset !== undefined) {
                zSlider.value = gj.zOffset;
                loadImage(gj.kind, gj.idx, gj.zOffset);
            }
        }

        // Jump Z-slider to the offset closest to a clicked Z coordinate
        function jumpToNearestZ(ov, clickedZ) {
            // ov.z = z_base_nm (absolute Z of lowest EM slice)
            const rawOffset = Math.round((clickedZ - ov.z) / 40);
            let best = Math.max(parseInt(zSlider.min), Math.min(parseInt(zSlider.max), rawOffset));
            best = snapToValidZ(best);
            zSlider.value = best;
            loadImage('overlap', ov.idx, best);
        }

        function selectItem(kind, idx, x, y, z, source, target) {
            currentKind = kind;
            currentIdx = idx;
            currentSource = source;
            currentTarget = target;
            updateGJTrace();
            currentList = getVisibleItems(kind);
            currentListIndex = currentList.indexOf(idx);

            emTitle.textContent = kind.charAt(0).toUpperCase() + kind.slice(1) + ' #' + idx;
            emLocation.textContent = source + ' \u2192 ' + target
                + ' at (' + Math.round(x) + ', ' + Math.round(y) + ', ' + Math.round(z) + ')';
            itemInfo.textContent = kind.charAt(0).toUpperCase() + kind.slice(1)
                + ' ' + (currentListIndex + 1) + '/' + currentList.length + ' (idx: ' + idx + ')';

            // Show delete buttons for contacts AND overlaps
            btnDeleteSlice.style.display = (kind === 'contact' || kind === 'overlap')
                ? 'inline-block' : 'none';
            btnDeleteAll.style.display = (kind === 'overlap')
                ? 'inline-block' : 'none';
            deletedBanner.style.display = deletedIdxSet.has(kind + ':' + idx) ? 'block' : 'none';

            // Dynamic Z-slider range — set min/max BEFORE value
            if (kind === 'overlap') {
                const ov = overlapList.find(o => o.idx === idx);
                const zLo = ov ? ov.z_lo : -20;
                const zHi = ov ? ov.z_hi : 20;
                const vz = (ov && ov.valid_z && ov.valid_z.length) ? ov.valid_z : null;
                zSlider.min = zLo;
                zSlider.max = zHi;
                zSlider.dataset.validZ = vz ? JSON.stringify(vz) : '';
                // Snap to nearest valid Z (0 may not be in valid_z)
                const startZ = snapToValidZ(0);
                zSlider.value = startZ;
                currentZ = startZ;
                const nSlices = vz ? vz.length : (zHi - zLo + 1);
                zNote.textContent = nSlices + ' EM slices  (Z: ' + zLo + ' to ' + zHi + ')';
            } else {
                zSlider.min = -20;
                zSlider.max = 20;
                zSlider.dataset.validZ = '';
                zSlider.value = 0;
                currentZ = 0;
            }

            // Set base Z and initial diamond position
            curItemZnm = z;  // default: use the passed-in Z
            if (kind === 'overlap') {
                const ov = overlapList.find(o => o.idx === idx);
                if (ov) {
                    // ov.z = z_base_nm (absolute Z of the first EM slice)
                    curItemZnm = ov.z;
                    // Start diamond at EM-slice-0 center coords
                    const sc0 = ov.slice_coords && ov.slice_coords[String(currentZ)];
                    if (sc0) { x = sc0[0]; y = sc0[1]; }
                }
            }
            curItemX = x; curItemY = y;
            update3DIndicator(x, y, curItemZnm + currentZ * 40);

            loadImage(kind, idx, currentZ);
        }

        // ── Image loading ───────────────────────────────────────────
        let _loadGen = 0;  // generation counter to prevent stale callbacks
        function loadImage(kind, idx, zOffset) {
            const imgData = snapshotMap[kind] && snapshotMap[kind][idx];
            if (!imgData) {
                emImage.style.display = 'none';
                emPlaceholder.style.display = 'block';
                emPlaceholder.textContent = 'No snapshot for ' + kind + ' ' + idx;
                return;
            }
            const gen = ++_loadGen;
            currentZ = zOffset;  // sync so stepValidZ always sees latest offset
            if (zOffset === 0) {
                // Clear stale handlers from previous loadZStackImage to prevent
                // race condition where old onload resets currentZ
                emImage.onload = null;
                emImage.onerror = null;
                emImage.src = imgData;
                emImage.style.display = 'block';
                emPlaceholder.style.display = 'none';
                updateZValue(0);
            } else {
                loadZStackImage(kind, idx, zOffset, gen);
            }
            // For overlaps: update X,Y from per-slice EM center coords
            if (kind === 'overlap') {
                const ov = overlapList.find(o => o.idx === idx);
                if (ov) {
                    if (ov.slice_coords) {
                        const sc = ov.slice_coords[String(zOffset)];
                        if (sc) {
                            curItemX = sc[0]; curItemY = sc[1];
                        } else {
                            // No per-slice coords for this z — fall back to overlap centroid
                            curItemX = ov.x; curItemY = ov.y;
                        }
                    } else {
                        curItemX = ov.x; curItemY = ov.y;
                    }
                }
            }
            // Update 3D diamond to match current EM slice position
            const newZ = curItemZnm + zOffset * 40;
            update3DIndicator(curItemX, curItemY, newZ);
            // Update location display with diamond coords
            emLocation.textContent = currentSource + ' \u2192 ' + currentTarget
                + ' at (' + Math.round(curItemX) + ', ' + Math.round(curItemY) + ', ' + Math.round(curItemZ) + ')';
        }

        function loadZStackImage(kind, idx, zOffset, gen) {
            const sign = zOffset >= 0 ? '+' : '-';
            const zStr = 'z' + sign + String(Math.abs(zOffset)).padStart(3, '0');
            const mapped = snapshotZMap
                && snapshotZMap[kind]
                && snapshotZMap[kind][idx]
                && snapshotZMap[kind][idx][zOffset];
            emImage.onerror = function() {
                if (gen !== _loadGen) return;  // stale callback
                const center = snapshotMap[kind] && snapshotMap[kind][idx];
                if (center && zOffset !== 0) emImage.src = center;
            };
            emImage.onload = function() {
                if (gen !== _loadGen) return;  // stale callback
                emImage.style.display = 'block';
                emPlaceholder.style.display = 'none';
                updateZValue(zOffset);
            };
            let filePrefix = kind + '_' + idx;
            if (kind === 'contact' && contactClusterMap[idx] !== undefined) {
                filePrefix = 'cluster_' + contactClusterMap[idx];
            }
            emImage.src = mapped || ('em_snaps/' + filePrefix + '_' + zStr + '.png');
        }

        function updateZValue(z) {
            currentZ = z;
            const nm = z * 40;
            if (z === 0) {
                zValue.textContent = '0 (center)';
                zNote.style.color = '#0a0';
            } else {
                const s = z > 0 ? '+' : '';
                zValue.textContent = s + z + ' (' + s + nm + 'nm)';
                zNote.style.color = '#888';
            }
            // Check if this overlap slice was deleted
            if (currentKind === 'overlap') {
                const key = currentIdx + ':' + z;
                deletedBanner.style.display = deletedOverlapSlices.has(key) ? 'block' : 'none';
                deletedBanner.textContent = deletedOverlapSlices.has(key)
                    ? '\u2717 DELETED \u2014 this slice has been removed' : '';
            } else if (currentKind === 'contact') {
                const isDel = deletedIdxSet.has('contact:' + currentIdx);
                deletedBanner.style.display = isDel ? 'block' : 'none';
                deletedBanner.textContent = isDel ? '\u2717 DELETED' : '';
            }
        }

        // Debounced EM contrast update to avoid excessive re-renders
        let emOpacityDebounceTimer = null;
        function updateEMOpacity() {
            if (emOpacityDebounceTimer !== null) clearTimeout(emOpacityDebounceTimer);
            emOpacityDebounceTimer = setTimeout(() => {
                const pct = Math.max(50, Math.min(250, parseInt(emOpacitySlider.value || '100', 10)));
                emImage.style.opacity = '1';
                emImage.style.filter = 'contrast(' + pct + '%)';
                emOpacityValue.textContent = pct + '%';
                emOpacityDebounceTimer = null;
            }, 50);  // 50ms debounce
        }

        function downloadCurrentEMSnapshot() {
            if (!currentKind || currentIdx === null || !emImage.src) {
                alert('Select an item first to download its EM image.');
                return;
            }
            const img = emImage;
            const width = img.naturalWidth || img.width || 1024;
            const height = img.naturalHeight || img.height || 768;
            const absZ = Math.round(curItemZnm + currentZ * 40);
            const midX = Math.round(curItemX);
            const midY = Math.round(curItemY);
            const touching = (currentSource && currentTarget) ? (currentSource + ' <-> ' + currentTarget) : 'n/a';
            const line1 = currentKind.toUpperCase() + ' #' + currentIdx + '  z=' + currentZ + ' (' + absZ + ' nm)';
            const line2 = 'slice midpoint: (' + midX + ', ' + midY + ', ' + absZ + ')';
            const line3 = 'touching cells: ' + touching;
            const fileName = 'em_' + currentKind + '_' + currentIdx
                + '_x' + midX + '_y' + midY + '_z' + absZ
                + '_zoff' + (currentZ >= 0 ? '+' : '-') + String(Math.abs(currentZ)).padStart(3, '0') + '.png';

            let pngUrl = img.src;
            try {
                const canvas = document.createElement('canvas');
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                if (!ctx) throw new Error('No 2D context');

                ctx.drawImage(img, 0, 0, width, height);
                const boxH = 62;
                ctx.fillStyle = 'rgba(0,0,0,0.60)';
                ctx.fillRect(0, height - boxH, width, boxH);
                ctx.fillStyle = '#ffffff';
                ctx.font = '16px sans-serif';
                ctx.fillText(line1, 12, height - 40);
                ctx.font = '14px sans-serif';
                ctx.fillText(line2, 12, height - 22);
                ctx.fillText(line3, 12, height - 6);

                pngUrl = canvas.toDataURL('image/png');
            } catch (e) {
                // Fallback for security-restricted canvases (e.g., strict file:// contexts)
                pngUrl = img.src;
            }

            const popup = window.open('', '_blank');
            if (!popup) {
                const dl = document.createElement('a');
                dl.download = fileName;
                dl.href = pngUrl;
                document.body.appendChild(dl);
                dl.click();
                dl.remove();
                return;
            }

            const safeTitle = (currentSource || '?') + ' -> ' + (currentTarget || '?');
            popup.document.write(
                '<!doctype html><html><head><meta charset="utf-8"><title>' + fileName + '</title>'
                + '<style>body{font-family:Arial,sans-serif;background:#111;color:#eee;margin:16px;}'
                + '.meta{margin-bottom:10px;line-height:1.45;} img{max-width:100%;height:auto;border:1px solid #444;}'
                + 'button{margin-right:8px;padding:6px 10px;background:#2a2a2a;color:#fff;border:1px solid #555;cursor:pointer;}'
                + '</style></head><body>'
                + '<div class="meta"><div><b>' + safeTitle + '</b></div><div>' + line1 + '</div><div>' + line2 + '</div><div>' + line3 + '</div></div>'
                + '<div><button id="dlBtn">Download PNG</button><button id="closeBtn">Close</button></div>'
                + '<div style="margin-top:12px;"><img src="' + pngUrl + '" alt="EM snapshot"></div>'
                + '<script>'
                + 'const u=' + JSON.stringify(pngUrl) + ';'
                + 'const f=' + JSON.stringify(fileName) + ';'
                + 'document.getElementById("dlBtn").onclick=function(){const a=document.createElement("a");a.href=u;a.download=f;document.body.appendChild(a);a.click();a.remove();};'
                + 'document.getElementById("closeBtn").onclick=function(){window.close();};'
                + 'setTimeout(function(){document.getElementById("dlBtn").click();}, 40);'
                + '<\/script>'
                + '</body></html>'
            );
            popup.document.close();
        }

        emOpacitySlider.addEventListener('input', updateEMOpacity);
        btnDownloadEM.addEventListener('click', downloadCurrentEMSnapshot);
        updateEMOpacity();

        // ── Z-stack navigation ──────────────────────────────────────
        zSlider.addEventListener('input', function() {
            if (currentKind && currentIdx !== null) {
                let z = parseInt(this.value);
                z = snapToValidZ(z);
                this.value = z;
                loadImage(currentKind, currentIdx, z);
            }
        });
        function stepValidZ(cur, direction) {
            const raw = zSlider.dataset.validZ;
            if (!raw) return cur + direction;
            let vz;
            try { vz = JSON.parse(raw); } catch(e) { return cur + direction; }
            if (!vz || !vz.length) return cur + direction;
            if (direction < 0) {
                for (let i = vz.length - 1; i >= 0; i--) {
                    if (vz[i] < cur) return vz[i];
                }
                // No valid Z below cur — if cur is not in valid_z, snap to nearest
                if (vz.indexOf(cur) === -1) return snapToValidZ(cur);
                return cur;  // already at lowest valid Z
            } else {
                for (let i = 0; i < vz.length; i++) {
                    if (vz[i] > cur) return vz[i];
                }
                // No valid Z above cur — if cur is not in valid_z, snap to nearest
                if (vz.indexOf(cur) === -1) return snapToValidZ(cur);
                return cur;  // already at highest valid Z
            }
        }
        function snapToValidZ(z) {
            const raw = zSlider.dataset.validZ;
            if (!raw) return z;
            let vz;
            try { vz = JSON.parse(raw); } catch(e) { return z; }
            if (!vz || !vz.length) return z;
            let best = vz[0], bestDist = Math.abs(z - vz[0]);
            for (let i = 1; i < vz.length; i++) {
                const d = Math.abs(z - vz[i]);
                if (d < bestDist) { best = vz[i]; bestDist = d; }
            }
            return best;
        }
        document.getElementById('btnPrevZ').addEventListener('click', () => {
            if (currentKind && currentIdx !== null) {
                const nz = stepValidZ(currentZ, -1);
                if (nz >= parseInt(zSlider.min)) {
                    zSlider.value = nz;
                    loadImage(currentKind, currentIdx, nz);
                }
            }
        });
        document.getElementById('btnNextZ').addEventListener('click', () => {
            if (currentKind && currentIdx !== null) {
                const nz = stepValidZ(currentZ, +1);
                if (nz <= parseInt(zSlider.max)) {
                    zSlider.value = nz;
                    loadImage(currentKind, currentIdx, nz);
                }
            }
        });

        // ── Item navigation ─────────────────────────────────────────
        document.getElementById('btnPrevItem').addEventListener('click', () => {
            if (currentKind && currentListIndex > 0)
                navigateItem(currentListIndex - 1);
        });
        document.getElementById('btnNextItem').addEventListener('click', () => {
            if (currentKind && currentListIndex < currentList.length - 1)
                navigateItem(currentListIndex + 1);
        });
        function navigateItem(newPos) {
            currentListIndex = newPos;
            const newIdx = currentList[newPos];
            currentIdx = newIdx;
            const list = currentKind === 'contact' ? contactList
                       : currentKind === 'synapse' ? synapseList : overlapList;
            const item = list.find(i => i.idx === newIdx);
            if (item) {
                emTitle.textContent = currentKind.charAt(0).toUpperCase()
                    + currentKind.slice(1) + ' #' + newIdx;
                emLocation.textContent = item.source + ' \u2192 ' + item.target;
                curItemX = item.x; curItemY = item.y;
                curItemZnm = item.z;
                currentSource = item.source;
                currentTarget = item.target;
                updateGJTrace();
            }
            itemInfo.textContent = currentKind.charAt(0).toUpperCase()
                + currentKind.slice(1) + ' '
                + (newPos + 1) + '/' + currentList.length + ' (idx: ' + newIdx + ')';
            deletedBanner.style.display = 'none';
            btnDeleteSlice.style.display = (currentKind === 'contact' || currentKind === 'overlap')
                ? 'inline-block' : 'none';
            btnDeleteAll.style.display = (currentKind === 'overlap')
                ? 'inline-block' : 'none';

            // Set slider range BEFORE value, then snap to valid Z
            if (currentKind === 'overlap') {
                const ov = overlapList.find(o => o.idx === newIdx);
                const zLo = ov ? ov.z_lo : -20;
                const zHi = ov ? ov.z_hi : 20;
                const vz = (ov && ov.valid_z && ov.valid_z.length) ? ov.valid_z : null;
                zSlider.min = zLo; zSlider.max = zHi;
                zSlider.dataset.validZ = vz ? JSON.stringify(vz) : '';
                const startZ = snapToValidZ(0);
                zSlider.value = startZ;
                currentZ = startZ;
                const nSlices = vz ? vz.length : (zHi - zLo + 1);
                zNote.textContent = nSlices + ' EM slices  (Z: ' + zLo + ' to ' + zHi + ')';
            } else {
                zSlider.min = -20; zSlider.max = 20;
                zSlider.dataset.validZ = '';
                zSlider.value = 0;
                currentZ = 0;
            }
            loadImage(currentKind, newIdx, currentZ);
        }

        // ── Helper: update slider text after deletion ────────────
        function updateSliderInfo(ov) {
            if (!ov) return;
            const vz = ov.valid_z || [];
            if (vz.length > 0) {
                const lo = Math.min(...vz);
                const hi = Math.max(...vz);
                zSlider.min = lo;
                zSlider.max = hi;
                zNote.textContent = vz.length + ' EM slices  (Z: ' + lo + ' to ' + hi + ')';
            } else {
                zNote.textContent = '0 EM slices  (all deleted)';
            }
        }

        // ── Helper: update BOTH directions in overlapTable ──────────
        function findTableRows(source, target) {
            return overlapTable.filter(
                r => (r.source === source && r.target === target)
                  || (r.source === target && r.target === source));
        }

        // ── Helper: recalculate pair area from ALL sub-clusters ─────
        // Each sub-cluster has area_um2 and orig_n_slices.
        // Compute overall remaining fraction, apply to each tableRow's original area.
        function recalcPairArea(source, target) {
            // FEATURE: EM SLICE DELETION → AREA RECALCULATION
            // When user deletes individual Z-slices or entire overlap pairs,
            // this function recalculates the total contact/overlap area.
            // 
            // Algorithm:
            // 1. Find all spatial sub-clusters for this neuron pair (there may be multiple
            //    disconnected contact regions > 10 um apart, each getting its own cluster)
            // 2. For each sub-cluster, compute remaining area as:
            //    orig_area_um2 * (remaining_slices / original_slices)
            //    This gives proportional area loss (e.g., delete 2/10 slices → 80% area remains)
            // 3. Sum across all sub-clusters = total pair area
            // 4. Update the overlap matrix table to show new area
            
            const subs = overlapList.filter(
                o => (o.source === source && o.target === target)
                  || (o.source === target && o.target === source));
            let totalOrigArea = 0;
            let totalRemainArea = 0;
            let allEliminated = true;
            subs.forEach(sub => {
                const origN = sub.orig_n_slices || 1;  // Original slice count
                const curN = (sub.valid_z || []).length;  // Current remaining slices
                totalOrigArea += sub.area_um2;
                if (curN > 0) {
                    // Proportional area: only count remaining slices
                    totalRemainArea += sub.area_um2 * (curN / origN);
                    allEliminated = false;
                }
            });
            const fraction = totalOrigArea > 0
                ? totalRemainArea / totalOrigArea : 0;
            // Update both directions in overlapTable (undirected pairs)
            const tableRows = findTableRows(source, target);
            tableRows.forEach(row => {
                if (!row._orig_area) row._orig_area = row.area;  // Save original before first deletion
                row.area = allEliminated ? 0 : row._orig_area * fraction;
                row.status = allEliminated ? 'eliminated' : 'active';
            });
        }

        // ── Helper: refresh 3D traces + matrix after any deletion ───
        function refreshAfterDelete() {
            // Rebuild overlap vertex and face traces for all neurons
            neuronNames.forEach(n => {
                rebuildAllOverlapTrace(n);
                rebuildCurrOverlapTrace(n);
                rebuildAllOverlapFacesTrace(n);
                rebuildCurrOverlapFacesTrace(n);
            });
            // Invalidate cached heatmap; re-render if modal is currently open
            _tabRendered.overlaps = false;
            if (matrixModal.classList.contains('active')) {
                renderHeatmap();
                _tabRendered.overlaps = true;
            }
        }

        // ── DELETE SLICE (single contact or single overlap z-slice) ─
        btnDeleteSlice.addEventListener('click', function() {
            if (currentIdx === null) return;

            if (currentKind === 'contact') {
                const item = contactList.find(c => c.idx === currentIdx);
                if (!item) return;
                const msg = 'Delete contact #' + currentIdx
                    + ' (' + item.source + ' \u2192 ' + item.target + ')?';
                if (!confirm(msg)) return;

                deletedIdxSet.add('contact:' + currentIdx);
                deletedItems.push({
                    kind: 'contact', idx: currentIdx,
                    source: item.source, target: item.target,
                    patch_num: item.patch_num || 0,
                    patch_area: item.patch_area || 0,
                    x: item.x, y: item.y, z: item.z
                });
                deletedBanner.style.display = 'block';
                deletedBanner.textContent = '\u2717 DELETED';
                contactList = contactList.filter(c => c.idx !== currentIdx);
                rebuildNeuronCaches();
                neuronNames.forEach(n => rebuildTraceData(n, 'contacts'));
                const tableRow = overlapTable.find(
                    r => r.source === item.source && r.target === item.target);
                if (tableRow) {
                    tableRow.area = Math.max(0, tableRow.area - (item.patch_area || 0));
                    tableRow.patches = Math.max(0, tableRow.patches - 1);
                    if (tableRow.patches === 0) { tableRow.area = 0; tableRow.status = 'eliminated'; }
                }
                infoText.textContent = '\u2717 Deleted contact #' + currentIdx
                    + '  [' + deletedItems.length + ' deletions]';
                // Invalidate heatmap cache and re-render if modal is open
                _tabRendered.overlaps = false;
                if (matrixModal.classList.contains('active')) { renderHeatmap(); _tabRendered.overlaps = true; }

            } else if (currentKind === 'overlap') {
                const key = currentIdx + ':' + currentZ;
                if (deletedOverlapSlices.has(key)) return;  // already deleted
                const ov = overlapList.find(o => o.idx === currentIdx);
                const pairLabel = ov ? ov.source + ' \u2194 ' + ov.target : '';
                const msg = 'Delete overlap slice #' + currentIdx
                    + ' Z=' + currentZ + (pairLabel ? ' (' + pairLabel + ')' : '') + '?';
                if (!confirm(msg)) return;

                deletedOverlapSlices.add(key);
                deletedItems.push({
                    kind: 'overlap_slice', idx: currentIdx,
                    z_offset: currentZ,
                    source: ov ? ov.source : '', target: ov ? ov.target : ''
                });
                deletedBanner.style.display = 'block';
                deletedBanner.textContent = '\u2717 DELETED \u2014 this slice has been removed';

                // Remove this z from valid_z and recalculate area
                if (ov && ov.valid_z) {
                    ov.valid_z = ov.valid_z.filter(z => {
                        return !deletedOverlapSlices.has(currentIdx + ':' + z);
                    });
                    // Update slider valid_z + text
                    zSlider.dataset.validZ = ov.valid_z.length
                        ? JSON.stringify(ov.valid_z) : '';
                    updateSliderInfo(ov);

                    // Recalculate pair area from ALL sub-clusters
                    recalcPairArea(ov.source, ov.target);
                    if (ov.valid_z.length === 0) {
                        deletedOverlapPairs.add(currentIdx);
                    }
                }
                infoText.textContent = '\u2717 Deleted overlap slice #' + currentIdx
                    + ' Z=' + currentZ + '  [' + deletedItems.length + ' deletions]';
                refreshAfterDelete();

                // Auto-advance to next valid slice
                if (ov && ov.valid_z && ov.valid_z.length > 0) {
                    const nextZ = stepValidZ(currentZ, +1);
                    if (nextZ !== currentZ) {
                        zSlider.value = nextZ;
                        loadImage(currentKind, currentIdx, nextZ);
                    } else {
                        const prevZ = stepValidZ(currentZ, -1);
                        if (prevZ !== currentZ) {
                            zSlider.value = prevZ;
                            loadImage(currentKind, currentIdx, prevZ);
                        }
                    }
                }
            }
        });

        // ── DELETE ALL (entire overlap pair — all slices) ───────────
        btnDeleteAll.addEventListener('click', function() {
            if (currentIdx === null || currentKind !== 'overlap') return;
            const ov = overlapList.find(o => o.idx === currentIdx);
            if (!ov) return;
            const pairLabel = ov.source + ' \u2194 ' + ov.target;
            const nSlices = ov.valid_z ? ov.valid_z.length : 0;
            const msg = 'DELETE ALL ' + nSlices + ' slices of overlap #' + currentIdx
                + ' (' + pairLabel + ')?\n\nThis will eliminate the entire overlap pair.';
            if (!confirm(msg)) return;

            // Mark every slice as deleted
            if (ov.valid_z) {
                ov.valid_z.forEach(z => {
                    const key = currentIdx + ':' + z;
                    if (!deletedOverlapSlices.has(key)) {
                        deletedOverlapSlices.add(key);
                    }
                });
            }
            deletedItems.push({
                kind: 'overlap_all', idx: currentIdx,
                source: ov.source, target: ov.target,
                n_slices: nSlices
            });

            // Clear valid_z and set area to 0
            ov.valid_z = [];
            zSlider.dataset.validZ = '';
            updateSliderInfo(ov);

            // Recalculate pair area from ALL sub-clusters
            // (other sub-clusters of this pair may still have slices)
            recalcPairArea(ov.source, ov.target);

            deletedBanner.style.display = 'block';
            deletedBanner.textContent = '\u2717 DELETED \u2014 entire overlap pair eliminated ('
                + nSlices + ' slices)';
            infoText.textContent = '\u2717 Deleted ALL slices of overlap #' + currentIdx
                + ' (' + pairLabel + ')  [' + deletedItems.length + ' deletions]';

            deletedOverlapPairs.add(currentIdx);
            // Mark overlap as eliminated in overlapList
            ov._eliminated = true;
            // Refresh 3D traces and matrix
            refreshAfterDelete();
            // Update navigation list (exclude eliminated overlaps)
            currentList = getVisibleItems(currentKind);
            currentListIndex = currentList.indexOf(currentIdx);
            itemInfo.textContent = 'Overlap '
                + (currentListIndex + 1) + '/' + currentList.length
                + ' (idx: ' + currentIdx + ') [ELIMINATED]';
        });

        // ── Export deletions ────────────────────────────────────────
        btnExport.addEventListener('click', function() {
            if (deletedItems.length === 0 && gapJunctions.length === 0) {
                alert('No deletions or gap junctions yet.'); return;
            }
            const exportData = {
                deletedItems: deletedItems,
                gapJunctions: gapJunctions
            };
            const blob = new Blob(
                [JSON.stringify(exportData, null, 2)],
                {type: 'application/json'}
            );
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'viewer_annotations.json';
            a.click();
            URL.revokeObjectURL(a.href);
            infoText.textContent = '\u2713 Exported ' + deletedItems.length + ' deletions, '
                + gapJunctions.length + ' gap junctions';
        });

        // ── Gap-junction marking ────────────────────────────────────
        function focusPutativeGJOnCurrentPair() {
            if (!currentSource || !currentTarget) return;
            neuronNames.forEach(n => {
                const allCb = document.getElementById('alloverlapfaces_' + n);
                const curCb = document.getElementById('curroverlapfaces_' + n);
                if (allCb) allCb.checked = false;
                if (curCb) curCb.checked = (n === currentSource || n === currentTarget);
            });
        }
        btnMarkGJ.addEventListener('click', function() {
            if (currentKind === null || currentIdx === null) {
                alert('Select an overlap, contact, or synapse first.'); return;
            }
            if (!currentSource || !currentTarget) {
                alert('Cannot determine neuron pair for current item.'); return;
            }

            const gj = {
                x: curItemX, y: curItemY, z: curItemZ,
                source: currentSource, target: currentTarget,
                kind: currentKind, idx: currentIdx,
                zOffset: currentZ,
                timestamp: new Date().toISOString()
            };
            gapJunctions.push(gj);
            focusPutativeGJOnCurrentPair();
            updateGJTrace();
            updateRemoveGJButton();
            // Invalidate GJ + connectivity tabs so they re-render next visit
            _tabRendered.gapjunctions = false;
            _tabRendered.connectivity = false;
            infoText.textContent = '\u26a1 Gap junction #' + gapJunctions.length
                + ' marked at ' + currentSource + ' \u2194 ' + currentTarget
                + ' (' + Math.round(curItemX) + ', ' + Math.round(curItemY)
                + ', ' + Math.round(curItemZ) + ')';
        });

        btnRemoveGJ.addEventListener('click', function() {
            // Find the nearest gap junction to the current 3D indicator position
            let bestIdx = -1, bestDist = Infinity;
            gapJunctions.forEach((gj, i) => {
                const d = Math.sqrt(
                    (curItemX - gj.x) ** 2 +
                    (curItemY - gj.y) ** 2 +
                    (curItemZ - gj.z) ** 2);
                if (d < bestDist) { bestDist = d; bestIdx = i; }
            });
            if (bestIdx >= 0) {
                const removed = gapJunctions.splice(bestIdx, 1)[0];
                updateGJTrace();
                updateRemoveGJButton();
                // Invalidate GJ + connectivity tabs
                _tabRendered.gapjunctions = false;
                _tabRendered.connectivity = false;
                infoText.textContent = '\u2716 Removed gap junction #' + (bestIdx + 1)
                    + ' (' + removed.source + ' \u2194 ' + removed.target + ')';
            }
        });

        function updateRemoveGJButton() {
            // Show Remove GJ button only when there's a nearby gap junction
            if (gapJunctions.length === 0) { btnRemoveGJ.style.display = 'none'; return; }
            let minDist = Infinity;
            gapJunctions.forEach(gj => {
                const d = Math.sqrt(
                    (curItemX - gj.x) ** 2 +
                    (curItemY - gj.y) ** 2 +
                    (curItemZ - gj.z) ** 2);
                if (d < minDist) minDist = d;
            });
            // Show if any GJ is within 5000nm of current indicator
            btnRemoveGJ.style.display = (minDist < 5000) ? 'inline-block' : 'none';
        }

        function updateGJTrace() {
            rebuildPutativeGJTrace();
        }

        // ── Matrix modal ────────────────────────────────────────────
        // LAZY TAB RENDERING: only render a tab's content when it is first
        // clicked (or when data changes).  The three static-data tabs
        // (overlaps, gapjunctions, connectivity) are rendered once on first
        // open; Tier 1 / Tier 2 circuit tabs re-render every visit because
        // they contain live simulation state.
        const _tabRendered = { overlaps: false, gapjunctions: false, connectivity: false };

        btnMatrix.addEventListener('click', function() {
            // Always refresh the default (overlaps) tab on open; the others
            // stay cached until explicitly clicked.
            if (!_tabRendered.overlaps) {
                renderHeatmap();
                _tabRendered.overlaps = true;
            } else {
                // Re-render only if deletion state changed since last open
                // (overlapTable may have been mutated by delete operations).
                renderHeatmap();
            }
            matrixModal.classList.add('active');
        });
        document.getElementById('matrixClose').addEventListener('click', () => {
            matrixModal.classList.remove('active');
        });
        matrixModal.addEventListener('click', function(e) {
            if (e.target === matrixModal) matrixModal.classList.remove('active');
        });
        // Tab switching — render each tab's content on first click only
        const tabMap = {
            overlaps:     { el: 'tabOverlaps',      title: 'Overlap Area Matrix (\u00b5m\u00b2)' },
            gapjunctions: { el: 'tabGapJunctions',   title: 'Putative Gap Junctions' },
            connectivity: { el: 'tabConnectivity',   title: 'Connectivity Matrix (GJ + Chemical Synapses)' },
            circuit:      { el: 'tabCircuit',         title: 'Tier 1 Circuit Model' },
            mc:           { el: 'tabMC',              title: 'Multi-Compartment Model (Tier 2)' },
        };
        document.querySelectorAll('.modal-tab').forEach(tab => {
            tab.addEventListener('click', function() {
                document.querySelectorAll('.modal-tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.modal-tab-content').forEach(c => c.classList.remove('active'));
                this.classList.add('active');
                const target = this.dataset.tab;
                const info = tabMap[target];
                if (info) {
                    document.getElementById(info.el).classList.add('active');
                    modalTitle.textContent = info.title;
                    // Lazy render: GJ and connectivity run once; circuit tabs run every visit
                    if (target === 'gapjunctions' && !_tabRendered.gapjunctions) {
                        renderGJTab();
                        _tabRendered.gapjunctions = true;
                    } else if (target === 'gapjunctions') {
                        // Always refresh GJ tab — user may have added/removed GJs
                        renderGJTab();
                    }
                    if (target === 'connectivity' && !_tabRendered.connectivity) {
                        renderConnectivityMatrix();
                        _tabRendered.connectivity = true;
                    }
                    if (target === 'circuit') renderCircuitModel();
                    if (target === 'mc') renderMCModel();
                }
            });
        });

        // ── Heatmap helper: build B/W table HTML ──────────────────
        function _buildHeatTable(dataMap, names, effectiveMax) {
            let html = '<table><thead><tr><th class="corner"></th>';
            names.forEach(n => { html += '<th>' + n.replace('_', '<br>') + '</th>'; });
            html += '</tr></thead><tbody>';
            names.forEach(src => {
                html += '<tr><th class="row-header" style="text-align:right;">' + src + '</th>';
                names.forEach(tgt => {
                    if (src === tgt) { html += '<td class="diagonal">\u2014</td>'; return; }
                    const val = dataMap[src + '|' + tgt];
                    if (val === undefined || val <= 0) {
                        html += '<td class="no-data" data-src="' + src + '" data-tgt="' + tgt + '">-</td>';
                        return;
                    }
                    const frac = Math.min(1, val / effectiveMax);
                    const v = Math.round(255 * frac);  // 0=black → 255=white
                    const fg = frac > 0.45 ? '#000' : '#ccc';
                    const style = 'background:rgb(' + v + ',' + v + ',' + v + ');color:' + fg + ';';
                    html += '<td style="' + style + '" data-src="' + src + '" data-tgt="' + tgt
                        + '" title="' + src + ' \u2192 ' + tgt + ': ' + val.toFixed(3)
                        + ' \u00b5m\u00b2">' + val.toFixed(2) + '</td>';
                });
                html += '</tr>';
            });
            html += '</tbody></table>';
            return html;
        }

        // ── Heatmap helper: attach click handlers ────────────────
        function _attachHeatClicks(container) {
            container.querySelectorAll('td[data-src]').forEach(td => {
                td.addEventListener('click', () => {
                    const src = td.dataset.src, tgt = td.dataset.tgt;
                    matrixModal.classList.remove('active');
                    const ov = overlapList.find(
                        o => (o.source === src && o.target === tgt)
                          || (o.source === tgt && o.target === src));
                    if (ov) { selectItem('overlap', ov.idx, ov.x, ov.y, ov.z, ov.source, ov.target); return; }
                    const c = contactList.find(c => c.source === src && c.target === tgt);
                    if (c) selectItem('contact', c.idx, c.x, c.y, c.z, c.source, c.target);
                });
            });
        }

        // ── Heatmap helper: derive pair name without L/R suffix ──
        function _pairBase(name) {
            // e.g. "VS1_L" → "VS1", "HSN_R" → "HSN", "BIPS_L" → "BIPS"
            return name.replace(/_[LR]$/, '');
        }

        function renderHeatmap() {
            // ─────────────────────────────────────────────────────────────────
            // OVERLAP AREA HEATMAP  (4 panels)
            // Panel 1 — Full 22×22 matrix: every neuron pair, raw area (µm²).
            // Panel 2 — L/R pair mean: base names (e.g. VS1), L and R averaged.
            // Panel 3 — Group mean: functional groups (VS, HS, MOT, MOS, BIPS).
            // Panel 4 — Bidir mean: MOS & MOT rows only, mean(A→B, B→A).
            // Each panel has its own intensity slider; moving the slider re-renders
            // only that panel's table (not the other three).
            // ─────────────────────────────────────────────────────────────────
            // Build full lookup
            const lookup = {};
            overlapTable.forEach(row => { lookup[row.source + '|' + row.target] = row.area; });
            const allNames = new Set();
            overlapTable.forEach(r => { allNames.add(r.source); allNames.add(r.target); });
            CELL_NAMES.forEach(n => allNames.add(n));
            const names = Array.from(allNames).sort();
            let globalMax = 0;
            overlapTable.forEach(r => { if (r.area > globalMax) globalMax = r.area; });
            if (globalMax === 0) globalMax = 1;

            // ── Panel 1: Full matrix ────────────────────────────
            const heatmapDiv = document.getElementById('heatmapContainer');
            const sliderFull = document.getElementById('heatSliderFull');
            const sliderFullVal = document.getElementById('heatSliderFullVal');

            function renderFull() {
                const pct = parseFloat(sliderFull.value);
                const effMax = globalMax * (pct / 100);
                sliderFullVal.textContent = 'max: ' + effMax.toFixed(2) + ' \u00b5m\u00b2';
                heatmapDiv.innerHTML = _buildHeatTable(lookup, names, effMax);
                _attachHeatClicks(heatmapDiv);
            }
            // Use .oninput assignment (not addEventListener) so that re-opening
            // the modal replaces the handler rather than stacking a new copy.
            sliderFull.oninput = renderFull;
            renderFull();

            // ── Panel 2: L/R pair mean ──────────────────────────
            // For each pair (e.g. MOS_L, MOT_R) compute mean of (A_L→B_R, A_R→B_L)
            // i.e. the pair base names, keeping _L and _R distinction
            const pairLookup = {};
            const pairNames = new Set();
            // Group neuron names by base: {VS1: [VS1_L, VS1_R], ...}
            const baseGroups = {};
            names.forEach(n => {
                const b = _pairBase(n);
                if (!baseGroups[b]) baseGroups[b] = [];
                baseGroups[b].push(n);
            });
            const baseNames = Object.keys(baseGroups).sort();
            // For each base pair (src_base, tgt_base) where src != tgt,
            // average over all L/R combinations
            baseNames.forEach(sb => {
                baseNames.forEach(tb => {
                    if (sb === tb) return;
                    const srcNeurons = baseGroups[sb];
                    const tgtNeurons = baseGroups[tb];
                    let sum = 0, cnt = 0;
                    srcNeurons.forEach(s => {
                        tgtNeurons.forEach(t => {
                            const v = lookup[s + '|' + t];
                            if (v !== undefined && v > 0) { sum += v; cnt++; }
                        });
                    });
                    if (cnt > 0) pairLookup[sb + '|' + tb] = sum / cnt;
                });
            });
            baseNames.forEach(n => pairNames.add(n));
            let pairMax = 0;
            Object.values(pairLookup).forEach(v => { if (v > pairMax) pairMax = v; });
            if (pairMax === 0) pairMax = 1;

            const heatPairDiv = document.getElementById('heatmapPairContainer');
            const sliderPair = document.getElementById('heatSliderPair');
            const sliderPairVal = document.getElementById('heatSliderPairVal');
            const sortedBaseNames = Array.from(pairNames).sort();

            function renderPair() {
                const pct = parseFloat(sliderPair.value);
                const effMax = pairMax * (pct / 100);
                sliderPairVal.textContent = 'max: ' + effMax.toFixed(2) + ' \u00b5m\u00b2';
                heatPairDiv.innerHTML = _buildHeatTable(pairLookup, sortedBaseNames, effMax);
                _attachHeatClicks(heatPairDiv);
            }
            sliderPair.oninput = renderPair;
            renderPair();

            // ── Panel 3: Group mean (L+R collapsed) ─────────────
            // Group neurons by functional group (MOT, MOS, VS1..VS4, HSN/HSE/HSS, BIPS, H2)
            // Then average the pair means
            function _groupName(baseName) {
                // VS1→VS, VS2→VS, etc. ; HSN→HS, HSE→HS, HSS→HS
                if (baseName.match(/^VS\d/)) return 'VS';
                if (baseName.match(/^HS[NES]/)) return 'HS';
                return baseName;  // MOT, MOS, BIPS, H2
            }
            const groupLookup = {};
            const groupSet = new Set();
            const groupBases = {};  // group → [base names]
            baseNames.forEach(b => {
                const g = _groupName(b);
                groupSet.add(g);
                if (!groupBases[g]) groupBases[g] = [];
                groupBases[g].push(b);
            });
            const groupNames = Array.from(groupSet).sort();
            groupNames.forEach(sg => {
                groupNames.forEach(tg => {
                    if (sg === tg) return;
                    const srcBases = groupBases[sg];
                    const tgtBases = groupBases[tg];
                    let sum = 0, cnt = 0;
                    srcBases.forEach(sb => {
                        tgtBases.forEach(tb => {
                            const v = pairLookup[sb + '|' + tb];
                            if (v !== undefined && v > 0) { sum += v; cnt++; }
                        });
                    });
                    if (cnt > 0) groupLookup[sg + '|' + tg] = sum / cnt;
                });
            });
            let groupMax = 0;
            Object.values(groupLookup).forEach(v => { if (v > groupMax) groupMax = v; });
            if (groupMax === 0) groupMax = 1;

            const heatGroupDiv = document.getElementById('heatmapGroupContainer');
            const sliderGroup = document.getElementById('heatSliderGroup');
            const sliderGroupVal = document.getElementById('heatSliderGroupVal');

            function renderGroup() {
                const pct = parseFloat(sliderGroup.value);
                const effMax = groupMax * (pct / 100);
                sliderGroupVal.textContent = 'max: ' + effMax.toFixed(2) + ' \u00b5m\u00b2';
                heatGroupDiv.innerHTML = _buildHeatTable(groupLookup, groupNames, effMax);
            }
            sliderGroup.oninput = renderGroup;
            renderGroup();

            // ── Panel 4: MOS/MOT bidirectional mean ─────────────
            // For each group target, compute mean(A→B, B→A) and show only MOS and MOT rows
            const bidirRows = ['MOS', 'MOT'];
            const bidirCols = groupNames.filter(g => g !== 'MOS' && g !== 'MOT');
            const bidirLookup = {};
            bidirRows.forEach(src => {
                bidirCols.forEach(tgt => {
                    const ab = groupLookup[src + '|' + tgt] || 0;
                    const ba = groupLookup[tgt + '|' + src] || 0;
                    const vals = [ab, ba].filter(v => v > 0);
                    if (vals.length > 0) {
                        bidirLookup[src + '|' + tgt] = vals.reduce((a, b) => a + b, 0) / vals.length;
                    }
                });
            });
            let bidirMax = 0;
            Object.values(bidirLookup).forEach(v => { if (v > bidirMax) bidirMax = v; });
            if (bidirMax === 0) bidirMax = 1;

            const heatBidirDiv = document.getElementById('heatmapBidirContainer');
            const sliderBidir = document.getElementById('heatSliderBidir');
            const sliderBidirVal = document.getElementById('heatSliderBidirVal');

            function _buildBidirTable(dataMap, rowNames, colNames, effectiveMax) {
                let html = '<table><thead><tr><th class="corner"></th>';
                colNames.forEach(n => { html += '<th>' + n + '</th>'; });
                html += '</tr></thead><tbody>';
                rowNames.forEach(src => {
                    html += '<tr><th class="row-header" style="text-align:right;">' + src + '</th>';
                    colNames.forEach(tgt => {
                        const val = dataMap[src + '|' + tgt];
                        if (val === undefined || val <= 0) {
                            html += '<td class="no-data" data-src="' + src + '" data-tgt="' + tgt + '">-</td>';
                            return;
                        }
                        const frac = Math.min(1, val / effectiveMax);
                        const v = Math.round(255 * frac);
                        const fg = frac > 0.45 ? '#000' : '#ccc';
                        const style = 'background:rgb(' + v + ',' + v + ',' + v + ');color:' + fg + ';';
                        html += '<td style="' + style + '" data-src="' + src + '" data-tgt="' + tgt
                            + '" title="' + src + ' \u2194 ' + tgt + ' (mean): ' + val.toFixed(3)
                            + ' \u00b5m\u00b2">' + val.toFixed(2) + '</td>';
                    });
                    html += '</tr>';
                });
                html += '</tbody></table>';
                return html;
            }

            function renderBidir() {
                const pct = parseFloat(sliderBidir.value);
                const effMax = bidirMax * (pct / 100);
                sliderBidirVal.textContent = 'max: ' + effMax.toFixed(2) + ' \u00b5m\u00b2';
                heatBidirDiv.innerHTML = _buildBidirTable(bidirLookup, bidirRows, bidirCols, effMax);
            }
            sliderBidir.oninput = renderBidir;
            renderBidir();
        }

        function renderGJTab() {
            if (gapJunctions.length === 0) {
                gjContainer.innerHTML = '<p style="color:#888;font-size:12px;">No putative gap junctions marked yet.<br>'
                    + 'Select an overlap or contact, navigate to the location, then click \u26a1 Putative Gap-Junc.</p>';
                return;
            }
            // Group by undirected pair
            const pairMap = {};
            gapJunctions.forEach((gj, i) => {
                const pairKey = [gj.source, gj.target].sort().join(' \u2194 ');
                if (!pairMap[pairKey]) pairMap[pairKey] = [];
                pairMap[pairKey].push({ ...gj, _i: i });
            });
            let html = '<p style="color:#ccc;font-size:11px;margin:0 0 8px;">'
                + gapJunctions.length + ' putative gap junction(s) across '
                + Object.keys(pairMap).length + ' pair(s)</p>';
            html += '<table class="gj-table"><thead><tr>'
                + '<th>#</th><th>Pair</th><th>X (nm)</th><th>Y (nm)</th><th>Z (nm)</th>'
                + '<th>Type</th><th>Idx</th><th>Z-off</th><th></th>'
                + '</tr></thead><tbody>';
            gapJunctions.forEach((gj, i) => {
                html += '<tr class="gj-row" data-gji="' + i + '" style="cursor:pointer;" title="Click to navigate to this GJ">'
                    + '<td>' + (i + 1) + '</td>'
                    + '<td>' + gj.source + ' \u2194 ' + gj.target + '</td>'
                    + '<td>' + Math.round(gj.x) + '</td>'
                    + '<td>' + Math.round(gj.y) + '</td>'
                    + '<td>' + Math.round(gj.z) + '</td>'
                    + '<td>' + gj.kind + '</td>'
                    + '<td>' + gj.idx + '</td>'
                    + '<td>' + gj.zOffset + '</td>'
                    + '<td class="gj-delete" data-gji="' + i + '" title="Remove this gap junction">\u2716</td>'
                    + '</tr>';
            });
            html += '</tbody></table>';

            // Pair summary
            html += '<h4 style="color:#39FF14;margin:12px 0 6px;">Pairs with Gap Junctions</h4>';
            html += '<table class="gj-table"><thead><tr>'
                + '<th>Pair</th><th>Count</th><th>Locations</th>'
                + '</tr></thead><tbody>';
            Object.keys(pairMap).sort().forEach(pair => {
                const entries = pairMap[pair];
                const locs = entries.map(g =>
                    '(' + Math.round(g.x) + ', ' + Math.round(g.y) + ', ' + Math.round(g.z) + ')'
                ).join(', ');
                html += '<tr><td>' + pair + '</td><td>' + entries.length + '</td>'
                    + '<td style="font-size:9px;">' + locs + '</td></tr>';
            });
            html += '</tbody></table>';

            gjContainer.innerHTML = html;
            // Wire up row clicks — navigate to GJ location
            gjContainer.querySelectorAll('.gj-row').forEach(row => {
                row.addEventListener('click', function(e) {
                    if (e.target.classList.contains('gj-delete')) return; // let delete handle it
                    const idx = parseInt(this.dataset.gji);
                    const gj = gapJunctions[idx];
                    if (gj) {
                        matrixModal.classList.remove('active');
                        navigateToGJ(gj);
                    }
                });
            });
            // Wire up delete buttons
            gjContainer.querySelectorAll('.gj-delete').forEach(btn => {
                btn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    const idx = parseInt(this.dataset.gji);
                    gapJunctions.splice(idx, 1);
                    updateGJTrace();
                    renderGJTab();
                });
            });
        }

        // ── Connectivity matrix (GJ + Chemical Synapses) ───────────
        function renderConnectivityMatrix() {
            // Collect all neuron names from overlaps + synapses + gap junctions + RAW_COUNTS
            const allNames = new Set();
            overlapTable.forEach(r => { allNames.add(r.source); allNames.add(r.target); });
            synapseList.forEach(s => { allNames.add(s.source); allNames.add(s.target); });
            gapJunctions.forEach(gj => { allNames.add(gj.source); allNames.add(gj.target); });
            CELL_NAMES.forEach(n => allNames.add(n));
            const names = Array.from(allNames).sort();

            // Build chemical synapse count from RAW_COUNTS (curated mat783 cleft>=50)
            const chemCount = {};
            for (let pi = 0; pi < N_CELLS; pi++) {
                for (let qi = 0; qi < N_CELLS; qi++) {
                    const cnt = RAW_COUNTS[pi][qi];
                    if (cnt > 0) {
                        const key = CELL_NAMES[pi] + '|' + CELL_NAMES[qi];
                        chemCount[key] = cnt;
                    }
                }
            }

            // Build GJ count: undirected pair
            const gjCount = {};
            gapJunctions.forEach(gj => {
                const a = gj.source, b = gj.target;
                const k1 = a + '|' + b, k2 = b + '|' + a;
                gjCount[k1] = (gjCount[k1] || 0) + 1;
                gjCount[k2] = (gjCount[k2] || 0) + 1;
            });

            let html = '<p style="color:#aaa;font-size:10px;margin:0 0 6px;">'
                + 'Rows \u2192 Columns (mat783, cleft\u226550). '
                + '<span style="color:#39FF14;">\u25cf GJ</span> &nbsp; '
                + '<span style="color:#FFD700;">\u25cf Excitatory (ACh/Glut)</span> &nbsp; '
                + '<span style="color:#64b5f6;">\u25cf Inhibitory (GABA)</span> &nbsp; '
                + '<span style="color:#00BFFF;">\u25cf GJ+Syn</span></p>';
            html += '<table><thead><tr><th class="corner"></th>';
            names.forEach(n => { html += '<th>' + n.replace('_', '<br>') + '</th>'; });
            html += '</tr></thead><tbody>';
            names.forEach(src => {
                html += '<tr><th class="row-header" style="text-align:right;">' + src + '</th>';
                names.forEach(tgt => {
                    if (src === tgt) { html += '<td class="diagonal">\u2014</td>'; return; }
                    const key = src + '|' + tgt;
                    const nChem = chemCount[key] || 0;
                    const nGJ = gjCount[key] || 0;
                    if (nChem === 0 && nGJ === 0) {
                        html += '<td class="no-data">-</td>';
                        return;
                    }
                    // Determine NT type for color-coding
                    const pi = CI[src], qi = CI[tgt];
                    const isInh = (pi !== undefined && qi !== undefined && SYN_ESYN[pi][qi] < -10);
                    let bg, fg;
                    if (nGJ > 0 && nChem > 0) { bg = '#00BFFF'; fg = '#000'; }
                    else if (nGJ > 0)          { bg = '#39FF14'; fg = '#000'; }
                    else if (isInh)            { bg = '#64b5f6'; fg = '#000'; }
                    else                       { bg = '#FFD700'; fg = '#000'; }
                    const parts = [];
                    if (nGJ > 0) parts.push(nGJ + ' GJ');
                    if (nChem > 0) parts.push(nChem + (isInh ? ' inh' : ' exc'));
                    const cellText = parts.join('+');
                    const ntLabel = isInh ? ' (GABA)' : (nChem > 0 ? ' (ACh/Glut)' : '');
                    const tip = src + ' \u2192 ' + tgt + ': ' + parts.join(', ') + ntLabel;
                    html += '<td style="background:' + bg + ';color:' + fg
                        + ';font-size:9px;font-weight:bold;" title="' + tip + '">'
                        + cellText + '</td>';
                });
                html += '</tr>';
            });
            html += '</tbody></table>';
            connectivityContainer.innerHTML = html;
        }

        // ── Cell names & index mapping (shared by Tier 1 + Tier 2) ──
        const CELL_NAMES = [
            'MOT_L','MOT_R','MOS_L','MOS_R',
            'VS1_L','VS1_R','VS2_L','VS2_R',
            'VS3_L','VS3_R','VS4_L','VS4_R',
            'HSN_L','HSN_R','HSE_L','HSE_R',
            'HSS_L','HSS_R',
            'BIPS_L','BIPS_R','H2_L','H2_R'
        ];
        const N_CELLS = CELL_NAMES.length;
        const CI = {}; CELL_NAMES.forEach((n,i) => { CI[n] = i; });
        const SPIKING = new Set(['MOT_L','MOT_R','MOS_L','MOS_R']);

        // ── RAW_COUNTS: rows=pre, cols=post  [from synapses.csv] ──
        const RAW_COUNTS = [
         // MOT_L MOT_R MOS_L MOS_R VS1_L VS1_R VS2_L VS2_R VS3_L VS3_R VS4_L VS4_R HSN_L HSN_R HSE_L HSE_R HSS_L HSS_R BIPS_L BIPS_R H2_L H2_R
            [0,    0,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOT_L
            [0,    0,    0,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOT_R
            [3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOS_L
            [0,    6,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOS_R
            [0,    0,    0,    0,    0,    0,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS1_L
            [0,    0,    0,    0,    0,    0,    0,    4,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS1_R
            [0,    0,    6,    0,    1,    0,    0,    0,    4,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS2_L
            [0,    0,    0,    0,    0,    6,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS2_R
            [0,    0,    6,    0,    0,    0,   12,    0,    0,    0,    2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS3_L
            [0,    0,    0,    5,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS3_R
            [0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS4_L
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    0,    0,    0,    0,    0,    0],  // VS4_R
            [2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    8,    0,    0],  // HSN_L
            [0,    9,    0,    4,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    7,    0,    3,   29,    4,    0,    0],  // HSN_R
            [4,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    5,    0,    0,    0,    0,    0,    2,   45,    0,    0],  // HSE_L
            [0,    6,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    2,    0,    0,    0,    0,   63,   12,    0,    0],  // HSE_R
            [4,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    0,    0,    0,    0,    0,    0,   41,    0,    0],  // HSS_L
            [0,    3,    0,    2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   37,    0,    0,    0],  // HSS_R
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,   23,    0,    8,    0,    0,    0,    0,    0,    0,    0],  // BIPS_L
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    4,    0,   15,    1,    0,    1,    0,    0,    0],  // BIPS_R
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // H2_L
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // H2_R
        ];

        // ── SYN_ESYN: per-connection reversal potential (mV) ──
        // FlyWire NT predictions: ACh/Glut → excitatory (0 mV), GABA → inhibitory (-80 mV)
        // VS chain fwd (VS1→VS2, VS2→VS3, VS3→VS4) = GABA(-80)
        // VS chain back (VS3→VS2, VS2→VS1, VS4→VS2) = ACh(0)
        // HS→MN, VS→MOS, HS→BIPS, HS chain, MN↔MN = excitatory(0)
        // BIPS→HS, BIPS→BIPS = GABA(-80) (GABAergic interneurons)
        const E_EXC = 0, E_INH = -80;
        const SYN_ESYN = [
         // MOT_L MOT_R MOS_L MOS_R VS1_L VS1_R VS2_L VS2_R VS3_L VS3_R VS4_L VS4_R HSN_L HSN_R HSE_L HSE_R HSS_L HSS_R BIPS_L BIPS_R H2_L H2_R
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // MOT_L
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // MOT_R
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // MOS_L
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // MOS_R
            [0,    0,    0,    0,    0,    0,  -80,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // VS1_L  fwd→VS2
            [0,    0,    0,    0,    0,    0,    0,  -80,    0,  -80,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // VS1_R  fwd→VS2,VS3
            [0,    0,    0,    0,    0,    0,    0,    0,  -80,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // VS2_L  fwd→VS3
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // VS2_R
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  -80,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // VS3_L  fwd→VS4
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // VS3_R
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // VS4_L
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // VS4_R
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // HSN_L
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // HSN_R
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // HSE_L
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // HSE_R
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // HSS_L
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // HSS_R
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  -80,    0,  -80,    0,    0,    0,    0,    0,    0,    0  ],  // BIPS_L  →HSN,HSE GABA
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  -80,    0,  -80,  -80,    0,  -80,    0,    0,    0  ],  // BIPS_R  →HSN,HSE,HSS,BIPS GABA
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // H2_L
            [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],  // H2_R
        ];

        // ── Tier 1 Circuit Model ────────────────────────────────────
        let circuitInitialized = false;
        // ═══════════════════════════════════════════════════════════════════════════
        // TIER 1 BIOPHYSICAL CIRCUIT MODEL — FULL DOCUMENTATION
        // ═══════════════════════════════════════════════════════════════════════════
        //
        // OVERVIEW
        // --------
        // A single-compartment conductance-based circuit model of the first-order
        // visual motion pathway in Drosophila, focusing on the LPTC (Lobula Plate
        // Tangential Cell) → motor neuron (MN) circuit.
        //
        // The model contains 22 neurons (left + right):
        //   • VS1-VS4  (×2 sides = 8):  Vertical System LPTCs, wide-field optic flow detectors
        //   • HSN, HSE, HSS (×2 = 6):   Horizontal System LPTCs, horizontal optic flow
        //   • MOS_L/R  (2):              Neck motor neurons for orbit/smooth eye movements
        //   • MOT_L/R  (2):              Neck motor neurons for compensatory head/eye rotation
        //   • BIPS_L/R (2):              Bistratified inhibitory neurons (GABAergic, project onto HS)
        //   • H2_L/R   (2):              H2 horizontal motion detectors (currently passive)
        //
        // CELL TYPES
        // ----------
        // LPTC (VS, HS):
        //   - Graded (non-spiking) neurons. Wide-field optic flow sensors.
        //   - Respond to retinal slip with slow, sustained depolarisation.
        //   - Model: leak + delayed-rectifier K⁺ + T-type Ca²⁺ (optional)
        //   - Resting potential set by VL (leak reversal); Rin sets gain.
        //   - VS cells are electrically coupled in a chain (VS1↔VS2↔VS3↔VS4)
        //     via gap junctions, enabling spatial integration along the dorso-ventral axis.
        //   - HS cells (HSN↔HSE↔HSS) similarly coupled.
        //
        // MN (MOT, MOS):
        //   - Spiking (Hodgkin-Huxley) neurons. Motor command neurons.
        //   - Fire tonically at rest (MOT: ~120 Hz, MOS: ~100 Hz).
        //   - Primary driver of spontaneous firing: persistent Na⁺ (gNaP).
        //     Higher gNaP → higher tonic rate → stronger motor output at rest.
        //   - Receive inputs from LPTCs via (a) gap junctions (LPTC axon ↔ MN dendrite)
        //     and (b) graded chemical synapses.
        //   - VS→MOS and HS→MOS/MOT gap junctions (bidirectional).
        //     NOTE: VS does NOT electrically couple to MOT; only HS→MOT.
        //   - MN↔MN chemical synapses (MOT_L→MOS_L: excitatory, 3 contacts each).
        //
        // BIOPHYSICAL PARAMETERS
        // ----------------------
        // Shared constants (physiological):
        //   Cm  = 1.0 nF     membrane capacitance (whole-cell patch, Drosophila MN scale)
        //   V_Na = +50 mV    Na⁺ reversal potential
        //   V_K  = -77 mV    K⁺ reversal potential
        //   VCa  = +120 mV   Ca²⁺ reversal potential (T-type, low-threshold)
        //   dt   = 0.01 ms   integration step (forward Euler, stable for HH at this dt)
        //
        // Calibrated MN defaults (from auto-calibration against patch-clamp targets):
        //   MOT: VL=-58.81 mV, gNaP=1.748 nS, gK=52.68 nS, Ibias=2.84 nA
        //   MOS: VL=-62.76 mV, gNaP=1.510 nS, gK=45.60 nS, Ibias=2.10 nA
        //   gNa=145/170 nS, gL=0.42/0.28 nS, Rin=260/360 MΩ for MOT/MOS respectively
        //
        // INTEGRATION METHOD
        // ------------------
        // Forward Euler (explicit), dt = 0.01 ms.
        // Each time step:
        //   1. Compute gap junction currents (bidirectional, LP-filtered)
        //   2. Compute graded or alpha-function chemical synapse currents
        //   3. Add external stimulus + tonic Ibias + white noise
        //   4. Step each cell's gating variables (m, h, n) with explicit Euler
        //   5. Update membrane voltage: V += (Itot - I_channels) / Cm * dt
        //
        // A 500 ms "pre-roll" (simPreRollMs) runs before t=0 so the circuit settles
        // to its steady firing rate before the stimulus is applied.
        // ═══════════════════════════════════════════════════════════════════════════

        function renderCircuitModel() {
            if (circuitInitialized) return;
            circuitInitialized = true;

            const dt = 0.01;  // Integration step (ms). Forward Euler is stable for HH at dt ≤ 0.025 ms.
            const Cm = 1.0;   // Whole-cell membrane capacitance (nF). Shared by all cell types.
            // Ionic reversal potentials (mV), based on Drosophila ionic concentrations.
            const VCa = 120, V_Na = 50, V_K = -77;

            // ═════════════════════════════════════════════════════════════════════
            // HODGKIN-HUXLEY GATE KINETICS  (Hodgkin & Huxley, 1952)
            // ═════════════════════════════════════════════════════════════════════
            // The three gating variables control ion channel conductances:
            //
            //   m(t) : Na⁺ activation gate        — fast, activates at depolarisation
            //   h(t) : Na⁺ inactivation gate       — slow, inactivates after activation
            //   n(t) : K⁺ (delayed-rectifier) gate — repolarises the action potential
            //
            // Each gate follows: dx/dt = αx(V)·(1−x) − βx(V)·x
            // Steady state: x∞(V) = α / (α + β)     [computed by ss()]
            // Time constant: τx(V) = 1 / (α + β)
            //
            // Na⁺ channel: I_Na = gNa · m³ · h · (V − V_Na)
            //   m³ factor: cubic activation reflects 3 independent activation subunits
            //   h factor:  inactivation — Na⁺ channel closes after ~1 ms; critical for spike
            //
            // K⁺ channel: I_K = gK · n⁴ · (V − V_K)
            //   n⁴ factor: four activation subunits in series (delayed-rectifier Kv)

            // αm: Na⁺ activation opening rate. L'Hôpital rule at V=-40 prevents 0/0.
            function alphaM(V) { const x = V+40; return Math.abs(x)<1e-7 ? 1 : 0.1*x/(1-Math.exp(-x/10)); }
            // βm: Na⁺ activation closing rate.
            function betaM(V)  { return 4*Math.exp(-(V+65)/18); }
            // αh: Na⁺ inactivation opening rate (slow exponential).
            function alphaH(V) { return 0.07*Math.exp(-(V+65)/20); }
            // βh: Na⁺ inactivation closing rate (sigmoidal, closes at plateau).
            function betaH(V)  { return 1/(1+Math.exp(-(V+35)/10)); }
            // αn: K⁺ activation opening rate. L'Hôpital rule at V=-55.
            function alphaN(V) { const x = V+55; return Math.abs(x)<1e-7 ? 0.1 : 0.01*x/(1-Math.exp(-x/10)); }
            // βn: K⁺ activation closing rate.
            function betaN(V)  { return 0.125*Math.exp(-(V+65)/80); }
            // ss(): steady-state gate value x∞(V) = α/(α+β). Used at initialisation.
            function ss(af,bf,V) { const a=af(V),b=bf(V); return a/(a+b); }

            // ═════════════════════════════════════════════════════════════════════
            // T-TYPE Ca²⁺ CHANNEL  (low-voltage activated, LVA)
            // ═════════════════════════════════════════════════════════════════════
            // Present in both LPTCs and (optionally) MNs. Primarily relevant for
            // dendritic integration in VS/HS; in MNs keep gVT=0 to avoid
            // burst-then-silence artefacts from Ca²⁺-mediated K⁺ currents.
            //
            // I_T = gVT · m∞_Ca(V)³ · h_Ca(t) · (V − VCa)
            //
            // m∞_Ca: instantaneous activation (fast, no state variable needed)
            //   Half-activation at V½ = -61 mV, slope k = 4.2 mV
            // h_Ca: slow inactivation (tracks h∞_Ca with time constant tauHCa(V))
            //   Half-inactivation at V½ = -85.5 mV (deeply hyperpolarised → de-inactivated)
            //   tauHCa(V): voltage-dependent time constant, slow (40–70 ms range)

            // m∞_Ca: T-Ca²⁺ steady-state activation. Boltzmann sigmoid.
            function mInfCa(V)  { return 1/(1+Math.exp((-61-V)/4.2)); }
            // h∞_Ca: T-Ca²⁺ steady-state inactivation. Deeply negative V→ fully de-inactivated.
            function hInfCa(V)  { return 1/(1+Math.exp((V+85.5)/8.6)); }
            // τh_Ca(V): voltage-dependent inactivation time constant (ms). Peak ~70 ms near -84 mV.
            function tauHCa(V)  { return 40+30/(1+Math.exp((V+84)/7.3))*Math.exp((V+160)/30); }

            // ═════════════════════════════════════════════════════════════════════
            // PERSISTENT Na⁺ (NaP) — PRIMARY DRIVER OF TONIC SPIKING IN MNs
            // ═════════════════════════════════════════════════════════════════════
            // I_NaP = gNaP · m∞_NaP(V) · (V − V_Na)
            //
            // NaP does not inactivate (persistent). It activates at sub-threshold
            // potentials (~−55 to −45 mV), providing a depolarising current that
            // sustains tonic firing without a falling transient Na⁺ current.
            //
            // Tuning gNaP is the primary lever for setting tonic firing rate:
            //   • Increase gNaP → higher spontaneous rate (more depolarising drive)
            //   • Decrease gNaP → lower rate or even tonic silence
            // This is why MOS vs MOT can be tuned to different baseline rates:
            //   MOS typically has slightly lower gNaP than MOT in the calibrated preset.
            //
            // m∞_NaP: half-activation at −52 mV, slope 5 mV (steep, near-threshold).
            function mNaPinf(V) { return 1/(1+Math.exp(-(V+52)/5)); }

            // ═════════════════════════════════════════════════════════════════════
            // LPTC CELL MODEL  (graded, non-spiking: VS1-VS4, HSN, HSE, HSS)
            // ═════════════════════════════════════════════════════════════════════
            // Single-compartment conductance model. No Na⁺ spike; responds to synaptic
            // inputs with slow, graded voltage changes that are transmitted to MNs.
            //
            // Membrane equation:
            //   Cm · dV/dt = −I_T − I_K − I_L + I_input
            //
            //   I_T = gVT · m∞_Ca³ · h_Ca · (V − VCa)    [T-type Ca²⁺, optional]
            //   I_K = gK  · n⁴      · (V − V_K)            [delayed-rectifier K⁺]
            //   I_L = (gL + 1/Rin) · (V − VL)              [leak + resting conductance]
            //
            // The 1/Rin term in I_L represents the standing membrane conductance that
            // fixes the resting potential at VL when no input is present.
            // Larger Rin → higher voltage gain per unit input current.
            //
            // VS cell diversity: VS1-VS4 differ in VL and Rin by a user-set per-cell step.
            //   VS1: VL = Vr1, Rin = Rin1
            //   VS2: VL = Vr1 + ΔVrStep, Rin = Rin1 + ΔRinStep   (typically more hyperpolarised)
            //   VS3: VL = Vr1 + 2·ΔVrStep, etc.
            // This gradient creates different gain/offset profiles matching the known
            // visual tuning gradient along the VS1–VS4 array.
            //
            // State variables: V (membrane potential), hCa (T-Ca inactivation), n (K⁺ activation)
            // All other gate variables (mNa, hNa) are stored but not used in stepLPTC.

            // Create a new LPTC cell object with all state variables at steady state.
            function createLPTC(name, Rin, VL, gVT, gL, gK) {
                return {
                    name, type: 'LPTC',
                    V: VL,          // Start at rest (leak reversal)
                    gVT, gL, Rin, gK, VL,
                    hCa: hInfCa(VL),                    // T-Ca inactivation at resting V
                    mNa: ss(alphaM,betaM,VL),            // (unused in LPTC step, stored for completeness)
                    hNa: ss(alphaH,betaH,VL),
                    n:   ss(alphaN,betaN,VL),            // K⁺ gate at resting V
                };
            }
            // Advance LPTC one time step (forward Euler).
            function stepLPTC(c, Itot) {
                const v = c.V;
                // T-type Ca²⁺ inactivation: slow approach to h∞_Ca with τh_Ca(v)
                c.hCa += (hInfCa(v)-c.hCa)/tauHCa(v)*dt;
                // T-Ca²⁺ current (inward, depolarising if V < VCa = +120 mV)
                const iT = c.gVT * Math.pow(mInfCa(v),3) * c.hCa * (v-VCa);
                // K⁺ delayed rectifier gate update
                c.n += (alphaN(v)*(1-c.n)-betaN(v)*c.n)*dt;
                // K⁺ current (outward, repolarising)
                const iK = c.gK * Math.pow(c.n,4) * (v-V_K);
                // Leak + standing conductance (holds V near VL at rest)
                const iL = (c.gL + 1/c.Rin) * (v - c.VL);
                // Membrane voltage update: Cm·dV/dt = −iT − iK − iL + Itot
                c.V = v + (-iT - iK - iL + Itot)/Cm*dt;
                return c.V;
            }

            // ═════════════════════════════════════════════════════════════════════
            // MN CELL MODEL  (Hodgkin-Huxley spiking: MOT, MOS)
            // ═════════════════════════════════════════════════════════════════════
            // Full Hodgkin-Huxley model extended with persistent Na⁺ (NaP) and
            // optional T-type Ca²⁺. Fires action potentials tonically at rest.
            //
            // Membrane equation:
            //   Cm · dV/dt = −I_T − I_Na − I_K − I_NaP − I_L + I_input
            //
            //   I_Na  = gNa  · m³ · h       · (V − V_Na)   [transient Na⁺, spike upstroke]
            //   I_K   = gK   · n⁴            · (V − V_K)    [delayed rectifier K⁺, repolarisation]
            //   I_NaP = gNaP · m∞_NaP(V)    · (V − V_Na)   [persistent Na⁺, tonic firing driver]
            //   I_T   = gVT  · m∞_Ca³ · h_Ca · (V − VCa)   [T-Ca²⁺, keep gVT=0 to avoid bursting]
            //   I_L   = (gL + 1/Rin)         · (V − VL)     [leak]
            //
            // Tonic firing mechanism:
            //   At rest (V ~ -60 to -55 mV), I_NaP provides sustained inward current
            //   that slowly depolarises the cell until the Na⁺ spike threshold (~-40 mV)
            //   is reached. After each spike, K⁺ repolarisation resets V, and the cycle
            //   repeats. The inter-spike interval (ISI) is primarily set by:
            //     1. gNaP  (more NaP → shorter ISI → higher rate)
            //     2. VL    (more depolarised VL → shorter ISI)
            //     3. Ibias (additional tonic bias current)
            //     4. gK    (stronger K⁺ repolarisation → longer AHP → lower rate)
            //
            // Auto-calibration targets:  MOT ~ 120 Hz,  MOS ~ 100 Hz
            // The 'Auto-Calibrate' button adjusts gNaP (primary) then Ibias (secondary)
            // to hit these targets via a fast bisection search over the pre-roll window.
            //
            // State variables: V, hCa, mNa (Na⁺ m), hNa (Na⁺ h), n (K⁺)

            // Create a new MN cell object initialised to steady state.
            function createMN(name, Rin, VL, gVT, gL, gNa, gK, gNaP) {
                return {
                    name, type: 'MN',
                    V: VL,          // Start at resting potential
                    gVT, gL, Rin, gNa, gK, VL, gNaP,
                    hCa: hInfCa(VL),                    // T-Ca inactivation at rest
                    mNa: ss(alphaM,betaM,VL),            // Na⁺ activation gate at rest
                    hNa: ss(alphaH,betaH,VL),            // Na⁺ inactivation gate at rest
                    n:   ss(alphaN,betaN,VL),            // K⁺ gate at rest
                };
            }
            // Advance MN one time step (forward Euler).
            function stepMN(c, Itot) {
                const v = c.V;
                // T-type Ca²⁺ inactivation (optional; gVT=0 by default for MNs)
                c.hCa += (hInfCa(v)-c.hCa)/tauHCa(v)*dt;
                const iT = c.gVT * Math.pow(mInfCa(v),3) * c.hCa * (v-VCa);
                // Transient Na⁺ gating variables (HH kinetics)
                c.mNa += (alphaM(v)*(1-c.mNa)-betaM(v)*c.mNa)*dt;  // fast activation
                c.hNa += (alphaH(v)*(1-c.hNa)-betaH(v)*c.hNa)*dt;  // slow inactivation
                c.n   += (alphaN(v)*(1-c.n)  -betaN(v)*c.n  )*dt;  // K⁺ delayed rectifier
                // Ionic currents
                const iNa  = c.gNa  * Math.pow(c.mNa,3)*c.hNa * (v-V_Na); // spike upstroke
                const iK   = c.gK   * Math.pow(c.n,4)          * (v-V_K);  // repolarisation/AHP
                const iNaP = c.gNaP * mNaPinf(v)                * (v-V_Na); // persistent, tonic drive
                const iL   = (c.gL + 1/c.Rin) * (v - c.VL);                // leak
                // Membrane voltage update: Cm·dV/dt = −iT − iNa − iK − iNaP − iL + Itot
                c.V = v + (-iT - iNa - iK - iNaP - iL + Itot)/Cm*dt;
                return c.V;
            }

            // ═════════════════════════════════════════════════════════════════════
            // GAP JUNCTION MODEL  (bidirectional, low-pass filtered)
            // ═════════════════════════════════════════════════════════════════════
            // Models an electrical synapse (connexin/innexin pore) between two cells.
            //
            // Ideal gap junction:  I_A→B = G · (V_B − V_A)
            // LP-filtered version: The voltage difference is filtered by a first-order
            //   low-pass with time constant τ = C/G. This accounts for the effective
            //   capacitance of the coupling compartment and prevents instantaneous
            //   artefacts from spike propagation (which would otherwise cause ringing).
            //
            //   dVf/dt = (V_B − V_A − Vf) / τ
            //   I_A    = G · Vf       (into cell A)
            //   I_B    = −G · Vf      (into cell B, equal and opposite)
            //
            // The filter is particularly important for LPTC ↔ MN junctions:
            //   τ = C/G (pCmn / gJunction) ≈ 0.8/0.1 = 8 ms at default settings.
            //   This 8 ms time constant prevents individual MN spikes from causing
            //   large voltage transients in the connected LPTC, while still passing
            //   the slower rate-coded signal.
            //
            // Within-chain LPTC GJs (VS1↔VS2 etc.):
            //   τ = pClptc / pGlptc ≈ 0.05/0.05 = 1 ms — faster, coupling adjacent cells
            //   tightly for spatial integration of the VS/HS visual response.

            // Create a new GJ object. G = conductance (nS); C = filter capacitance (nF).
            function createGJ(G, C) {
                return { G, tau: (G>1e-9 ? C/G : 0), Vf: 0 };  // τ = C/G; Vf = filtered voltage difference
            }
            // One integration step: update Vf and return [I_into_A, I_into_B].
            function gjPair(gj, Va, Vb) {
                const raw = Vb - Va;                                 // instantaneous voltage difference
                if (gj.tau > 1e-9) gj.Vf += (raw - gj.Vf)/gj.tau*dt;  // LP filter update
                else gj.Vf = raw;                                    // τ≈0 → ideal (no filter)
                const IA = gj.G * gj.Vf;                            // current into cell A
                return [IA, -IA];  // bidirectional: equal and opposite
            }

            // ═════════════════════════════════════════════════════════════════════
            // GRADED CHEMICAL SYNAPSE  (LPTC → MN or LPTC → LPTC)
            // ═════════════════════════════════════════════════════════════════════
            // Used for non-spiking (LPTC) presynaptic neurons. In graded transmission
            // the release probability scales continuously with presynaptic voltage,
            // not with discrete spikes.
            //
            // I_syn = −gMax · rel(Vpre) · (Vpost − Erev)
            //
            //   rel(Vpre) = clamp( (Vpre − Vthresh) / Vscale, 0, 1 )
            //     → linear ramp from Vthresh (no release) to Vthresh+Vscale (full release)
            //   Erev: 0 mV for excitatory (depolarising); −80 mV for inhibitory (GABA, Cl⁻)
            //   gMax = nSyn · gPerSyn   (total max conductance = contacts × per-contact value)
            //
            // Default thresholds: Vthresh = −40 mV, Vscale = 20 mV.
            //   At Vpre = −40 mV: no release (below threshold, baseline LPTC potential)
            //   At Vpre = −20 mV: full release (strongly depolarised LPTC)

            // Create a graded synapse. nSyn = number of anatomical contacts from RAW_COUNTS.
            function createGradedSyn(nSyn, gPerSyn, Erev) {
                return { gMax: nSyn*gPerSyn, Vthresh: pVthresh, Vscale: pVscale, Erev };
            }
            // Compute synaptic current for one time step.
            function gradedCurrent(s, Vpre, Vpost) {
                if (s.gMax < 1e-15) return 0;
                const rel = Math.max(0, Math.min(1, (Vpre-s.Vthresh)/s.Vscale)); // release fraction [0,1]
                return -s.gMax * rel * (Vpost - s.Erev);                          // driving force × conductance
            }

            // ═════════════════════════════════════════════════════════════════════
            // ALPHA-FUNCTION SYNAPSE  (MN → MN, spike-triggered)
            // ═════════════════════════════════════════════════════════════════════
            // Used for spiking (MN) presynaptic neurons. Each action potential triggers
            // a stereotyped conductance transient in the postsynaptic cell.
            //
            // The alpha-function conductance profile is:
            //   g(t) = (gMax/τ) · t · exp(−t/τ)   for t ≥ 0 after spike detection
            //
            // Implemented as a two-variable ODE (equivalent to the impulse response
            // of a second-order system):
            //   dg/dt  =  dg_kick                  (kick on spike, then decays)
            //   ddg/dt = −dg / τ
            //
            // I_syn = −g · (Vpost − Erev)
            //
            // On spike detection (Vpre crosses 0 mV threshold upward):
            //   dg += gMax / τ   (kick to the derivative, creates the rising phase)
            //
            // τ_syn = 5 ms default — roughly matching fast excitatory MN→MN synapses.
            // gMax = nSyn · gPerSyn.

            // Create an alpha-function synapse object.
            function createAlphaSyn(nSyn, gPerSyn, tau, Erev) {
                return { gMax: nSyn*gPerSyn, tau, g: 0, dg: 0, prevV: -65, thresh: 0, Erev };
            }
            // One time step: detect spike, update conductance, return synaptic current.
            function alphaStep(s, Vpre, Vpost) {
                // Spike detection: upward crossing of threshold (0 mV by default)
                if (Vpre > s.thresh && s.prevV <= s.thresh) s.dg += s.gMax/s.tau;
                s.prevV = Vpre;
                // Advance alpha-function ODE: g rises then decays with time constant tau
                s.g  += s.dg * dt;
                s.dg -= s.dg / s.tau * dt;
                s.g   = Math.max(0, s.g);                   // prevent negative conductance
                if (s.gMax < 1e-15) return 0;
                return -s.g * (Vpost - s.Erev);             // synaptic current (sign: inward = positive)
            }

            // ── Default parameters ──
            let pVS_GVT = 0.5, pVS_GL = 0.05, pVS_GK = 2.0;
            let pHS_GVT = 0.5, pHS_GL = 0.05, pHS_GK = 2.0;
            let pVS_Rin1 = 150, pVS_RinStep = -10, pVS_Vr1 = -40, pVS_VrStep = -5;
            let pHS_Rin = 150, pHS_Vr = -45;

            let pMOS_GVT = 0.0, pMOS_GL = 0.3, pMOS_Rin = 300;
            let pMOS_GNa = 120, pMOS_GK = 36, pMOS_VL = -65, pMOS_GNaP = 0.5, pMOS_Ibias = 0;
            let pMOT_GVT = 0.0, pMOT_GL = 0.3, pMOT_Rin = 300;
            let pMOT_GNa = 120, pMOT_GK = 36, pMOT_VL = -65, pMOT_GNaP = 0.5, pMOT_Ibias = 0;
            let pupilBaselineStart = 0, pupilBaselineWindow = 90, pupilDisableJitter = false, pupilUseRawRate = false;

            let pGlptc = 0.05, pClptc = 0.05;
            let pGvsmos = 0.1, pGhsmos = 0.1, pGhsmot = 0.1, pCmn = 0.8;
            let pGgradExc = 0.005, pGgradInh = 0.004;
            let pGspikeExc = 0.02, pGspikeInh = 0.016, pTauSyn = 5;
            let pVthresh = -40, pVscale = 20;
            const simPreRollMs = 500;

            function buildAndRun() {
                // FEATURE: CIRCUIT NEURON DELETION
                // Support for removing individual neurons from the simulation.
                // This is used to understand circuit dependency and test single-neuron lesions.
                // 
                // Deleted neurons (stored in circuitDisabledNodes set) are excluded from:
                // 1. Synapse transmission (both directions)
                // 2. Gap junction coupling (bidirectional electrical coupling)
                // 3. Stimulus injection (they cannot be stimulated)
                // 4. Output recording (cleaner traces)
                
                const disabledNodeSet = new Set(circuitDisabledNodes);  // User-selected deletions
                // ── Instantiate cells ──
                const cells = [];
                const VS_Vr  = [pVS_Vr1, pVS_Vr1 + pVS_VrStep, pVS_Vr1 + 2 * pVS_VrStep, pVS_Vr1 + 3 * pVS_VrStep];
                const VS_Rin = [pVS_Rin1, pVS_Rin1 + pVS_RinStep, pVS_Rin1 + 2 * pVS_RinStep, pVS_Rin1 + 3 * pVS_RinStep];
                CELL_NAMES.forEach(n => {
                    const enabled = !disabledNodeSet.has(n);  // Mark deleted neurons as disabled
                    let cell;
                    if (n.startsWith('VS')) {
                        const k = parseInt(n[2]) - 1;
                        cell = createLPTC(n, VS_Rin[k], VS_Vr[k], pVS_GVT, pVS_GL, pVS_GK);
                    } else if (n.startsWith('HS')) {
                        cell = createLPTC(n, pHS_Rin, pHS_Vr, pHS_GVT, pHS_GL, pHS_GK);
                    } else if (n.startsWith('MOS')) {
                        cell = createMN(n, pMOS_Rin, pMOS_VL, pMOS_GVT, pMOS_GL, pMOS_GNa, pMOS_GK, pMOS_GNaP);
                    } else if (n.startsWith('MOT')) {
                        cell = createMN(n, pMOT_Rin, pMOT_VL, pMOT_GVT, pMOT_GL, pMOT_GNa, pMOT_GK, pMOT_GNaP);
                    } else {
                        // Other cells (BIPS/H2) use MOT defaults unless split out later.
                        cell = createMN(n, pMOT_Rin, pMOT_VL, pMOT_GVT, pMOT_GL, pMOT_GNa, pMOT_GK, pMOT_GNaP);
                    }
                    cell.enabled = enabled;  // Store enabled flag for synapse checks
                    cells.push(cell);
                });

                // ── Chemical synapses from RAW_COUNTS ──
                // Only create synapses between enabled neurons
                const synapses = [];
                for (let pi = 0; pi < N_CELLS; pi++) {
                    for (let qi = 0; qi < N_CELLS; qi++) {
                        const cnt = RAW_COUNTS[pi][qi];
                        // DELETION CHECK: Skip synapses if pre or post neuron is disabled
                        if (cnt === 0 || !cells[pi].enabled || !cells[qi].enabled) continue;
                        const Erev = SYN_ESYN[pi][qi];
                        const gPerSyn = (Erev < -10)
                            ? (SPIKING.has(CELL_NAMES[pi]) ? pGspikeInh : pGgradInh)
                            : (SPIKING.has(CELL_NAMES[pi]) ? pGspikeExc : pGgradExc);
                        if (SPIKING.has(CELL_NAMES[pi])) {
                            synapses.push({ pre: pi, post: qi, obj: createAlphaSyn(cnt, gPerSyn, pTauSyn, Erev) });
                        } else {
                            synapses.push({ pre: pi, post: qi, obj: createGradedSyn(cnt, gPerSyn, Erev) });
                        }
                    }
                }

                // ── Gap junctions (bidirectional, LP-filtered) ──
                // Only create gap junctions between enabled neuron pairs
                const gjList = [];
                // Within VS chains (VS1↔VS2↔VS3↔VS4)
                ['L','R'].forEach(s => {
                    for (let k = 1; k <= 3; k++) {
                        const a = CI['VS'+k+'_'+s], b = CI['VS'+(k+1)+'_'+s];
                        // DELETION CHECK: Only create GJ if both neurons are enabled
                        if (cells[a].enabled && cells[b].enabled)
                            gjList.push({ a, b, gj: createGJ(pGlptc, pClptc) });
                    }
                    // Within HS chains (HSN↔HSE↔HSS)
                    if (cells[CI['HSN_'+s]].enabled && cells[CI['HSE_'+s]].enabled)
                        gjList.push({ a: CI['HSN_'+s], b: CI['HSE_'+s], gj: createGJ(pGlptc, pClptc) });
                    if (cells[CI['HSE_'+s]].enabled && cells[CI['HSS_'+s]].enabled)
                        gjList.push({ a: CI['HSE_'+s], b: CI['HSS_'+s], gj: createGJ(pGlptc, pClptc) });
                    // VS ↔ MOS (bidirectional)
                    const mos = CI['MOS_'+s], mot = CI['MOT_'+s];
                    for (let k = 1; k <= 4; k++) {
                        const a = CI['VS'+k+'_'+s];
                        if (cells[a].enabled && cells[mos].enabled)
                            gjList.push({ a, b: mos, gj: createGJ(pGvsmos, pCmn) });
                    }
                    // HS ↔ MOS (bidirectional)
                    ['HSN','HSE','HSS'].forEach(h => {
                        const a = CI[h+'_'+s];
                        if (cells[a].enabled && cells[mos].enabled)
                            gjList.push({ a, b: mos, gj: createGJ(pGhsmos, pCmn) });
                    });
                    // HS ↔ MOT (bidirectional; VS does NOT connect to MOT)
                    ['HSN','HSE','HSS'].forEach(h => {
                        const a = CI[h+'_'+s];
                        if (cells[a].enabled && cells[mot].enabled)
                            gjList.push({ a, b: mot, gj: createGJ(pGhsmot, pCmn) });
                    });
                });

                // ── Simulation ──
                const totalSimTime = simTime + simPreRollMs;
                const nSteps = Math.round(totalSimTime / dt);
                const rec = {};
                CELL_NAMES.forEach(n => { rec[n] = new Float32Array(nSteps); });
                const tArr = new Float32Array(nSteps);

                for (let n = 0; n < N_CELLS; n++) rec[CELL_NAMES[n]][0] = cells[n].V;

                for (let step = 1; step < nSteps; step++) {
                    const t = step * dt;
                    tArr[step] = t;

                    // 1. Gap junction currents (bidirectional)
                    const gjI = new Float64Array(N_CELLS);
                    gjList.forEach(g => {
                        const [iA, iB] = gjPair(g.gj, cells[g.a].V, cells[g.b].V);
                        gjI[g.a] += iA;
                        gjI[g.b] += iB;
                    });

                    // 2. Chemical synapse currents
                    const chemI = new Float64Array(N_CELLS);
                    synapses.forEach(s => {
                        const Vpre = cells[s.pre].V, Vpost = cells[s.post].V;
                        let I;
                        if (s.obj.g !== undefined) I = alphaStep(s.obj, Vpre, Vpost);
                        else I = gradedCurrent(s.obj, Vpre, Vpost);
                        chemI[s.post] += I;
                    });

                    // 3. External stimulus
                    const extI = new Float64Array(N_CELLS);
                    const stimStartAbs = simPreRollMs + stimStart;
                    const stimEndAbs = simPreRollMs + stimEnd;
                    if (t >= stimStartAbs && t <= stimEndAbs) {
                        stimTargets.forEach(sn => {
                            if (CI[sn] !== undefined && cells[CI[sn]].enabled) extI[CI[sn]] += stimAmp;
                        });
                    }
                    // Tonic bias for MN cells (active throughout simulation)
                    for (let n = 0; n < N_CELLS; n++) {
                        if (!cells[n].enabled || cells[n].type === 'LPTC') continue;
                        const nm = CELL_NAMES[n];
                        if (nm.startsWith('MOS')) extI[n] += pMOS_Ibias;
                        else extI[n] += pMOT_Ibias;
                    }
                    // Noise
                    for (let n = 0; n < N_CELLS; n++)
                        if (cells[n].enabled) extI[n] += noiseLevel * (Math.random()*2-1);

                    // 4. Step all cells
                    for (let n = 0; n < N_CELLS; n++) {
                        if (!cells[n].enabled) {
                            cells[n].V = cells[n].VL;
                            rec[CELL_NAMES[n]][step] = cells[n].V;
                            continue;
                        }
                        const Itot = gjI[n] + chemI[n] + extI[n];
                        if (cells[n].type === 'LPTC') stepLPTC(cells[n], Itot);
                        else stepMN(cells[n], Itot);
                        rec[CELL_NAMES[n]][step] = cells[n].V;
                    }
                }
                return { t: tArr, records: rec, preRollMs: simPreRollMs };
            }

            // ── Simulation state ──
            let simTime = 1500, stimStart = 90, stimEnd = 590, stimAmp = 10;
            let noiseLevel = 3;
            let stimTargets = ['VS1_L','VS2_L','VS3_L','VS4_L'];
            const circuitDisabledNodes = new Set();

            // ── Build UI ──
            let html = '<div style="display:flex;flex-direction:column;gap:6px;">';

            // Wiring diagram (SVG)
            html += '<div id="wiringDiagram" style="background:#1a1a2e;border:1px solid #444;'
                + 'border-radius:4px;padding:8px;overflow-x:auto;"></div>';
            html += '<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;">';
            html += '<div id="circuitToggleStatus" style="color:#9aa0a6;font-size:10px;">'
                + 'Click a neuron in the wiring diagram to deactivate it and all of its connections.'
                + '</div>';
            html += '<button id="circResetNodes" style="background:#37474f;color:#fff;border:1px solid #607d8b;'
                + 'padding:3px 10px;border-radius:3px;cursor:pointer;font-size:10px;">Reset node toggles</button>';
            html += '</div>';

            // Controls
            html += '<div style="display:flex;gap:8px;align-items:flex-start;flex-wrap:wrap;padding:4px 0;">';
            html += '<label style="color:#ccc;font-size:11px;display:flex;flex-direction:column;gap:2px;">'
                + '<span>Stim targets <span style="color:#888;font-size:9px;">(Ctrl/\u2318 multi)</span></span>'
                + '<select id="circStimGroup" multiple size="6" '
                + 'style="background:#1e1e1e;color:#fff;border:1px solid #555;font-size:10px;min-width:110px;">'
                + '<optgroup label="\u2014 Groups \u2014" style="color:#888;">'
                + '<option value="VS_L" selected>VS Left (1-4)</option>'
                + '<option value="VS_R">VS Right (1-4)</option>'
                + '<option value="HS_L">HS Left</option>'
                + '<option value="HS_R">HS Right</option>'
                + '<option value="VS_ALL">All VS (both sides)</option>'
                + '<option value="HS_ALL">All HS (both sides)</option>'
                + '<option value="ALL_L">All Left LPTCs</option>'
                + '<option value="ALL_R">All Right LPTCs</option>'
                + '<option value="MN_L">MOS+MOT Left</option>'
                + '<option value="MN_R">MOS+MOT Right</option>'
                + '</optgroup>'
                + '<optgroup label="\u2014 MOT / MOS \u2014" style="color:#888;">'
                + '<option value="MOT_L">MOT_L</option>'
                + '<option value="MOT_R">MOT_R</option>'
                + '<option value="MOS_L">MOS_L</option>'
                + '<option value="MOS_R">MOS_R</option>'
                + '</optgroup>'
                + '<optgroup label="\u2014 VS \u2014" style="color:#888;">'
                + '<option value="VS1_L">VS1_L</option>'
                + '<option value="VS1_R">VS1_R</option>'
                + '<option value="VS2_L">VS2_L</option>'
                + '<option value="VS2_R">VS2_R</option>'
                + '<option value="VS3_L">VS3_L</option>'
                + '<option value="VS3_R">VS3_R</option>'
                + '<option value="VS4_L">VS4_L</option>'
                + '<option value="VS4_R">VS4_R</option>'
                + '</optgroup>'
                + '<optgroup label="\u2014 HS \u2014" style="color:#888;">'
                + '<option value="HSN_L">HSN_L</option>'
                + '<option value="HSN_R">HSN_R</option>'
                + '<option value="HSE_L">HSE_L</option>'
                + '<option value="HSE_R">HSE_R</option>'
                + '<option value="HSS_L">HSS_L</option>'
                + '<option value="HSS_R">HSS_R</option>'
                + '</optgroup>'
                + '</select></label>';
            html += '<div style="display:flex;flex-direction:column;gap:4px;">';
            html += '<label style="color:#ccc;font-size:10px;">Amp (nA): '
                + '<input id="circStimAmp" type="number" value="10" step="1" min="-100" max="100" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ccc;font-size:10px;">t<sub>start</sub>: '
                + '<input id="circStimStart" type="number" value="90" step="10" min="0" max="8000" '
                + 'style="width:55px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ccc;font-size:10px;">t<sub>end</sub>: '
                + '<input id="circStimEnd" type="number" value="590" step="10" min="0" max="8000" '
                + 'style="width:55px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ccc;font-size:10px;">T<sub>max</sub>: '
                + '<input id="circSimTime" type="number" value="1500" step="100" min="100" max="10000" '
                + 'style="width:55px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ccc;font-size:10px;">Noise: '
                + '<input id="circNoise" type="number" value="3" step="0.5" min="0" max="10" '
                + 'style="width:45px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ffab91;font-size:10px;" title="Target resting MOT firing rate (Hz) used by auto-calibration.">MOT rest target: '
                + '<input id="pMOT_TargetRestHz" type="number" value="120" step="5" min="0" max="300" '
                + 'style="width:45px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ef9a9a;font-size:10px;" title="Target resting MOS firing rate (Hz) used by auto-calibration.">MOS rest target: '
                + '<input id="pMOS_TargetRestHz" type="number" value="100" step="5" min="0" max="300" '
                + 'style="width:45px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<button id="circPresetMotor" style="background:#455a64;color:#fff;border:1px solid #607d8b;'
                + 'padding:4px 10px;border-radius:3px;cursor:pointer;font-size:10px;">'
                + 'Preset: MOT~120 / MOS~100</button>';
            html += '<button id="circCalibrateRestHz" style="background:#5d4037;color:#fff;border:1px solid #8d6e63;'
                + 'padding:4px 10px;border-radius:3px;cursor:pointer;font-size:10px;">'
                + 'Auto-Calibrate Rest Hz (Intrinsic-first)</button>';
            html += '<button id="circRun" style="background:#2E7D32;color:#fff;border:1px solid #4CAF50;'
                + 'padding:4px 14px;border-radius:3px;cursor:pointer;font-size:11px;font-weight:bold;">'
                + '\u25b6 Run</button>';
            html += '</div>';
            html += '<div id="circCalibStatus" style="margin-top:6px;color:#90caf9;font-size:10px;line-height:1.35;'
                + 'border:1px solid #37474f;background:#121820;padding:6px;border-radius:4px;">'
                + 'Calibration report: none yet.</div>';
            html += '</div>';

            // ── Cell Parameters ──
            html += '<details style="color:#aaa;font-size:10px;">'
                + '<summary style="cursor:pointer;color:#ef9a9a;">Cell Parameters (MN &amp; LPTC)</summary>';
            html += '<div style="display:flex;gap:10px;flex-wrap:wrap;padding:4px 0;">';

            // MOS-specific spiking parameters
            html += '<fieldset style="border:1px solid #555;padding:4px 8px;margin:0;">'
                + '<legend style="color:#ef5350;font-size:10px;">MOS cells (left + right)</legend>'
                + '<div style="display:flex;gap:6px;flex-wrap:wrap;">';
            html += '<label style="color:#ef5350;font-size:10px;" title="Leak reversal potential (mV). More negative \u2192 slower spontaneous rate.">V<sub>L-MOS</sub>: '
                + '<input id="pMOS_VLm" type="number" value="-65" step="1" min="-90" max="-40" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ef5350;font-size:10px;" title="Persistent Na\u207a conductance (nS). Primary driver of spontaneous firing.">g<sub>NaP-MOS</sub>: '
                + '<input id="pMOS_GNaP" type="number" value="0.5" step="0.05" min="0" max="3" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ef5350;font-size:10px;" title="Transient Na\u207a conductance (nS). Controls spike height.">g<sub>Na-MOS</sub>: '
                + '<input id="pMOS_GNa" type="number" value="120" step="5" min="10" max="300" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ef5350;font-size:10px;" title="Delayed-rectifier K\u207a conductance (nS).">g<sub>K-MOS</sub>: '
                + '<input id="pMOS_GKm" type="number" value="36" step="2" min="5" max="150" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ef5350;font-size:10px;" title="Leak conductance (nS).">g<sub>L-MOS</sub>: '
                + '<input id="pMOS_GLm" type="number" value="0.3" step="0.05" min="0" max="5" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ef5350;font-size:10px;" title="Input resistance (M\u03a9).">R<sub>in-MOS</sub>: '
                + '<input id="pMOS_RinM" type="number" value="300" step="10" min="50" max="1000" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ef5350;font-size:10px;" title="Tonic bias current for MOS cells (nA).">I<sub>bias-MOS</sub>: '
                + '<input id="pMOS_IbiasM" type="number" value="0" step="0.5" min="-20" max="20" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ef5350;font-size:10px;" title="T-Ca conductance in MOS cells (nS).">g<sub>VT-MOS</sub>: '
                + '<input id="pMOS_GVTm" type="number" value="0" step="0.05" min="0" max="2" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '</div></fieldset>';

            // MOT-specific spiking parameters
            html += '<fieldset style="border:1px solid #555;padding:4px 8px;margin:0;">'
                + '<legend style="color:#ff7043;font-size:10px;">MOT cells (left + right)</legend>'
                + '<div style="display:flex;gap:6px;flex-wrap:wrap;">';
            html += '<label style="color:#ff7043;font-size:10px;" title="Leak reversal potential (mV).">V<sub>L-MOT</sub>: '
                + '<input id="pMOT_VLm" type="number" value="-65" step="1" min="-90" max="-40" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ff7043;font-size:10px;" title="Persistent Na\u207a conductance (nS).">g<sub>NaP-MOT</sub>: '
                + '<input id="pMOT_GNaP" type="number" value="0.5" step="0.05" min="0" max="3" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ff7043;font-size:10px;" title="Transient Na\u207a conductance (nS).">g<sub>Na-MOT</sub>: '
                + '<input id="pMOT_GNa" type="number" value="120" step="5" min="10" max="300" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ff7043;font-size:10px;" title="Delayed-rectifier K\u207a conductance (nS).">g<sub>K-MOT</sub>: '
                + '<input id="pMOT_GKm" type="number" value="36" step="2" min="5" max="150" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ff7043;font-size:10px;" title="Leak conductance (nS).">g<sub>L-MOT</sub>: '
                + '<input id="pMOT_GLm" type="number" value="0.3" step="0.05" min="0" max="5" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ff7043;font-size:10px;" title="Input resistance (M\u03a9).">R<sub>in-MOT</sub>: '
                + '<input id="pMOT_RinM" type="number" value="300" step="10" min="50" max="1000" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ff7043;font-size:10px;" title="Tonic bias current for MOT cells (nA).">I<sub>bias-MOT</sub>: '
                + '<input id="pMOT_IbiasM" type="number" value="0" step="0.5" min="-20" max="20" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ff7043;font-size:10px;" title="T-Ca conductance in MOT cells (nS).">g<sub>VT-MOT</sub>: '
                + '<input id="pMOT_GVTm" type="number" value="0" step="0.05" min="0" max="2" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '</div></fieldset>';

            // VS non-spiking parameters
            html += '<fieldset style="border:1px solid #555;padding:4px 8px;margin:0;">'
                + '<legend style="color:#ab47bc;font-size:10px;">VS cells (left + right)</legend>'
                + '<div style="display:flex;gap:6px;flex-wrap:wrap;">';
            html += '<label style="color:#ab47bc;font-size:10px;" title="VS1 resting potential (mV). VS2/3/4 are offset by the step parameter.">V<sub>r-VS1</sub>: '
                + '<input id="pVS_Vr1" type="number" value="-40" step="1" min="-80" max="-10" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ab47bc;font-size:10px;" title="Per-index resting potential increment for VS2-4 (mV).">\u0394V<sub>r-step</sub>: '
                + '<input id="pVS_VrStep" type="number" value="-5" step="1" min="-20" max="20" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ab47bc;font-size:10px;" title="VS1 input resistance (M\u03a9).">R<sub>in-VS1</sub>: '
                + '<input id="pVS_Rin1" type="number" value="150" step="10" min="50" max="1000" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ab47bc;font-size:10px;" title="Per-index resistance increment for VS2-4 (M\u03a9).">\u0394R<sub>in-step</sub>: '
                + '<input id="pVS_RinStep" type="number" value="-10" step="1" min="-100" max="100" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ab47bc;font-size:10px;" title="VS T-type Ca\u00b2\u207a conductance (nS).">g<sub>VT-VS</sub>: '
                + '<input id="pVS_GVTl" type="number" value="0.5" step="0.05" min="0" max="3" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ab47bc;font-size:10px;" title="VS K\u207a conductance (nS).">g<sub>K-VS</sub>: '
                + '<input id="pVS_GKl" type="number" value="2.0" step="0.1" min="0" max="10" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ab47bc;font-size:10px;" title="VS leak conductance (nS).">g<sub>L-VS</sub>: '
                + '<input id="pVS_GLl" type="number" value="0.05" step="0.005" min="0" max="1" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '</div></fieldset>';

            // HS non-spiking parameters
            html += '<fieldset style="border:1px solid #555;padding:4px 8px;margin:0;">'
                + '<legend style="color:#4fc3f7;font-size:10px;">HS cells (left + right)</legend>'
                + '<div style="display:flex;gap:6px;flex-wrap:wrap;">';
            html += '<label style="color:#4fc3f7;font-size:10px;" title="HS resting potential (mV).">V<sub>r-HS</sub>: '
                + '<input id="pHS_Vr" type="number" value="-45" step="1" min="-80" max="-10" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#4fc3f7;font-size:10px;" title="HS input resistance (M\u03a9).">R<sub>in-HS</sub>: '
                + '<input id="pHS_Rin" type="number" value="150" step="10" min="50" max="1000" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#4fc3f7;font-size:10px;" title="HS T-type Ca\u00b2\u207a conductance (nS).">g<sub>VT-HS</sub>: '
                + '<input id="pHS_GVTl" type="number" value="0.5" step="0.05" min="0" max="3" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#4fc3f7;font-size:10px;" title="HS K\u207a conductance (nS).">g<sub>K-HS</sub>: '
                + '<input id="pHS_GKl" type="number" value="2.0" step="0.1" min="0" max="10" '
                + 'style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#4fc3f7;font-size:10px;" title="HS leak conductance (nS).">g<sub>L-HS</sub>: '
                + '<input id="pHS_GLl" type="number" value="0.05" step="0.005" min="0" max="1" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '</div></fieldset>';

            html += '</div></details>';

            // GJ/Synapse parameter row
            html += '<details style="color:#aaa;font-size:10px;"><summary style="cursor:pointer;color:#aed581;">GJ &amp; Synapse Parameters</summary>';
            html += '<div style="display:flex;gap:8px;flex-wrap:wrap;padding:4px 0;">';
            html += '<label style="color:#aed581;font-size:10px;">G<sub>LPTC-GJ</sub>: '
                + '<input id="pGlptc" type="number" value="0.05" step="0.005" min="0" max="1" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#aed581;font-size:10px;">C<sub>LPTC-GJ</sub>: '
                + '<input id="pClptc" type="number" value="0.05" step="0.005" min="0" max="1" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#80cbc4;font-size:10px;" title="VS1-4 to MOS gap junction conductance (nS)">G<sub>VS\u2194MOS</sub>: '
                + '<input id="pGvsmos" type="number" value="0.1" step="0.01" min="0" max="2" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#80cbc4;font-size:10px;" title="HS to MOS gap junction conductance (nS)">G<sub>HS\u2194MOS</sub>: '
                + '<input id="pGhsmos" type="number" value="0.1" step="0.01" min="0" max="2" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#80cbc4;font-size:10px;" title="HS to MOT gap junction conductance (nS). VS does not couple to MOT.">G<sub>HS\u2194MOT</sub>: '
                + '<input id="pGhsmot" type="number" value="0.1" step="0.01" min="0" max="2" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#80cbc4;font-size:10px;" title="LP-filter capacitance for LPTC\u2194MN gap junctions (sets tau = C/G)">C<sub>MN-GJ</sub>: '
                + '<input id="pCmn" type="number" value="0.8" step="0.01" min="0" max="2" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ffcc02;font-size:10px;" title="Per-contact graded chemical conductance for excitatory synapses (nS)">g<sub>grad-exc</sub>: '
                + '<input id="pGgradExc" type="number" value="0.005" step="0.001" min="0" max="0.05" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#64b5f6;font-size:10px;" title="Per-contact graded chemical conductance for inhibitory synapses (nS)">g<sub>grad-inh</sub>: '
                + '<input id="pGgradInh" type="number" value="0.004" step="0.001" min="0" max="0.05" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ffcc02;font-size:10px;" title="Per-contact alpha-synapse increment for excitatory MN-pre synapses (nS)">g<sub>spike-exc</sub>: '
                + '<input id="pGspikeExc" type="number" value="0.02" step="0.005" min="0" max="0.2" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#64b5f6;font-size:10px;" title="Per-contact alpha-synapse increment for inhibitory MN-pre synapses (nS)">g<sub>spike-inh</sub>: '
                + '<input id="pGspikeInh" type="number" value="0.016" step="0.005" min="0" max="0.2" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ffcc02;font-size:10px;">\u03c4<sub>syn</sub>: '
                + '<input id="pTauSyn" type="number" value="5" step="0.5" min="0.5" max="50" '
                + 'style="width:45px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ef9a9a;font-size:10px;">g<sub>VT-MN</sub>: '
                + '<input id="pGVTm" type="number" value="0.0" step="0.05" min="0" max="2" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;" '
                + 'title="T-Ca conductance in spiking MN cells. Keep at 0 to avoid burst-then-silence."></label>';
            html += '<label style="color:#ef9a9a;font-size:10px;">V<sub>thresh-grad</sub>: '
                + '<input id="pVthresh" type="number" value="-40" step="1" min="-80" max="0" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;" '
                + 'title="Release threshold for graded synapses (mV). Set equal to most depolarised LPTC resting V to prevent tonic release."></label>';
            html += '<label style="color:#ef9a9a;font-size:10px;">V<sub>scale-grad</sub>: '
                + '<input id="pVscale" type="number" value="20" step="1" min="1" max="60" '
                + 'style="width:45px;background:#333;color:#fff;border:1px solid #555;font-size:10px;" '
                + 'title="mV range over which graded release goes from 0 to 1."></label>';
            html += '</div></details>';

            // Pseudopupil analysis controls
            html += '<details style="color:#aaa;font-size:10px;"><summary style="cursor:pointer;color:#80deea;">Pseudopupil Analysis (baseline &amp; readout mode)</summary>';
            html += '<div style="display:flex;gap:6px;flex-wrap:wrap;padding:4px 0;align-items:center;">';
            html += '<label style="color:#80deea;font-size:10px;" title="Time (ms) offset from stimulus start for baseline window.">';
            html += 'Baseline start offset (ms): <input id="pupilBaselineStart" type="number" value="0" step="5" min="-200" max="200" style="width:55px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#80deea;font-size:10px;" title="Duration (ms) of baseline sampling window.">';
            html += 'Baseline window (ms): <input id="pupilBaselineWindow" type="number" value="90" step="5" min="10" max="500" style="width:55px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#80deea;font-size:10px;"><input id="pupilDisableJitter" type="checkbox" style="width:12px;height:12px;" title="Disable rate-transient-driven jitter."> Disable transient jitter</label>';
            html += '<label style="color:#80deea;font-size:10px;"><input id="pupilUseRawRate" type="checkbox" style="width:12px;height:12px;" title="Use raw firing rate instead of baseline-subtracted response."> Use raw rate (not Δ)</label>';
            html += '</div></details>';

            // Plot areas
            html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">'
                + '<div id="circPlotLPTC_R" style="width:100%;height:180px;background:#ffffff;border:1px solid #ccc;"></div>'
                + '<div id="circPlotLPTC_L" style="width:100%;height:180px;background:#ffffff;border:1px solid #ccc;"></div>'
                + '</div>';
            html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">'
                + '<div id="circPlotMN_R" style="width:100%;height:180px;background:#ffffff;border:1px solid #ccc;"></div>'
                + '<div id="circPlotMN_L" style="width:100%;height:180px;background:#ffffff;border:1px solid #ccc;"></div>'
                + '</div>';
            html += '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">'
                + '<div id="circPlotPupilTime_R" style="width:100%;height:210px;background:#ffffff;border:1px solid #ccc;"></div>'
                + '<div id="circPlotPupilTime_L" style="width:100%;height:210px;background:#ffffff;border:1px solid #ccc;"></div>'
                + '</div>';
            html += '<div id="circPlotPupilPolar" style="width:100%;height:230px;background:#ffffff;border:1px solid #ccc;"></div>';
            html += '</div>';

            circuitContainer.innerHTML = html;

            function updateCircuitToggleStatus() {
                const el = document.getElementById('circuitToggleStatus');
                if (!el) return;
                const disabled = Array.from(circuitDisabledNodes).sort();
                el.textContent = disabled.length
                    ? ('Inactive nodes: ' + disabled.join(', '))
                    : 'Click a neuron in the wiring diagram to deactivate it and all of its connections.';
            }

            function drawWiring() {
            try {
                const W = 880, H = 560;
                const CVS='#ce93d8', CHS='#4fc3f7', CMOS='#ef5350', CMOT='#ff7043';
                const CGJ='#aed581', CCHEM='#ffcc02', CT='#ccc', CBG='#0d1b2e';
                const isActive = name => !circuitDisabledNodes.has(name);
                const allActive = names => names.every(isActive);
                const lineOpacity = (names, on, off) => allActive(names) ? on : off;
                const lineColor = (names, on, off) => allActive(names) ? on : off;
                let svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+W+' '+H+'" '
                    + 'style="width:100%;max-height:560px;font-family:sans-serif;">';
                svg += '<rect width="'+W+'" height="'+H+'" fill="#1a1a2e"/>';
                svg += '<text x="'+W/2+'" y="20" text-anchor="middle" fill="'+CT+'" font-size="13" font-weight="bold">Circuit Wiring Diagram</text>';
                svg += '<text x="'+W/2+'" y="34" text-anchor="middle" fill="#888" font-size="9">\u27f7 green dashed = bidirectional GJ (LP-filtered) &nbsp; \u2192 yellow = chemical synapse (n = count) &nbsp; click node = toggle</text>';

                const pos = {};
                const lx_vs = 55, lx_hs = 165, lx_mos = 290, lx_mot = 290;
                const vs_ys = [70, 125, 180, 235];
                const hs_ys = [70, 125, 180];
                for (let k = 0; k < 4; k++) pos['VS'+(k+1)+'_L'] = [lx_vs, vs_ys[k]];
                ['HSN','HSE','HSS'].forEach((h,i) => { pos[h+'_L'] = [lx_hs, hs_ys[i]]; });
                pos['MOS_L'] = [lx_mos, 97];
                pos['MOT_L'] = [lx_mot, 210];
                pos['BIPS_L'] = [lx_hs, 260];
                pos['H2_L'] = [(lx_hs + lx_mos) / 2, 48];

                const rx_vs = W-55, rx_hs = W-165, rx_mos = W-290, rx_mot = W-290;
                for (let k = 0; k < 4; k++) pos['VS'+(k+1)+'_R'] = [rx_vs, vs_ys[k]];
                ['HSN','HSE','HSS'].forEach((h,i) => { pos[h+'_R'] = [rx_hs, hs_ys[i]]; });
                pos['MOS_R'] = [rx_mos, 97];
                pos['MOT_R'] = [rx_mot, 210];
                pos['BIPS_R'] = [rx_hs, 260];
                pos['H2_R'] = [(rx_hs + rx_mos) / 2, 48];

                function neuronBox(name, color) {
                    const p = pos[name];
                    if (!p) return;
                    const active = isActive(name);
                    const bw = 72, bh = 30;
                    svg += '<g class="circuit-node" data-neuron="'+name+'" style="cursor:pointer">';
                    svg += '<rect x="'+(p[0]-bw/2)+'" y="'+(p[1]-bh/2)+'" width="'+bw+'" height="'+bh+'" '
                        + 'rx="5" fill="'+(active ? CBG : '#141414')+'" stroke="'+(active ? color : '#5f6368')+'" stroke-width="1.5" opacity="'+(active ? '1' : '0.72')+'"/>';
                    svg += '<text x="'+p[0]+'" y="'+(p[1]+3)+'" text-anchor="middle" fill="'+(active ? color : '#9aa0a6')+'" font-size="9" font-weight="bold" pointer-events="none">'+name+'</text>';
                    if (!active) svg += '<text x="'+p[0]+'" y="'+(p[1]+12)+'" text-anchor="middle" fill="#b0b4b8" font-size="6.5" pointer-events="none">off</text>';
                    svg += '</g>';
                }
                CELL_NAMES.forEach(n => neuronBox(n, neuronColors[n] || CMOS));

                function gjLine(a, b) {
                    const p1 = pos[a], p2 = pos[b];
                    const op = lineOpacity([a, b], 0.85, 0.08);
                    const col = lineColor([a, b], CGJ, '#455a64');
                    svg += '<line x1="'+p1[0]+'" y1="'+p1[1]+'" x2="'+p2[0]+'" y2="'+p2[1]+'" stroke="'+col+'" stroke-width="2" stroke-dasharray="5,3" opacity="'+op+'"/>';
                    const mx = (p1[0]+p2[0])/2, my = (p1[1]+p2[1])/2;
                    svg += '<line x1="'+(mx-5)+'" y1="'+my+'" x2="'+(mx+5)+'" y2="'+my+'" stroke="'+col+'" stroke-width="3" opacity="'+op+'"/>';
                }
                ['L','R'].forEach(s => {
                    for (let k = 1; k <= 3; k++) gjLine('VS'+k+'_'+s, 'VS'+(k+1)+'_'+s);
                    gjLine('HSN_'+s, 'HSE_'+s);
                    gjLine('HSE_'+s, 'HSS_'+s);
                });

                function contraGJ(a, b) {
                    const p1 = pos[a], p2 = pos[b];
                    if (!p1 || !p2) return;
                    const mx = W/2, my = Math.min(p1[1], p2[1]) - 15;
                    svg += '<path d="M'+p1[0]+','+p1[1]+' Q'+mx+','+my+' '+p2[0]+','+p2[1]+'" fill="none" stroke="'+lineColor([a, b], CGJ, '#455a64')+'" stroke-width="1.5" stroke-dasharray="5,3" opacity="'+lineOpacity([a, b], 0.55, 0.08)+'"/>';
                }
                ['HSN','HSE','HSS'].forEach(h => {
                    contraGJ('H2_L', h+'_R');
                    contraGJ('H2_R', h+'_L');
                });

                function gjArrow(a, b) {
                    const p1 = pos[a], p2 = pos[b];
                    const dx = p2[0]-p1[0], dy = p2[1]-p1[1];
                    const len = Math.sqrt(dx*dx+dy*dy);
                    const ux = dx/len, uy = dy/len;
                    const x1 = p1[0]+ux*38, y1 = p1[1]+uy*16;
                    const x2 = p2[0]-ux*38, y2 = p2[1]-uy*16;
                    svg += '<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+lineColor([a, b], CGJ, '#455a64')+'" stroke-width="1.5" stroke-dasharray="4,3" opacity="'+lineOpacity([a, b], 0.75, 0.08)+'"/>';
                }
                ['L','R'].forEach(s => {
                    for (let k = 1; k <= 4; k++) gjArrow('VS'+k+'_'+s, 'MOS_'+s);
                    ['HSN','HSE','HSS'].forEach(h => {
                        gjArrow(h+'_'+s, 'MOS_'+s);
                        gjArrow(h+'_'+s, 'MOT_'+s);
                    });
                });

                function chemArrow(pre, post, n, dy) {
                    const pi = CI[pre], qi = CI[post];
                    const p1 = pos[pre], p2 = pos[post];
                    if (!p1 || !p2) return;
                    const active = allActive([pre, post]);
                    const isInh = SYN_ESYN[pi][qi] < -10;
                    const col = active ? (isInh ? '#64b5f6' : CCHEM) : '#455a64';
                    const dx2 = p2[0]-p1[0], dy2 = p2[1]-p1[1];
                    const len = Math.sqrt(dx2*dx2+dy2*dy2);
                    if (len < 1) return;
                    const ux = dx2/len, uy = dy2/len;
                    const x1 = p1[0]+ux*38, y1 = p1[1]+uy*16 + (dy||0);
                    const x2 = p2[0]-ux*38, y2 = p2[1]-uy*16 + (dy||0);
                    const aid = 'ca_'+pre+'_'+post;
                    if (isInh) {
                        svg += '<defs><marker id="'+aid+'" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto"><line x1="0" y1="0" x2="0" y2="8" stroke="'+col+'" stroke-width="2"/></marker></defs>';
                    } else {
                        svg += '<defs><marker id="'+aid+'" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="'+col+'"/></marker></defs>';
                    }
                    svg += '<path d="M'+x1+','+y1+' L'+x2+','+y2+'" stroke="'+col+'" stroke-width="'+Math.max(0.5, Math.sqrt(n) * 0.4)+'" fill="none" marker-end="url(#'+aid+')" opacity="'+(active ? '0.7' : '0.08')+'"/>';
                    const mx = (x1+x2)/2, my = (y1+y2)/2 - 4;
                    svg += '<text x="'+mx+'" y="'+my+'" text-anchor="middle" fill="'+(active ? col : '#7b8794')+'" font-size="7" font-weight="bold" opacity="'+(active ? '1' : '0.35')+'">'+n+'</text>';
                }
                for (let pi = 0; pi < N_CELLS; pi++) {
                    for (let qi = 0; qi < N_CELLS; qi++) {
                        const cnt = RAW_COUNTS[pi][qi];
                        if (cnt === 0) continue;
                        chemArrow(CELL_NAMES[pi], CELL_NAMES[qi], cnt, (pi%2===0?-3:3));
                    }
                }

                const mLx = W/2 - 65, mRx = W/2 + 65;
                const mHy = pos['MOS_L'][1], mVy = pos['MOT_L'][1];
                function muscleBox(cx, cy, label, sublabel, subCol) {
                    svg += '<rect x="'+(cx-32)+'" y="'+(cy-14)+'" width="64" height="28" rx="5" fill="'+CBG+'" stroke="#ff8a65" stroke-width="1.5"/>';
                    svg += '<text x="'+cx+'" y="'+(cy-2)+'" text-anchor="middle" fill="#ff8a65" font-size="7.5" font-weight="bold">'+label+'</text>';
                    svg += '<text x="'+cx+'" y="'+(cy+9)+'" text-anchor="middle" fill="'+subCol+'" font-size="6">'+sublabel+'</text>';
                }
                muscleBox(mLx, mHy, 'L Musc \u2194', 'horiz (MOS)', CMOS);
                muscleBox(mLx, mVy, 'L Musc \u2195', 'vert (MOT)', CMOT);
                muscleBox(mRx, mHy, 'R Musc \u2194', 'horiz (MOS)', CMOS);
                muscleBox(mRx, mVy, 'R Musc \u2195', 'vert (MOT)', CMOT);

                ['L','R'].forEach(s => {
                    ['MOS','MOT'].forEach(mn => {
                        const name = mn+'_'+s, p1 = pos[name];
                        if (!p1) return;
                        const targX = s==='L' ? mLx : mRx;
                        const targY = mn==='MOS' ? mHy : mVy;
                        const bw = 72;
                        const x1 = s==='L' ? p1[0]+bw/2 : p1[0]-bw/2;
                        const x2 = s==='L' ? (targX-32) : (targX+32);
                        const aid = 'marr_'+name;
                        const active = isActive(name);
                        const col = active ? '#ff8a65' : '#455a64';
                        svg += '<defs><marker id="'+aid+'" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="'+col+'"/></marker></defs>';
                        svg += '<line x1="'+x1+'" y1="'+p1[1]+'" x2="'+x2+'" y2="'+targY+'" stroke="'+col+'" stroke-width="1.5" marker-end="url(#'+aid+')" opacity="'+(active ? '1' : '0.08')+'"/>';
                    });
                });

                const retY = 340;
                svg += '<rect x="'+(mLx-38)+'" y="'+(retY-12)+'" width="76" height="24" rx="4" fill="'+CBG+'" stroke="#26c6da" stroke-width="1.5" stroke-dasharray="4,2"/>';
                svg += '<text x="'+mLx+'" y="'+(retY+4)+'" text-anchor="middle" fill="#26c6da" font-size="8" font-weight="bold">L Eye / Retina</text>';
                svg += '<rect x="'+(mRx-38)+'" y="'+(retY-12)+'" width="76" height="24" rx="4" fill="'+CBG+'" stroke="#26c6da" stroke-width="1.5" stroke-dasharray="4,2"/>';
                svg += '<text x="'+mRx+'" y="'+(retY+4)+'" text-anchor="middle" fill="#26c6da" font-size="8" font-weight="bold">R Eye / Retina</text>';

                const retMkr = 'mret';
                svg += '<defs><marker id="'+retMkr+'" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#ff8a65"/></marker></defs>';
                [mLx, mRx].forEach(mx => {
                    svg += '<line x1="'+mx+'" y1="'+(mHy+14)+'" x2="'+mx+'" y2="'+(retY-12)+'" stroke="#ff8a65" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#'+retMkr+')" opacity="0.7"/>';
                    svg += '<line x1="'+mx+'" y1="'+(mVy+14)+'" x2="'+mx+'" y2="'+(retY-12)+'" stroke="#ff8a65" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#'+retMkr+')" opacity="0.7"/>';
                });

                const fbCol = '#26c6da', fbMkr = 'sfb';
                svg += '<defs><marker id="'+fbMkr+'" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="'+fbCol+'"/></marker></defs>';
                const hsBot = hs_ys[2]+16, vsBot = vs_ys[3]+16;
                svg += '<path d="M'+(mLx-38)+' '+retY+' Q '+lx_hs+' '+(retY+15)+' '+lx_hs+' '+hsBot+'" fill="none" stroke="'+lineColor(['HSN_L','HSE_L','HSS_L'], fbCol, '#455a64')+'" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#'+fbMkr+')" opacity="'+lineOpacity(['HSN_L','HSE_L','HSS_L'], 0.7, 0.08)+'"/>';
                svg += '<text x="'+((mLx-38+lx_hs)/2)+'" y="'+(retY+12)+'" text-anchor="middle" fill="'+lineColor(['HSN_L','HSE_L','HSS_L'], fbCol, '#7b8794')+'" font-size="5.5" opacity="'+lineOpacity(['HSN_L','HSE_L','HSS_L'], 0.7, 0.2)+'">horiz\u2192HS</text>';
                svg += '<path d="M'+(mLx-38)+' '+retY+' Q '+lx_vs+' '+(retY+25)+' '+lx_vs+' '+vsBot+'" fill="none" stroke="'+lineColor(['VS1_L','VS2_L','VS3_L','VS4_L'], fbCol, '#455a64')+'" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#'+fbMkr+')" opacity="'+lineOpacity(['VS1_L','VS2_L','VS3_L','VS4_L'], 0.7, 0.08)+'"/>';
                svg += '<text x="'+((mLx-38+lx_vs)/2)+'" y="'+(retY+22)+'" text-anchor="middle" fill="'+lineColor(['VS1_L','VS2_L','VS3_L','VS4_L'], fbCol, '#7b8794')+'" font-size="5.5" opacity="'+lineOpacity(['VS1_L','VS2_L','VS3_L','VS4_L'], 0.7, 0.2)+'">vert\u2192VS</text>';
                svg += '<path d="M'+(mRx+38)+' '+retY+' Q '+rx_hs+' '+(retY+15)+' '+rx_hs+' '+hsBot+'" fill="none" stroke="'+lineColor(['HSN_R','HSE_R','HSS_R'], fbCol, '#455a64')+'" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#'+fbMkr+')" opacity="'+lineOpacity(['HSN_R','HSE_R','HSS_R'], 0.7, 0.08)+'"/>';
                svg += '<text x="'+((mRx+38+rx_hs)/2)+'" y="'+(retY+12)+'" text-anchor="middle" fill="'+lineColor(['HSN_R','HSE_R','HSS_R'], fbCol, '#7b8794')+'" font-size="5.5" opacity="'+lineOpacity(['HSN_R','HSE_R','HSS_R'], 0.7, 0.2)+'">horiz\u2192HS</text>';
                svg += '<path d="M'+(mRx+38)+' '+retY+' Q '+rx_vs+' '+(retY+25)+' '+rx_vs+' '+vsBot+'" fill="none" stroke="'+lineColor(['VS1_R','VS2_R','VS3_R','VS4_R'], fbCol, '#455a64')+'" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#'+fbMkr+')" opacity="'+lineOpacity(['VS1_R','VS2_R','VS3_R','VS4_R'], 0.7, 0.08)+'"/>';
                svg += '<text x="'+((mRx+38+rx_vs)/2)+'" y="'+(retY+22)+'" text-anchor="middle" fill="'+lineColor(['VS1_R','VS2_R','VS3_R','VS4_R'], fbCol, '#7b8794')+'" font-size="5.5" opacity="'+lineOpacity(['VS1_R','VS2_R','VS3_R','VS4_R'], 0.7, 0.2)+'">vert\u2192VS</text>';

                svg += '<text x="'+W/2+'" y="'+(retY+38)+'" text-anchor="middle" fill="#26c6da" font-size="9" font-weight="bold" opacity="0.6">\u27f2 VISUOMOTOR FEEDBACK: muscles \u2192 retinal shift \u2192 visual scene \u2192 LPTC</text>';
                svg += '<text x="10" y="'+(H-54)+'" fill="#888" font-size="8">GJ (green dashed): LPTC chains = axo-axonal | LPTC\u2194MN = axon\u2194dendrite (bidirectional, LP-filtered)</text>';
                svg += '<text x="10" y="'+(H-42)+'" fill="#888" font-size="8">Chem syn: LPTC\u2194LPTC = dendro-dendritic | LPTC\u2192MN = axon\u2192dendrite | BIPS\u2192HS = axon\u2192dendrite</text>';
                svg += '<text x="10" y="'+(H-30)+'" fill="#888" font-size="8">Synapse kinetics: Graded (LPTC pre) / Alpha (MN pre) | E_rev: ACh/Glut=0mV, GABA=\u221280mV</text>';
                svg += '<text x="10" y="'+(H-18)+'" fill="#888" font-size="8">Orange = motor output to muscle | Cyan dashed = visuomotor feedback (muscle\u2192retina\u2192LPTC)</text>';
                svg += '<line x1="'+(W-200)+'" y1="'+(H-52)+'" x2="'+(W-170)+'" y2="'+(H-52)+'" stroke="'+CGJ+'" stroke-width="2" stroke-dasharray="5,3"/>';
                svg += '<text x="'+(W-165)+'" y="'+(H-49)+'" fill="'+CGJ+'" font-size="8">Bidirectional GJ</text>';
                svg += '<line x1="'+(W-200)+'" y1="'+(H-40)+'" x2="'+(W-170)+'" y2="'+(H-40)+'" stroke="'+CCHEM+'" stroke-width="2"/>';
                svg += '<text x="'+(W-165)+'" y="'+(H-37)+'" fill="'+CCHEM+'" font-size="8">Chemical synapse</text>';
                svg += '<line x1="'+(W-200)+'" y1="'+(H-28)+'" x2="'+(W-170)+'" y2="'+(H-28)+'" stroke="#ff8a65" stroke-width="2"/>';
                svg += '<text x="'+(W-165)+'" y="'+(H-25)+'" fill="#ff8a65" font-size="8">Motor output</text>';
                svg += '<line x1="'+(W-200)+'" y1="'+(H-16)+'" x2="'+(W-170)+'" y2="'+(H-16)+'" stroke="#26c6da" stroke-width="2" stroke-dasharray="4,3"/>';
                svg += '<text x="'+(W-165)+'" y="'+(H-13)+'" fill="#26c6da" font-size="8">Sensory feedback</text>';
                svg += '</svg>';
                const wiringEl = document.getElementById('wiringDiagram');
                wiringEl.innerHTML = svg;
                wiringEl.querySelectorAll('.circuit-node[data-neuron]').forEach(nodeEl => {
                    nodeEl.addEventListener('click', function() {
                        const name = this.dataset.neuron;
                        if (!name) return;
                        if (circuitDisabledNodes.has(name)) circuitDisabledNodes.delete(name);
                        else circuitDisabledNodes.add(name);
                        updateCircuitToggleStatus();
                        drawWiring();
                    });
                });
                updateCircuitToggleStatus();
            } catch(e) { console.error('drawWiring error:', e); }
            }

            drawWiring();

            // ── Wire up controls ──
            const stimGroupMap = {
                'VS_L':  ['VS1_L','VS2_L','VS3_L','VS4_L'],
                'VS_R':  ['VS1_R','VS2_R','VS3_R','VS4_R'],
                'HS_L':  ['HSN_L','HSE_L','HSS_L'],
                'HS_R':  ['HSN_R','HSE_R','HSS_R'],
                'VS_ALL':['VS1_L','VS2_L','VS3_L','VS4_L','VS1_R','VS2_R','VS3_R','VS4_R'],
                'HS_ALL':['HSN_L','HSE_L','HSS_L','HSN_R','HSE_R','HSS_R'],
                'ALL_L': ['VS1_L','VS2_L','VS3_L','VS4_L','HSN_L','HSE_L','HSS_L'],
                'ALL_R': ['VS1_R','VS2_R','VS3_R','VS4_R','HSN_R','HSE_R','HSS_R'],
                'MN_L':  ['MOS_L','MOT_L'],
                'MN_R':  ['MOS_R','MOT_R'],
            };

            function readParams() {
                const pf = (id, def) => { const v = parseFloat(document.getElementById(id).value); return isNaN(v) ? def : v; };
                // Multi-select: expand group keys to neuron lists, or use individual name directly
                const sel = document.getElementById('circStimGroup');
                stimTargets = Array.from(sel.selectedOptions)
                    .flatMap(o => stimGroupMap[o.value] || [o.value])
                    .filter((name, index, arr) => arr.indexOf(name) === index && !circuitDisabledNodes.has(name));
                stimAmp    = pf('circStimAmp',  10);
                simTime    = pf('circSimTime',  1500);
                stimStart  = pf('circStimStart', 90);
                stimEnd    = pf('circStimEnd',   590);
                noiseLevel = pf('circNoise',     3);
                // MOS parameters
                pMOS_VL    = pf('pMOS_VLm', -65);
                pMOS_GNaP  = pf('pMOS_GNaP', 0.5);
                pMOS_GNa   = pf('pMOS_GNa', 120);
                pMOS_GK    = pf('pMOS_GKm', 36);
                pMOS_GL    = pf('pMOS_GLm', 0.3);
                pMOS_Rin   = pf('pMOS_RinM', 300);
                pMOS_Ibias = pf('pMOS_IbiasM', 0);
                pMOS_GVT   = pf('pMOS_GVTm', 0.0);
                // MOT parameters
                pMOT_VL    = pf('pMOT_VLm', -65);
                pMOT_GNaP  = pf('pMOT_GNaP', 0.5);
                pMOT_GNa   = pf('pMOT_GNa', 120);
                pMOT_GK    = pf('pMOT_GKm', 36);
                pMOT_GL    = pf('pMOT_GLm', 0.3);
                pMOT_Rin   = pf('pMOT_RinM', 300);
                pMOT_Ibias = pf('pMOT_IbiasM', 0);
                pMOT_GVT   = pf('pMOT_GVTm', 0.0);
                // VS parameters
                pVS_Vr1    = pf('pVS_Vr1', -40);
                pVS_VrStep = pf('pVS_VrStep', -5);
                pVS_Rin1   = pf('pVS_Rin1', 150);
                pVS_RinStep= pf('pVS_RinStep', -10);
                pVS_GVT    = pf('pVS_GVTl', 0.5);
                pVS_GK     = pf('pVS_GKl', 2.0);
                pVS_GL     = pf('pVS_GLl', 0.05);
                // HS parameters
                pHS_Vr     = pf('pHS_Vr', -45);
                pHS_Rin    = pf('pHS_Rin', 150);
                pHS_GVT    = pf('pHS_GVTl', 0.5);
                pHS_GK     = pf('pHS_GKl', 2.0);
                pHS_GL     = pf('pHS_GLl', 0.05);
                // GJ params
                pGlptc   = pf('pGlptc',   0.05);
                pClptc   = pf('pClptc',   0.05);
                pGvsmos  = pf('pGvsmos',  0.1);
                pGhsmos  = pf('pGhsmos',  0.1);
                pGhsmot  = pf('pGhsmot',  0.1);
                pCmn     = pf('pCmn',     0.8);
                // Synapse params
                pGgradExc  = pf('pGgradExc',  0.005);
                pGgradInh  = pf('pGgradInh',  0.004);
                pGspikeExc = pf('pGspikeExc', 0.02);
                pGspikeInh = pf('pGspikeInh', 0.016);
                pTauSyn  = pf('pTauSyn',  5);
                pVthresh = pf('pVthresh', -40);
                pVscale  = pf('pVscale',   20);
                // Pseudopupil analysis
                pupilBaselineStart    = pf('pupilBaselineStart', 0);
                pupilBaselineWindow   = pf('pupilBaselineWindow', 90);
                pupilDisableJitter    = document.getElementById('pupilDisableJitter').checked;
                pupilUseRawRate       = document.getElementById('pupilUseRawRate').checked;
            }

            function applyMotorTargetPreset() {
                const setVal = (id, v) => {
                    const el = document.getElementById(id);
                    if (el) el.value = String(v);
                };

                // MOT: baseline around 120 Hz, mostly release-directed, capped near 140 Hz in typical runs.
                setVal('pMOT_VLm', -62);
                setVal('pMOT_GNaP', 0.95);
                setVal('pMOT_GNa', 145);
                setVal('pMOT_GKm', 58);
                setVal('pMOT_GLm', 0.42);
                setVal('pMOT_RinM', 260);
                setVal('pMOT_IbiasM', 3.6);
                setVal('pMOT_GVTm', 0.0);

                // MOS: baseline around 100 Hz with larger dynamic range up to about 200 Hz.
                setVal('pMOS_VLm', -63);
                setVal('pMOS_GNaP', 1.45);
                setVal('pMOS_GNa', 170);
                setVal('pMOS_GKm', 46);
                setVal('pMOS_GLm', 0.28);
                setVal('pMOS_RinM', 360);
                setVal('pMOS_IbiasM', 4.2);
                setVal('pMOS_GVTm', 0.0);

                // Cleaner pseudopupil inspection of true circuit motion.
                setVal('pupilBaselineStart', 0);
                setVal('pupilBaselineWindow', 140);
                const jitter = document.getElementById('pupilDisableJitter');
                if (jitter) jitter.checked = true;
                const rawRate = document.getElementById('pupilUseRawRate');
                if (rawRate) rawRate.checked = false;
            }

            function estimateRestRateHz(vm, tArr, tEndMs) {
                // Lightweight spike-rate estimator for pre-stim resting window.
                const dtMs = (tArr.length > 1) ? (tArr[1] - tArr[0]) : 0.01;
                const thr = -15;
                const refractory = Math.max(1, Math.round(4.0 / dtMs));
                let lastSpike = -refractory;
                let spikes = 0;
                let nSamples = 0;
                for (let i = 1; i < vm.length; i++) {
                    if (tArr[i] >= tEndMs) break;
                    nSamples++;
                    if (vm[i - 1] < thr && vm[i] >= thr && (i - lastSpike) >= refractory) {
                        spikes++;
                        lastSpike = i;
                    }
                }
                const durS = Math.max(1e-6, (nSamples * dtMs) / 1000.0);
                return spikes / durS;
            }

            function calibrateRestHzTargets() {
                const tMOT = parseFloat(document.getElementById('pMOT_TargetRestHz').value);
                const tMOS = parseFloat(document.getElementById('pMOS_TargetRestHz').value);
                const targetMOT = isNaN(tMOT) ? 120 : tMOT;
                const targetMOS = isNaN(tMOS) ? 100 : tMOS;
                const ibMotEl = document.getElementById('pMOT_IbiasM');
                const ibMosEl = document.getElementById('pMOS_IbiasM');
                const gGradInhEl = document.getElementById('pGgradInh');
                const gSpikeInhEl = document.getElementById('pGspikeInh');
                const vlMotEl = document.getElementById('pMOT_VLm');
                const vlMosEl = document.getElementById('pMOS_VLm');
                const gnapMotEl = document.getElementById('pMOT_GNaP');
                const gnapMosEl = document.getElementById('pMOS_GNaP');
                const gkMotEl = document.getElementById('pMOT_GKm');
                const gkMosEl = document.getElementById('pMOS_GKm');
                if (!ibMotEl || !ibMosEl || !vlMotEl || !vlMosEl || !gnapMotEl || !gnapMosEl || !gkMotEl || !gkMosEl) return;

                const keepGradInh = gGradInhEl ? gGradInhEl.value : null;
                const keepSpikeInh = gSpikeInhEl ? gSpikeInhEl.value : null;

                let ibMot = parseFloat(ibMotEl.value);
                let ibMos = parseFloat(ibMosEl.value);
                let vlMot = parseFloat(vlMotEl.value);
                let vlMos = parseFloat(vlMosEl.value);
                let gnapMot = parseFloat(gnapMotEl.value);
                let gnapMos = parseFloat(gnapMosEl.value);
                let gkMot = parseFloat(gkMotEl.value);
                let gkMos = parseFloat(gkMosEl.value);
                if (isNaN(ibMot)) ibMot = 0;
                if (isNaN(ibMos)) ibMos = 0;
                if (isNaN(vlMot)) vlMot = -65;
                if (isNaN(vlMos)) vlMos = -65;
                if (isNaN(gnapMot)) gnapMot = 0.5;
                if (isNaN(gnapMos)) gnapMos = 0.5;
                if (isNaN(gkMot)) gkMot = 36;
                if (isNaN(gkMos)) gkMos = 36;

                // Step 1 (option 4): fit intrinsic excitability first, with near-zero tonic bias.
                ibMot = Math.max(-2, Math.min(2, ibMot));
                ibMos = Math.max(-2, Math.min(2, ibMos));
                for (let iter = 0; iter < 10; iter++) {
                    vlMotEl.value = String(vlMot);
                    vlMosEl.value = String(vlMos);
                    gnapMotEl.value = String(gnapMot);
                    gnapMosEl.value = String(gnapMos);
                    gkMotEl.value = String(gkMot);
                    gkMosEl.value = String(gkMos);
                    ibMotEl.value = String(ibMot);
                    ibMosEl.value = String(ibMos);

                    readParams();
                    const res = buildAndRun();
                    const tArr = Array.from(res.t);
                    const motL = estimateRestRateHz(res.records['MOT_L'], tArr, simPreRollMs);
                    const motR = estimateRestRateHz(res.records['MOT_R'], tArr, simPreRollMs);
                    const mosL = estimateRestRateHz(res.records['MOS_L'], tArr, simPreRollMs);
                    const mosR = estimateRestRateHz(res.records['MOS_R'], tArr, simPreRollMs);
                    const motHz = 0.5 * (motL + motR);
                    const mosHz = 0.5 * (mosL + mosR);

                    const eMot = targetMOT - motHz;
                    const eMos = targetMOS - mosHz;
                    if (Math.abs(eMot) < 4 && Math.abs(eMos) < 4) break;

                    // Intrinsic-first updates:
                    // More depolarized VL, larger gNaP, smaller gK -> higher spontaneous firing.
                    vlMot += 0.012 * eMot;
                    vlMos += 0.012 * eMos;
                    gnapMot += 0.0030 * eMot;
                    gnapMos += 0.0030 * eMos;
                    gkMot -= 0.020 * eMot;
                    gkMos -= 0.020 * eMos;

                    vlMot = Math.max(-80, Math.min(-45, vlMot));
                    vlMos = Math.max(-80, Math.min(-45, vlMos));
                    gnapMot = Math.max(0.0, Math.min(3.0, gnapMot));
                    gnapMos = Math.max(0.0, Math.min(3.0, gnapMos));
                    gkMot = Math.max(8, Math.min(150, gkMot));
                    gkMos = Math.max(8, Math.min(150, gkMos));
                }

                // Step 2: small residual Ibias correction only.
                for (let iter = 0; iter < 4; iter++) {
                    vlMotEl.value = String(vlMot);
                    vlMosEl.value = String(vlMos);
                    gnapMotEl.value = String(gnapMot);
                    gnapMosEl.value = String(gnapMos);
                    gkMotEl.value = String(gkMot);
                    gkMosEl.value = String(gkMos);
                    ibMotEl.value = String(ibMot);
                    ibMosEl.value = String(ibMos);

                    readParams();
                    const res = buildAndRun();
                    const tArr = Array.from(res.t);
                    const motL = estimateRestRateHz(res.records['MOT_L'], tArr, simPreRollMs);
                    const motR = estimateRestRateHz(res.records['MOT_R'], tArr, simPreRollMs);
                    const mosL = estimateRestRateHz(res.records['MOS_L'], tArr, simPreRollMs);
                    const mosR = estimateRestRateHz(res.records['MOS_R'], tArr, simPreRollMs);
                    const motHz = 0.5 * (motL + motR);
                    const mosHz = 0.5 * (mosL + mosR);
                    const eMot = targetMOT - motHz;
                    const eMos = targetMOS - mosHz;
                    if (Math.abs(eMot) < 2 && Math.abs(eMos) < 2) break;

                    ibMot += 0.012 * eMot;
                    ibMos += 0.012 * eMos;
                    ibMot = Math.max(-6, Math.min(6, ibMot));
                    ibMos = Math.max(-6, Math.min(6, ibMos));
                }

                vlMotEl.value = vlMot.toFixed(2);
                vlMosEl.value = vlMos.toFixed(2);
                gnapMotEl.value = gnapMot.toFixed(3);
                gnapMosEl.value = gnapMos.toFixed(3);
                gkMotEl.value = gkMot.toFixed(2);
                gkMosEl.value = gkMos.toFixed(2);
                ibMotEl.value = ibMot.toFixed(2);
                ibMosEl.value = ibMos.toFixed(2);
                if (gGradInhEl && keepGradInh !== null) gGradInhEl.value = keepGradInh;
                if (gSpikeInhEl && keepSpikeInh !== null) gSpikeInhEl.value = keepSpikeInh;
                readParams();
                const res = buildAndRun();
                plotResults(res);

                const tArr = Array.from(res.t);
                const restEndMs = res.preRollMs || simPreRollMs;
                const motHz = 0.5 * (
                    estimateRestRateHz(res.records['MOT_L'], tArr, restEndMs) +
                    estimateRestRateHz(res.records['MOT_R'], tArr, restEndMs)
                );
                const mosHz = 0.5 * (
                    estimateRestRateHz(res.records['MOS_L'], tArr, restEndMs) +
                    estimateRestRateHz(res.records['MOS_R'], tArr, restEndMs)
                );
                const status = document.getElementById('circCalibStatus');
                if (status) {
                    const dMot = motHz - targetMOT;
                    const dMos = mosHz - targetMOS;
                    status.innerHTML =
                        'Calibration report: ' +
                        'MOT ' + motHz.toFixed(1) + ' Hz (target ' + targetMOT.toFixed(1) + ', err ' + dMot.toFixed(1) + ') | ' +
                        'MOS ' + mosHz.toFixed(1) + ' Hz (target ' + targetMOS.toFixed(1) + ', err ' + dMos.toFixed(1) + ')<br>' +
                        'Fitted: MOT [VL=' + vlMot.toFixed(2) + ', gNaP=' + gnapMot.toFixed(3) + ', gK=' + gkMot.toFixed(2) + ', Ibias=' + ibMot.toFixed(2) + '] ; ' +
                        'MOS [VL=' + vlMos.toFixed(2) + ', gNaP=' + gnapMos.toFixed(3) + ', gK=' + gkMos.toFixed(2) + ', Ibias=' + ibMos.toFixed(2) + ']';
                }
            }

            document.getElementById('circResetNodes').addEventListener('click', function() {
                circuitDisabledNodes.clear();
                updateCircuitToggleStatus();
                drawWiring();
            });

            document.getElementById('circPresetMotor').addEventListener('click', function() {
                applyMotorTargetPreset();
            });

            document.getElementById('circCalibrateRestHz').addEventListener('click', function() {
                const prev = this.textContent;
                const status = document.getElementById('circCalibStatus');
                this.textContent = '\u23f3 Calibrating...';
                this.disabled = true;
                if (status) status.textContent = 'Calibration running...';
                setTimeout(() => {
                    calibrateRestHzTargets();
                    this.textContent = prev;
                    this.disabled = false;
                }, 30);
            });

            document.getElementById('circRun').addEventListener('click', function() {
                readParams();
                this.textContent = '\u23f3 Running...'; this.disabled = true;
                setTimeout(() => {
                    const res = buildAndRun();
                    plotResults(res);
                    this.textContent = '\u25b6 Run'; this.disabled = false;
                }, 30);
            });

            function plotResults(res) {
                const tMs = Array.from(res.t);
                const preRollMs = res.preRollMs || simPreRollMs;
                let step = 1;
                if (tMs.length > 5000) step = Math.ceil(tMs.length / 5000);
                const tPlot = [], indices = [];
                for (let i = 0; i < tMs.length; i += step) {
                    if (tMs[i] < preRollMs) continue;
                    tPlot.push(tMs[i] - preRollMs);
                    indices.push(i);
                }

                // LPTC traces
                const lptcOrder = [];
                ['L','R'].forEach(s => {
                    for (let k = 1; k <= 4; k++) lptcOrder.push('VS'+k+'_'+s);
                    ['HSN','HSE','HSS'].forEach(h => lptcOrder.push(h+'_'+s));
                });
                // Use mesh colors from neuronColors (injected from neurons.json)
                // so simulation traces match the 3D viewer colors exactly.
                function addStimTopLine() {
                    return {
                        type: 'line', xref: 'x', yref: 'paper',
                        x0: stimStart, x1: stimEnd,
                        y0: 1.0, y1: 1.0,
                        line: { color: '#ff6f00', width: 3 },
                    };
                }
                function noGridAxis(cfg) {
                    return Object.assign({}, cfg, { showgrid: false, zeroline: false });
                }
                function noGridAxis2(cfg) {
                    return Object.assign({}, cfg, { showgrid: false });
                }
                function addStimMarkerTrace() {
                    return {
                        x: [stimStart, stimEnd],
                        y: [null, null],
                        line: { width: 0 },
                        showlegend: false, hoverinfo: 'skip',
                        type: 'scatter', mode: 'lines',
                    };
                }
                const lptcLeft = lptcOrder.filter(n => n.endsWith('_L')).map(n => ({
                    x: tPlot, y: indices.map(i => res.records[n][i]),
                    name: n, type: 'scatter', mode: 'lines',
                    line: { color: neuronColors[n] || '#888', width: 1 },
                }));
                const lptcRight = lptcOrder.filter(n => n.endsWith('_R')).map(n => ({
                    x: tPlot, y: indices.map(i => res.records[n][i]),
                    name: n, type: 'scatter', mode: 'lines',
                    line: { color: neuronColors[n] || '#888', width: 1 },
                }));
                lptcLeft.push(addStimMarkerTrace());
                lptcRight.push(addStimMarkerTrace());
                Plotly.react('circPlotLPTC_L', lptcLeft, {
                    title: { text: 'Left LPTCs — soma Vm', font: { size: 11, color: '#333' } },
                    xaxis: noGridAxis({ title: 'ms', color: '#444' }),
                    yaxis: noGridAxis({ title: 'mV', color: '#444', range: [-80, 20] }),
                    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                    legend: { font: { size: 7, color: '#333' }, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ccc', borderwidth: 1 },
                    shapes: [addStimTopLine()],
                    margin: { l: 40, r: 10, t: 26, b: 30 },
                }, { responsive: true });
                Plotly.react('circPlotLPTC_R', lptcRight, {
                    title: { text: 'Right LPTCs — soma Vm', font: { size: 11, color: '#333' } },
                    xaxis: noGridAxis({ title: 'ms', color: '#444' }),
                    yaxis: noGridAxis({ title: 'mV', color: '#444', range: [-80, 20] }),
                    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                    legend: { font: { size: 7, color: '#333' }, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ccc', borderwidth: 1 },
                    shapes: [addStimTopLine()],
                    margin: { l: 40, r: 10, t: 26, b: 30 },
                }, { responsive: true });

                // Use mesh colors from neuronColors for motor neuron traces
                const mnLeft = ['MOS_L','MOT_L'].map(n => ({
                    x: tPlot, y: indices.map(i => res.records[n][i]),
                    name: n, type: 'scatter', mode: 'lines',
                    line: { color: neuronColors[n] || '#888', width: 1.2 },
                }));
                const mnRight = ['MOS_R','MOT_R'].map(n => ({
                    x: tPlot, y: indices.map(i => res.records[n][i]),
                    name: n, type: 'scatter', mode: 'lines',
                    line: { color: neuronColors[n] || '#888', width: 1.2 },
                }));
                mnLeft.push(addStimMarkerTrace());
                mnRight.push(addStimMarkerTrace());
                Plotly.react('circPlotMN_L', mnLeft, {
                    title: { text: 'Left Motor Neurons — soma Vm', font: { size: 11, color: '#333' } },
                    xaxis: noGridAxis({ title: 'ms', color: '#444' }),
                    yaxis: noGridAxis({ title: 'mV', color: '#444', range: [-80, 60] }),
                    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                    legend: { font: { size: 9, color: '#333' }, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ccc', borderwidth: 1 },
                    shapes: [addStimTopLine()],
                    margin: { l: 40, r: 10, t: 26, b: 30 },
                }, { responsive: true });
                Plotly.react('circPlotMN_R', mnRight, {
                    title: { text: 'Right Motor Neurons — soma Vm', font: { size: 11, color: '#333' } },
                    xaxis: noGridAxis({ title: 'ms', color: '#444' }),
                    yaxis: noGridAxis({ title: 'mV', color: '#444', range: [-80, 60] }),
                    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                    legend: { font: { size: 9, color: '#333' }, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ccc', borderwidth: 1 },
                    shapes: [addStimTopLine()],
                    margin: { l: 40, r: 10, t: 26, b: 30 },
                }, { responsive: true });

                // Toy motor-output model: robust rate extraction + smoothed pupil kinematics.
                function lowpass1(v, dtMs, tauMs) {
                    const out = new Float32Array(v.length);
                    if (!v.length) return out;
                    out[0] = v[0];
                    const a = dtMs / Math.max(dtMs, tauMs);
                    for (let i = 1; i < v.length; i++) out[i] = out[i - 1] + a * (v[i] - out[i - 1]);
                    return out;
                }
                function median(arr) {
                    if (!arr.length) return 0;
                    const s = arr.slice().sort((a, b) => a - b);
                    const m = Math.floor(s.length / 2);
                    return (s.length % 2) ? s[m] : 0.5 * (s[m - 1] + s[m]);
                }
                function quantile(arr, q) {
                    if (!arr.length) return 0;
                    const s = arr.slice().sort((a, b) => a - b);
                    const p = Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))));
                    return s[p];
                }
                function robustSpikeRateSeries(v, t, winMs) {
                    const dtMs = (t.length > 1) ? (t[1] - t[0]) : 0.01;
                    const vf = lowpass1(v, dtMs, 0.9);    // keep spike crest shape
                    const vSlow = lowpass1(vf, dtMs, 35.0);
                    const hp = new Float32Array(v.length);
                    const hpAbs = [];
                    for (let i = 0; i < v.length; i++) {
                        hp[i] = vf[i] - vSlow[i];
                        hpAbs.push(Math.abs(hp[i]));
                    }

                    const q10 = quantile(Array.from(vf), 0.10);
                    const q90 = quantile(Array.from(vf), 0.90);
                    const span = Math.max(2.0, q90 - q10);
                    const thr = q10 + 0.52 * span;

                    const dv = new Float32Array(v.length);
                    for (let i = 1; i < v.length; i++) dv[i] = (vf[i] - vf[i - 1]) / dtMs;
                    const dvThr = Math.max(0.25, 0.85 * quantile(Array.from(dv).map(x => Math.abs(x)), 0.80));
                    const hpThr = Math.max(0.2, 0.9 * quantile(hpAbs, 0.70));

                    const spikes = new Float32Array(v.length);
                    const refractory = Math.max(1, Math.round(4.5 / dtMs));
                    let lastSpike = -refractory;

                    for (let i = 2; i < v.length - 2; i++) {
                        const upwardCross = (vf[i - 1] < thr && vf[i] >= thr);
                        const localPeak = (vf[i] >= vf[i - 1] && vf[i] > vf[i + 1]);
                        const steepEnough = (dv[i] > dvThr || dv[i - 1] > dvThr);
                        const prominent = hp[i] > hpThr;
                        if (!(upwardCross || (localPeak && steepEnough && prominent))) continue;
                        if (i - lastSpike < refractory) continue;
                        spikes[i] = 1;
                        lastSpike = i;
                    }

                    const half = Math.max(1, Math.round((winMs / dtMs) / 2));
                    const prefix = new Float32Array(v.length + 1);
                    for (let i = 0; i < v.length; i++) prefix[i + 1] = prefix[i] + spikes[i];
                    const hzRaw = new Float32Array(v.length);
                    for (let i = 0; i < v.length; i++) {
                        const a = Math.max(0, i - half), b = Math.min(v.length - 1, i + half);
                        const nsp = prefix[b + 1] - prefix[a];
                        const durS = Math.max(1e-6, (b - a + 1) * dtMs / 1000);
                        hzRaw[i] = nsp / durS;
                    }
                    return lowpass1(hzRaw, dtMs, 20.0);
                }
                function meanPre(arr, t, t0, duration) {
                    // Modified: if duration is provided, use sliding window [t0-duration, t0]
                    let s = 0, c = 0;
                    const t_min = duration ? (t0 - duration) : -Infinity;
                    for (let i = 0; i < arr.length; i++) if (t[i] >= t_min && t[i] < t0) { s += arr[i]; c++; }
                    return c ? s / c : 0;
                }
                function satPull(d, scale) {
                    if (d <= 0) return 0;
                    return 8 * (d / (d + scale));
                }
                function satRelease(d, scale) {
                    if (d <= 0) return 0;
                    return 8 * (d / (d + scale));
                }
                function addPolar(vec, mag, thetaDeg) {
                    const th = thetaDeg * Math.PI / 180;
                    vec.x += mag * Math.cos(th);
                    vec.y += mag * Math.sin(th);
                }
                function mirrorRightToLeft(thetaDeg) {
                    return (180 - thetaDeg + 360) % 360;
                }
                function buildPupilMotion(sideTag, mosRate, motRate) {
                    // Right-eye calibration (deg): 0° = rightward/front for right eye.
                    const rightCal = {
                        mosPull: 50.582827,
                        mosRelease: 204.228953,
                        bothPull: 354.23147,
                        bothRelease: 161.518836,
                        motRelease: 116.555959,
                        motPull: 320.258203,
                    };
                    const cal = (sideTag === 'R') ? rightCal : {
                        mosPull: mirrorRightToLeft(rightCal.mosPull),
                        mosRelease: mirrorRightToLeft(rightCal.mosRelease),
                        bothPull: mirrorRightToLeft(rightCal.bothPull),
                        bothRelease: mirrorRightToLeft(rightCal.bothRelease),
                        motRelease: mirrorRightToLeft(rightCal.motRelease),
                        motPull: mirrorRightToLeft(rightCal.motPull),
                    };
                    // Use user-defined baseline window (start offset + duration)
                    const baselineEndTime = preRollMs + stimStart + pupilBaselineStart;
                    const bMos = meanPre(mosRate, tMs, baselineEndTime, pupilBaselineWindow);
                    const bMot = meanPre(motRate, tMs, baselineEndTime, pupilBaselineWindow);
                    const DEAD = 5.0;  // Hz dead-zone: ignore sub-threshold fluctuations
                    const x = new Float32Array(mosRate.length);
                    const y = new Float32Array(mosRate.length);
                    const r = new Float32Array(mosRate.length);
                    const th = new Float32Array(mosRate.length);
                    const mosX = new Float32Array(mosRate.length), mosY = new Float32Array(mosRate.length);
                    const motX = new Float32Array(mosRate.length), motY = new Float32Array(mosRate.length);
                    for (let i = 0; i < mosRate.length; i++) {
                        // Option: use raw rate or baseline-subtracted
                        const dMos = pupilUseRawRate ? mosRate[i] : (mosRate[i] - bMos);
                        const dMot = pupilUseRawRate ? motRate[i] : (motRate[i] - bMot);
                        const rawDMos = dMos, rawDMot = dMot;
                        // Dead-zone: treat small fluctuations as zero
                        const dMosDZ = Math.abs(rawDMos) < DEAD ? 0 : rawDMos - Math.sign(rawDMos) * DEAD;
                        const dMotDZ = Math.abs(rawDMot) < DEAD ? 0 : rawDMot - Math.sign(rawDMot) * DEAD;

                        const vec = { x: 0, y: 0 };
                        const vecMos = { x: 0, y: 0 };
                        const vecMot = { x: 0, y: 0 };

                        // Individual component pulls/releases
                        addPolar(vecMos, satPull(Math.max(0, dMosDZ), 40), cal.mosPull);
                        addPolar(vecMos, satRelease(Math.max(0, -dMosDZ), 26), cal.mosRelease);
                        addPolar(vecMot, satPull(Math.max(0, dMotDZ), 40), cal.motPull);
                        addPolar(vecMot, satRelease(Math.max(0, -dMotDZ), 26), cal.motRelease);
                        vec.x += vecMos.x + vecMot.x;
                        vec.y += vecMos.y + vecMot.y;

                        // Cooperative term when both rise/fall together
                        const bothUp = Math.min(Math.max(0, dMosDZ), Math.max(0, dMotDZ));
                        const bothDown = Math.min(Math.max(0, -dMosDZ), Math.max(0, -dMotDZ));
                        addPolar(vec, satPull(bothUp, 50), cal.bothPull);
                        addPolar(vec, satRelease(bothDown, 34), cal.bothRelease);

                        // Spike-like jitter: converts rate transients into small scanpath wiggles (can be disabled).
                        if (!pupilDisableJitter) {
                            const dMosRate = (i > 0) ? Math.abs(mosRate[i] - mosRate[i - 1]) : 0;
                            const dMotRate = (i > 0) ? Math.abs(motRate[i] - motRate[i - 1]) : 0;
                            const transient = Math.min(1.0, (dMosRate + dMotRate) / 18.0);
                            const drive = Math.min(1.0, (Math.abs(dMosDZ) + Math.abs(dMotDZ)) / 70.0);
                            const jitterMag = 0.55 * transient * (0.3 + 0.7 * drive);
                            if (jitterMag > 1e-4) {
                                const seed = (sideTag === 'R') ? 0.91 : 0.37;
                                const jitterTheta = (180 / Math.PI) * (
                                    2.1 * Math.sin(0.33 * i + seed) +
                                    1.3 * Math.sin(0.91 * i + 0.7 + seed)
                                );
                                addPolar(vec, jitterMag, (jitterTheta + 360) % 360);
                            }
                        }

                        let px = vec.x;
                        let py = vec.y;
                        let rr = Math.sqrt(px * px + py * py);
                        if (rr > 10) { px *= 10 / rr; py *= 10 / rr; rr = 10; }
                        x[i] = px; y[i] = py; r[i] = rr;
                        mosX[i] = vecMos.x; mosY[i] = vecMos.y;
                        motX[i] = vecMot.x; motY[i] = vecMot.y;
                        th[i] = (Math.atan2(py, px) * 180 / Math.PI + 360) % 360;
                    }
                    return { x, y, r, th, mosX, mosY, motX, motY };
                }
                function overallVectorFromXY(xArr, yArr) {
                    const n = xArr.length;
                    const k = Math.max(5, Math.floor(0.1 * n));
                    let x0 = 0, y0 = 0, x1 = 0, y1 = 0;
                    for (let i = 0; i < k; i++) { x0 += xArr[i]; y0 += yArr[i]; }
                    for (let i = n - k; i < n; i++) { x1 += xArr[i]; y1 += yArr[i]; }
                    x0 /= k; y0 /= k; x1 /= k; y1 /= k;
                    const dx = x1 - x0, dy = y1 - y0;
                    return {
                        theta: (Math.atan2(dy, dx) * 180 / Math.PI + 360) % 360,
                        r: Math.min(10, Math.sqrt(dx * dx + dy * dy))
                    };
                }
                function overallVector(move) { return overallVectorFromXY(move.x, move.y); }
                // For component-force arrows (MOS/MOT): use the mean of all active samples so a
                // sustained pull shows a full-length arrow rather than a near-zero delta.
                function meanVectorFromXY(xArr, yArr) {
                    let sx = 0, sy = 0, n = 0;
                    for (let i = 0; i < xArr.length; i++) {
                        if (Math.abs(xArr[i]) + Math.abs(yArr[i]) > 0.02) {
                            sx += xArr[i]; sy += yArr[i]; n++;
                        }
                    }
                    if (n === 0) return { theta: 0, r: 0 };
                    sx /= n; sy /= n;
                    return {
                        theta: (Math.atan2(sy, sx) * 180 / Math.PI + 360) % 360,
                        r: Math.min(10, Math.sqrt(sx * sx + sy * sy))
                    };
                }
                // For overall eye direction, use the mean pupil displacement during the stim window.
                // This matches the intuitive "where the eye moved" direction in the polar plot.
                function windowMeanVectorFromXY(xArr, yArr, tArr, t0, t1) {
                    let sx = 0, sy = 0, n = 0;
                    for (let i = 0; i < xArr.length; i++) {
                        if (tArr[i] >= t0 && tArr[i] <= t1) {
                            sx += xArr[i];
                            sy += yArr[i];
                            n++;
                        }
                    }
                    if (n === 0) return overallVectorFromXY(xArr, yArr);
                    sx /= n; sy /= n;
                    return {
                        theta: (Math.atan2(sy, sx) * 180 / Math.PI + 360) % 360,
                        r: Math.min(10, Math.sqrt(sx * sx + sy * sy))
                    };
                }
                // Build an arrow entirely inside polar coordinates: shaft line + rotated triangle head + text label.
                // Marker angle: Plotly uses clockwise-from-north; CCW polar theta → (90 − theta + 360) % 360
                function makePolarArrow(subplot, vec, color, label, showlegend) {
                    const mAngle = (90 - vec.theta + 360) % 360;
                    const labelR = Math.min(10.5, vec.r + 1.2);
                                        const hasLabel = !!(label && label.trim());
                                        const labelName = hasLabel ? label : 'Vector';
                                        const traces = [
                        // ① Shaft
                        { type: 'scatterpolar', subplot,
                          theta: [vec.theta, vec.theta], r: [0, vec.r],
                                                    mode: 'lines', name: labelName, showlegend,
                          line: { color, width: 3 },
                                                    hovertemplate: labelName + ': %{r:.2f} at %{theta:.1f}<extra></extra>' },
                        // ② Arrowhead — triangle-up rotated to match the shaft direction
                        { type: 'scatterpolar', subplot,
                          theta: [vec.theta], r: [vec.r],
                          mode: 'markers', showlegend: false,
                          marker: { color, size: 13, symbol: 'triangle-up', angle: mAngle },
                          hoverinfo: 'skip' },
                    ];
                                        if (hasLabel) {
                                                // ③ Text label slightly beyond the tip
                                                traces.push({ type: 'scatterpolar', subplot,
                                                        theta: [vec.theta], r: [labelR],
                                                        mode: 'text', text: [label], showlegend: false,
                                                        textfont: { color, size: 9, family: 'Arial, sans-serif' },
                                                        hoverinfo: 'skip' });
                                        }
                                        return traces;
                }

                                const rateMOS_L = robustSpikeRateSeries(res.records['MOS_L'], tMs, 70);
                                const rateMOS_R = robustSpikeRateSeries(res.records['MOS_R'], tMs, 70);
                                const rateMOT_L = robustSpikeRateSeries(res.records['MOT_L'], tMs, 70);
                                const rateMOT_R = robustSpikeRateSeries(res.records['MOT_R'], tMs, 70);
                const moveL = buildPupilMotion('L', rateMOS_L, rateMOT_L);
                const moveR = buildPupilMotion('R', rateMOS_R, rateMOT_R);
                const stimStartAbs = preRollMs + stimStart;
                const stimEndAbs = preRollMs + stimEnd;
                const vecL = windowMeanVectorFromXY(moveL.x, moveL.y, tMs, stimStartAbs, stimEndAbs);
                const vecR = windowMeanVectorFromXY(moveR.x, moveR.y, tMs, stimStartAbs, stimEndAbs);
                const mosVecL = meanVectorFromXY(moveL.mosX, moveL.mosY), mosVecR = meanVectorFromXY(moveR.mosX, moveR.mosY);
                const motVecL = meanVectorFromXY(moveL.motX, moveL.motY), motVecR = meanVectorFromXY(moveR.motX, moveR.motY);

                Plotly.react('circPlotPupilTime_L', [
                    { x: tPlot, y: indices.map(i => rateMOS_L[i]), name: 'MOS_L rate', type: 'scatter', mode: 'lines', line: { color: neuronColors['MOS_L'] || '#4D9221', width: 1.1 } },
                    { x: tPlot, y: indices.map(i => rateMOT_L[i]), name: 'MOT_L rate', type: 'scatter', mode: 'lines', line: { color: neuronColors['MOT_L'] || '#5E3C99', width: 1.1 } },
                    { x: tPlot, y: indices.map(i => moveL.r[i]), name: 'Left pupil |Δ|', type: 'scatter', mode: 'lines', yaxis: 'y2', line: { color: '#222222', width: 1.6 } },
                    addStimMarkerTrace()
                ], {
                    title: { text: 'Left Pseudopupil Output', font: { size: 11, color: '#333' } },
                    xaxis: noGridAxis({ title: 'ms', color: '#444', range: [0, simTime] }),
                    yaxis: noGridAxis({ title: 'Hz', color: '#444', rangemode: 'tozero' }),
                    yaxis2: noGridAxis2({ title: 'deg', overlaying: 'y', side: 'right', color: '#00695c', range: [0, 10] }),
                    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                    legend: { font: { size: 8, color: '#333' }, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ccc', borderwidth: 1 },
                    shapes: [addStimTopLine()],
                    margin: { l: 38, r: 44, t: 24, b: 28 },
                }, { responsive: true });
                Plotly.react('circPlotPupilTime_R', [
                    { x: tPlot, y: indices.map(i => rateMOS_R[i]), name: 'MOS_R rate', type: 'scatter', mode: 'lines', line: { color: neuronColors['MOS_R'] || '#61A635', width: 1.1, dash: 'dot' } },
                    { x: tPlot, y: indices.map(i => rateMOT_R[i]), name: 'MOT_R rate', type: 'scatter', mode: 'lines', line: { color: neuronColors['MOT_R'] || '#7C5AB7', width: 1.1, dash: 'dot' } },
                    { x: tPlot, y: indices.map(i => moveR.r[i]), name: 'Right pupil |Δ|', type: 'scatter', mode: 'lines', yaxis: 'y2', line: { color: '#222222', width: 1.6 } },
                    addStimMarkerTrace()
                ], {
                    title: { text: 'Right Pseudopupil Output', font: { size: 11, color: '#333' } },
                    xaxis: noGridAxis({ title: 'ms', color: '#444', range: [0, simTime] }),
                    yaxis: noGridAxis({ title: 'Hz', color: '#444', rangemode: 'tozero' }),
                    yaxis2: noGridAxis2({ title: 'deg', overlaying: 'y', side: 'right', color: '#01579b', range: [0, 10] }),
                    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                    legend: { font: { size: 8, color: '#333' }, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ccc', borderwidth: 1 },
                    shapes: [addStimTopLine()],
                    margin: { l: 38, r: 44, t: 24, b: 28 },
                }, { responsive: true });

                const sampleStep = Math.max(1, Math.floor(indices.length / 520));
                const thR = [], rR = [], thL = [], rL = [];
                for (let i = 0; i < indices.length; i += sampleStep) {
                    const j = indices[i];
                    thR.push(moveR.th[j]); rR.push(moveR.r[j]);
                    thL.push(moveL.th[j]); rL.push(moveL.r[j]);
                }
                const mosNetR = Math.max(0.9, mosVecR.r), mosNetL = Math.max(0.9, mosVecL.r);
                const motNetR = Math.max(0.9, motVecR.r), motNetL = Math.max(0.9, motVecL.r);
                const netR = Math.max(1.0, vecR.r), netL = Math.max(1.0, vecL.r);
                const polarTraces = [
                    // Right eye trajectory + 3 arrows (MOS, MOT, net)
                    { type: 'scatterpolar', subplot: 'polar', mode: 'lines+markers', theta: thR, r: rR,
                      name: 'Right trajectory', line: { color: '#222222', width: 1.2 },
                      marker: { color: '#222222', size: 3, opacity: 0.55 } },
                    ...makePolarArrow('polar',  { theta: mosVecR.theta, r: mosNetR }, '#4D9221', '',      false),
                    ...makePolarArrow('polar',  { theta: motVecR.theta, r: motNetR }, '#5E3C99', '',      false),
                    ...makePolarArrow('polar',  { theta: vecR.theta,    r: netR    }, '#111111', 'Net R', true),
                    // Left eye trajectory + 3 arrows
                    { type: 'scatterpolar', subplot: 'polar2', mode: 'lines+markers', theta: thL, r: rL,
                      name: 'Left trajectory',  line: { color: '#222222', width: 1.2 },
                      marker: { color: '#222222', size: 3, opacity: 0.55 } },
                    ...makePolarArrow('polar2', { theta: mosVecL.theta, r: mosNetL }, '#4D9221', '',      false),
                    ...makePolarArrow('polar2', { theta: motVecL.theta, r: motNetL }, '#5E3C99', '',      false),
                    ...makePolarArrow('polar2', { theta: vecL.theta,    r: netL    }, '#111111', 'Net L', false),
                ];
                Plotly.react('circPlotPupilPolar', polarTraces, {
                    title: { text: 'Pseudopupil Direction Vectors (Toy Model)', font: { size: 11, color: '#333' } },
                    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
                    polar: {
                        domain: { x: [0.0, 0.48], y: [0, 1] },
                        bgcolor: '#f8f8f8', radialaxis: { range: [0, 11.5], color: '#555', showgrid: false },
                        angularaxis: {
                            direction: 'counterclockwise',
                            color: '#555',
                            showgrid: false,
                            tickmode: 'array',
                            tickvals: [0, 90, 180, 270],
                            ticktext: ['front', 'up', 'back', 'down']
                        }
                    },
                    polar2: {
                        domain: { x: [0.52, 1.0], y: [0, 1] },
                        bgcolor: '#f8f8f8', radialaxis: { range: [0, 11.5], color: '#555', showgrid: false },
                        angularaxis: {
                            direction: 'counterclockwise',
                            color: '#555',
                            showgrid: false,
                            tickmode: 'array',
                            tickvals: [0, 90, 180, 270],
                            ticktext: ['back', 'up', 'front', 'down']
                        }
                    },
                    annotations: [
                        { text: 'Right eye', x: 0.24, y: 1.06, xref: 'paper', yref: 'paper', showarrow: false, font: { color: '#0277bd', size: 10 } },
                        { text: 'Left eye',  x: 0.76, y: 1.06, xref: 'paper', yref: 'paper', showarrow: false, font: { color: '#2e7d32', size: 10 } }
                    ],
                    legend: { font: { size: 8, color: '#333' }, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#ccc', borderwidth: 1 },
                    margin: { l: 20, r: 20, t: 40, b: 20 }
                }, { responsive: true });
            }

            // Auto-run once panel content is constructed.
            setTimeout(() => {
                try {
                    readParams();
                    const res = buildAndRun();
                    plotResults(res);
                } catch(e) { console.error('Tier 1 circuit init error:', e); }
            }, 80);
        }

        // ── Multi-Compartment (Tier 2) model ─────────────────────────
        let mcInitialized = false;
        function renderMCModel() {
            if (mcInitialized) return;
            mcInitialized = true;
            const mcCont = document.getElementById('mcContainer');
            if (!mcCont) return;

            // ── Cell index helpers (same CI used globally) ──
            const isLPTC_ = n => n.startsWith('VS') || n.startsWith('HS');
            const isMN_   = n => n.startsWith('MO');

            // ── Build UI ──
            let h2 = '<div style="display:flex;flex-direction:column;gap:6px;">';

            // Wiring diagram placeholder
            h2 += '<div id="mcWiring" style="background:#1a1a2e;border:1px solid #444;border-radius:4px;padding:8px;overflow-x:auto;"></div>';

            // Stim controls
            h2 += '<div style="display:flex;gap:8px;align-items:flex-start;flex-wrap:wrap;padding:4px 0;">';
            h2 += '<label style="color:#ccc;font-size:11px;display:flex;flex-direction:column;gap:2px;">'
                + '<span>Stim targets <span style="color:#888;font-size:9px;">(Ctrl/\u2318 multi)</span></span>'
                + '<select id="mcStimGroup" multiple size="6" style="background:#1e1e1e;color:#fff;border:1px solid #555;font-size:10px;min-width:110px;">'
                + '<optgroup label="\u2014 Groups \u2014" style="color:#888;">'
                + '<option value="VS_L" selected>VS Left (1-4)</option>'
                + '<option value="VS_R">VS Right (1-4)</option>'
                + '<option value="HS_L">HS Left</option>'
                + '<option value="HS_R">HS Right</option>'
                + '</optgroup>'
                + '<optgroup label="\u2014 VS \u2014" style="color:#888;">'
                + '<option value="VS1_L">VS1_L</option><option value="VS1_R">VS1_R</option>'
                + '<option value="VS2_L">VS2_L</option><option value="VS2_R">VS2_R</option>'
                + '<option value="VS3_L">VS3_L</option><option value="VS3_R">VS3_R</option>'
                + '<option value="VS4_L">VS4_L</option><option value="VS4_R">VS4_R</option>'
                + '</optgroup>'
                + '<optgroup label="\u2014 HS \u2014" style="color:#888;">'
                + '<option value="HSN_L">HSN_L</option><option value="HSN_R">HSN_R</option>'
                + '<option value="HSE_L">HSE_L</option><option value="HSE_R">HSE_R</option>'
                + '<option value="HSS_L">HSS_L</option><option value="HSS_R">HSS_R</option>'
                + '</optgroup>'
                + '</select></label>';
            h2 += '<div style="display:flex;flex-direction:column;gap:4px;">';
            h2 += '<label style="color:#ccc;font-size:10px;">Amp (nA): <input id="mcAmp" type="number" value="10" step="1" min="-100" max="100" style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            h2 += '<label style="color:#ccc;font-size:10px;">t<sub>start</sub>: <input id="mcTstart" type="number" value="90" step="10" min="0" max="8000" style="width:55px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            h2 += '<label style="color:#ccc;font-size:10px;">t<sub>end</sub>: <input id="mcTend" type="number" value="590" step="10" min="0" max="8000" style="width:55px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            h2 += '<label style="color:#ccc;font-size:10px;">T<sub>max</sub>: <input id="mcTmax" type="number" value="1500" step="100" min="100" max="10000" style="width:55px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            h2 += '<label style="color:#ccc;font-size:10px;">Noise: <input id="mcNoise" type="number" value="3" step="0.5" min="0" max="10" style="width:45px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            h2 += '<button id="mcRun" style="background:#2E7D32;color:#fff;border:1px solid #4CAF50;padding:4px 14px;border-radius:3px;cursor:pointer;font-size:11px;font-weight:bold;">\u25b6 Run</button>';
            h2 += '</div></div>';

            // MC-specific params
            h2 += '<details style="color:#aaa;font-size:10px;"><summary style="cursor:pointer;color:#80cbc4;">MC Parameters (axial conductance &amp; compartment capacitance)</summary>';
            h2 += '<div style="display:flex;gap:8px;flex-wrap:wrap;padding:4px 0;">';
            h2 += '<label style="color:#80cbc4;font-size:10px;">G<sub>ax</sub>-LPTC (nS): <input id="mcGaxL" type="number" value="1.0" step="0.1" min="0.01" max="10" style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            h2 += '<label style="color:#80cbc4;font-size:10px;">G<sub>ax</sub>-MN (nS): <input id="mcGaxM" type="number" value="0.5" step="0.1" min="0.01" max="10" style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            h2 += '<label style="color:#80cbc4;font-size:10px;">C<sub>dend</sub> (pF): <input id="mcCdend" type="number" value="0.5" step="0.1" min="0.05" max="5" style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            h2 += '<label style="color:#80cbc4;font-size:10px;">C<sub>soma</sub> (pF): <input id="mcCsoma" type="number" value="0.3" step="0.05" min="0.05" max="5" style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            h2 += '<label style="color:#80cbc4;font-size:10px;">C<sub>axon</sub> (pF): <input id="mcCaxon" type="number" value="0.5" step="0.1" min="0.05" max="5" style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            h2 += '<label style="color:#80cbc4;font-size:10px;">\u03c4<sub>muscle</sub> (ms): <input id="mcTauMuscle" type="number" value="50" step="5" min="5" max="500" style="width:48px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            h2 += '</div></details>';

            // Plot areas
            h2 += '<div style="color:#888;font-size:10px;padding:2px 0;">LPTC soma Vm \u2014 all VS &amp; HS cells</div>';
            h2 += '<div id="mcPlotLPTC" style="width:100%;height:160px;background:#ffffff;border:1px solid #ccc;"></div>';
            h2 += '<div style="color:#555;font-size:10px;padding:2px 0;">MN compartments \u2014 MOS &amp; MOT (dotted=dend, dashed=soma, solid=axon)</div>';
            h2 += '<div id="mcPlotMN" style="width:100%;height:200px;background:#ffffff;border:1px solid #ccc;"></div>';
            h2 += '<div style="color:#555;font-size:10px;padding:2px 0;">Motor output \u2014 MOS axis vs MOT axis (orthogonal eye movement)</div>';
            h2 += '<div id="mcPlotMuscle" style="width:100%;height:130px;background:#ffffff;border:1px solid #ccc;"></div>';
            h2 += '<div style="color:#555;font-size:10px;padding:2px 0;">2D eye direction \u2014 MOS horizontal, MOT vertical (vector sum L\u2212R)</div>';
            h2 += '<div id="mcPlotEyeDir" style="width:100%;height:160px;background:#ffffff;border:1px solid #ccc;"></div>';
            h2 += '</div>';

            mcCont.innerHTML = h2;

            // ── Draw MC wiring diagram ──
            (function drawMCWiring() {
                const W = 880, H = 530;
                const CGJ = '#aed581', CCHEM = '#ffcc02', CT = '#ccc';
                let svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + W + ' ' + H + '" style="width:100%;max-height:530px;font-family:sans-serif;">';
                svg += '<rect width="' + W + '" height="' + H + '" fill="#1a1a2e"/>';
                svg += '<text x="' + (W/2) + '" y="16" text-anchor="middle" fill="' + CT + '" font-size="12" font-weight="bold">Multi-Compartment Wiring (Tier 2)</text>';
                svg += '<text x="' + (W/2) + '" y="28" text-anchor="middle" fill="#888" font-size="8">D=dendrite S=soma A=axon | axo-axonal GJ in LPTC chains | LPTC\u2194MN = axon\u2194dendrite GJ | LPTC\u2194LPTC syn = dendro-dendritic</text>';

                const pos = {};
                const vs_ys = [60, 120, 180, 240], hs_ys = [60, 120, 180];
                for (let k = 0; k < 4; k++) { pos['VS'+(k+1)+'_L'] = [55, vs_ys[k]]; pos['VS'+(k+1)+'_R'] = [W-55, vs_ys[k]]; }
                ['HSN','HSE','HSS'].forEach((h,i) => { pos[h+'_L'] = [165, hs_ys[i]]; pos[h+'_R'] = [W-165, hs_ys[i]]; });
                pos['MOS_L'] = [295, 90]; pos['MOT_L'] = [295, 210];
                pos['MOS_R'] = [W-295, 90]; pos['MOT_R'] = [W-295, 210];
                pos['BIPS_L'] = [165, 260]; pos['BIPS_R'] = [W-165, 260];
                pos['H2_L'] = [(165 + 295) / 2, 48]; pos['H2_R'] = [W - (165 + 295) / 2, 48];

                // 3-sub-box layout per cell
                const cW = 18, cH = 20, cGap = 2, totW = 3*cW + 2*cGap;
                function axonXc(name) { return pos[name][0] - totW/2 + 2*(cW+cGap) + cW/2; }
                function dendXc(name) { return pos[name][0] - totW/2 + cW/2; }
                function cellYc(name) { return pos[name][1]; }

                CELL_NAMES.forEach(name => {
                    const p = pos[name]; if (!p) return;
                    const col = neuronColors[name] || '#999';
                    const x0 = p[0] - totW/2, y0 = p[1] - cH/2;
                    svg += '<text x="' + p[0] + '" y="' + (y0-3) + '" text-anchor="middle" fill="' + col + '" font-size="6.5" font-weight="bold">' + name + '</text>';
                    ['D','S','A'].forEach((lbl, k) => {
                        const cx = x0 + k*(cW+cGap);
                        const op = k===2 ? 1.0 : 0.55, sw = k===2 ? 1.5 : 0.8;
                        svg += '<rect x="' + cx + '" y="' + y0 + '" width="' + cW + '" height="' + cH + '" rx="2" fill="#0d1b2e" stroke="' + col + '" stroke-width="' + sw + '" opacity="' + op + '"/>';
                        svg += '<text x="' + (cx+cW/2) + '" y="' + (y0+cH/2+3.5) + '" text-anchor="middle" fill="' + col + '" font-size="7" opacity="' + op + '">' + lbl + '</text>';
                    });
                });

                // Axo-axonal GJs within LPTC chains
                function axonGJ(a, b) {
                    const x1=axonXc(a),y1=cellYc(a),x2=axonXc(b),y2=cellYc(b);
                    svg += '<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+CGJ+'" stroke-width="1.5" stroke-dasharray="4,3"/>';
                    const mx=(x1+x2)/2, my=(y1+y2)/2;
                    svg += '<line x1="'+(mx-4)+'" y1="'+my+'" x2="'+(mx+4)+'" y2="'+my+'" stroke="'+CGJ+'" stroke-width="2.5"/>';
                }
                ['L','R'].forEach(s => {
                    for (let k=1;k<=3;k++) axonGJ('VS'+k+'_'+s,'VS'+(k+1)+'_'+s);
                    axonGJ('HSN_'+s,'HSE_'+s); axonGJ('HSE_'+s,'HSS_'+s);
                });

                // H2 ↔ contralateral HS GJ (crossing midline)
                function mc_contraGJ(a, b) {
                    const p1=pos[a], p2=pos[b];
                    if (!p1||!p2) return;
                    const mx=W/2, my=Math.min(p1[1],p2[1])-15;
                    const x1=axonXc(a),y1=cellYc(a),x2=axonXc(b),y2=cellYc(b);
                    svg+='<path d="M'+x1+','+y1+' Q'+mx+','+my+' '+x2+','+y2+'" '
                        +'fill="none" stroke="'+CGJ+'" stroke-width="1.2" stroke-dasharray="5,3" opacity="0.5"/>';
                }
                ['HSN','HSE','HSS'].forEach(h => {
                    mc_contraGJ('H2_L', h+'_R');
                    mc_contraGJ('H2_R', h+'_L');
                });

                // LPTC-axon <-> MN-dend GJ (dashed, shorter)
                function lmGJ(lptc, mn) {
                    const x1=axonXc(lptc),y1=cellYc(lptc),x2=dendXc(mn),y2=cellYc(mn);
                    svg += '<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" stroke="'+CGJ+'" stroke-width="0.8" stroke-dasharray="3,4" opacity="0.55"/>';
                }
                ['L','R'].forEach(s => {
                    for (let k=1;k<=4;k++) lmGJ('VS'+k+'_'+s,'MOS_'+s);
                    ['HSN','HSE','HSS'].forEach(h => { lmGJ(h+'_'+s,'MOS_'+s); lmGJ(h+'_'+s,'MOT_'+s); });
                });

                // ── Compartment logic: determine pre/post compartments per pair ──
                function _grp(n) {
                    if (n.startsWith('VS')) return 'VS';
                    if (n.startsWith('HS')) return 'HS';
                    if (n.startsWith('MOS')) return 'MOS';
                    if (n.startsWith('MOT')) return 'MOT';
                    if (n.startsWith('BIPS')) return 'BIPS';
                    if (n.startsWith('H2'))  return 'H2';
                    return '?';
                }
                function _isLPTC(g) { return g==='VS'||g==='HS'; }
                function _isMN(g)   { return g==='MOS'||g==='MOT'; }
                // Returns {src:'axon'|'dend', tgt:'axon'|'dend'}
                function synComp(pre, post) {
                    const gP=_grp(pre), gQ=_grp(post);
                    // LPTC ↔ LPTC (same or different type): dendro-dendritic
                    if (_isLPTC(gP) && _isLPTC(gQ)) return {src:'dend',tgt:'dend'};
                    // LPTC → MN: axo-dendritic
                    if (_isLPTC(gP) && _isMN(gQ))   return {src:'axon',tgt:'dend'};
                    // MN → LPTC: axo-dendritic
                    if (_isMN(gP) && _isLPTC(gQ))   return {src:'axon',tgt:'dend'};
                    // BIPS → LPTC: axo-dendritic
                    if (gP==='BIPS' && _isLPTC(gQ)) return {src:'axon',tgt:'dend'};
                    // LPTC → BIPS: axo-dendritic
                    if (_isLPTC(gP) && gQ==='BIPS')  return {src:'axon',tgt:'dend'};
                    // Default: axo-dendritic
                    return {src:'axon',tgt:'dend'};
                }

                // Chemical synapses (compartment-aware)
                for (let pi=0;pi<N_CELLS;pi++) {
                    for (let qi=0;qi<N_CELLS;qi++) {
                        const cnt=RAW_COUNTS[pi][qi]; if (!cnt) continue;
                        const pre=CELL_NAMES[pi], post=CELL_NAMES[qi];
                        if (!pos[pre]||!pos[post]) continue;
                        const isInh = SYN_ESYN[pi][qi] < -10;
                        const col = isInh ? '#64b5f6' : CCHEM;
                        const comp = synComp(pre, post);
                        const x1 = comp.src==='axon' ? axonXc(pre) : dendXc(pre);
                        const y1 = cellYc(pre);
                        const x2 = comp.tgt==='dend' ? dendXc(post) : axonXc(post);
                        const y2 = cellYc(post);
                        const dx=x2-x1,dy=y2-y1,len=Math.sqrt(dx*dx+dy*dy); if(len<4) continue;
                        const ux=dx/len,uy=dy/len;
                        const aid='mc_ca_'+pi+'_'+qi;
                        if (isInh) {
                            svg+='<defs><marker id="'+aid+'" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto"><line x1="0" y1="0" x2="0" y2="8" stroke="'+col+'" stroke-width="2"/></marker></defs>';
                        } else {
                            svg+='<defs><marker id="'+aid+'" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="'+col+'"/></marker></defs>';
                        }
                        svg+='<line x1="'+(x1+ux*2)+'" y1="'+(y1+uy*2)+'" x2="'+(x2-ux*2)+'" y2="'+(y2-uy*2)+'" stroke="'+col+'" stroke-width="'+Math.max(0.5,Math.sqrt(cnt)*0.35)+'" marker-end="url(#'+aid+')" opacity="0.6"/>';
                    }
                }

                // ── Per-eye muscles + retina + visuomotor feedback loop ──
                const mLx = W/2 - 65, mRx = W/2 + 65;
                const mHy = pos['MOS_L'][1], mVy = pos['MOT_L'][1];

                function muscleBox(cx, cy, label, sublabel, subCol) {
                    svg+='<rect x="'+(cx-30)+'" y="'+(cy-14)+'" width="60" height="28" rx="4" fill="#1a1a1a" stroke="#ff8a65" stroke-width="1.5"/>';
                    svg+='<text x="'+cx+'" y="'+(cy-2)+'" text-anchor="middle" fill="#ff8a65" font-size="7" font-weight="bold">'+label+'</text>';
                    svg+='<text x="'+cx+'" y="'+(cy+9)+'" text-anchor="middle" fill="'+subCol+'" font-size="5.5">'+sublabel+'</text>';
                }
                muscleBox(mLx, mHy, 'L Musc \u2194', 'horiz (MOS)', '#ef5350');
                muscleBox(mLx, mVy, 'L Musc \u2195', 'vert (MOT)', '#ff7043');
                muscleBox(mRx, mHy, 'R Musc \u2194', 'horiz (MOS)', '#ef5350');
                muscleBox(mRx, mVy, 'R Musc \u2195', 'vert (MOT)', '#ff7043');

                // Arrows from MOS/MOT axon → per-eye muscles
                ['L','R'].forEach(s => {
                    ['MOS','MOT'].forEach(mn => {
                        const nName=mn+'_'+s, ax=axonXc(nName), ay=cellYc(nName);
                        const targX = s==='L' ? mLx : mRx;
                        const targY = mn==='MOS' ? mHy : mVy;
                        const x2 = s==='L' ? (targX-30) : (targX+30);
                        const aid='marr_'+nName;
                        svg+='<defs><marker id="'+aid+'" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#ff8a65"/></marker></defs>';
                        svg+='<line x1="'+ax+'" y1="'+ay+'" x2="'+x2+'" y2="'+targY+'" stroke="#ff8a65" stroke-width="1.5" marker-end="url(#'+aid+')"/>';
                    });
                });

                // ── Retina boxes (one per eye) ──
                const retY = 290;
                svg+='<rect x="'+(mLx-38)+'" y="'+(retY-12)+'" width="76" height="24" rx="4" fill="#1a1a1a" stroke="#26c6da" stroke-width="1.5" stroke-dasharray="4,2"/>';
                svg+='<text x="'+mLx+'" y="'+(retY+4)+'" text-anchor="middle" fill="#26c6da" font-size="7.5" font-weight="bold">L Eye / Retina</text>';
                svg+='<rect x="'+(mRx-38)+'" y="'+(retY-12)+'" width="76" height="24" rx="4" fill="#1a1a1a" stroke="#26c6da" stroke-width="1.5" stroke-dasharray="4,2"/>';
                svg+='<text x="'+mRx+'" y="'+(retY+4)+'" text-anchor="middle" fill="#26c6da" font-size="7.5" font-weight="bold">R Eye / Retina</text>';

                // Muscle → retina arrows (both muscles move the same eye)
                const retMkr = 'mc_mret';
                svg+='<defs><marker id="'+retMkr+'" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#ff8a65"/></marker></defs>';
                [mLx, mRx].forEach(mx => {
                    svg+='<line x1="'+mx+'" y1="'+(mHy+14)+'" x2="'+mx+'" y2="'+(retY-12)+'" stroke="#ff8a65" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#'+retMkr+')" opacity="0.7"/>';
                    svg+='<line x1="'+mx+'" y1="'+(mVy+14)+'" x2="'+mx+'" y2="'+(retY-12)+'" stroke="#ff8a65" stroke-width="1" stroke-dasharray="3,2" marker-end="url(#'+retMkr+')" opacity="0.7"/>';
                });

                // ── Sensory feedback: retina → HS (horiz motion) & VS (vert motion) ──
                const fbCol='#26c6da', fbMkr='mc_sfb';
                svg+='<defs><marker id="'+fbMkr+'" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="'+fbCol+'"/></marker></defs>';
                const hsBot2=hs_ys[2]+16, vsBot2=vs_ys[3]+16;
                // L retina → HS_L
                svg+='<path d="M'+(mLx-38)+' '+retY+' Q 165 '+(retY+15)+' 165 '+hsBot2+'" fill="none" stroke="'+fbCol+'" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#'+fbMkr+')" opacity="0.7"/>';
                svg+='<text x="'+((mLx-38+165)/2)+'" y="'+(retY+12)+'" text-anchor="middle" fill="'+fbCol+'" font-size="5.5" opacity="0.7">horiz\u2192HS</text>';
                // L retina → VS_L
                svg+='<path d="M'+(mLx-38)+' '+retY+' Q 55 '+(retY+25)+' 55 '+vsBot2+'" fill="none" stroke="'+fbCol+'" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#'+fbMkr+')" opacity="0.7"/>';
                svg+='<text x="'+((mLx-38+55)/2)+'" y="'+(retY+22)+'" text-anchor="middle" fill="'+fbCol+'" font-size="5.5" opacity="0.7">vert\u2192VS</text>';
                // R retina → HS_R
                svg+='<path d="M'+(mRx+38)+' '+retY+' Q '+(W-165)+' '+(retY+15)+' '+(W-165)+' '+hsBot2+'" fill="none" stroke="'+fbCol+'" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#'+fbMkr+')" opacity="0.7"/>';
                svg+='<text x="'+((mRx+38+W-165)/2)+'" y="'+(retY+12)+'" text-anchor="middle" fill="'+fbCol+'" font-size="5.5" opacity="0.7">horiz\u2192HS</text>';
                // R retina → VS_R
                svg+='<path d="M'+(mRx+38)+' '+retY+' Q '+(W-55)+' '+(retY+25)+' '+(W-55)+' '+vsBot2+'" fill="none" stroke="'+fbCol+'" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#'+fbMkr+')" opacity="0.7"/>';
                svg+='<text x="'+((mRx+38+W-55)/2)+'" y="'+(retY+22)+'" text-anchor="middle" fill="'+fbCol+'" font-size="5.5" opacity="0.7">vert\u2192VS</text>';

                // Feedback loop banner
                svg+='<text x="'+W/2+'" y="'+(retY+38)+'" text-anchor="middle" fill="#26c6da" font-size="8.5" font-weight="bold" opacity="0.6">\u27f2 VISUOMOTOR FEEDBACK: muscles \u2192 retinal shift \u2192 visual scene \u2192 LPTC</text>';

                svg += '<text x="10" y="'+(H-18)+'" fill="#555" font-size="7">Green dashed = GJ | Yellow = excitatory syn (ACh/Glut) | Blue = inhibitory (GABA) | Orange = motor output</text>';
                svg += '<text x="10" y="'+(H-6)+'" fill="#555" font-size="7">Cyan dashed = sensory feedback: muscle \u2192 retinal motion \u2192 LPTC input | \u27f2 = closed visuomotor loop</text>';
                svg += '</svg>';
                const wd = document.getElementById('mcWiring');
                if (wd) wd.innerHTML = svg;
            })();

            // ── Parameter reader ──
            const mcpf = (id, def) => {
                const el = document.getElementById(id);
                const v = el ? parseFloat(el.value) : NaN;
                return isNaN(v) ? def : v;
            };

            function readMCParams() {
                const sel = document.getElementById('mcStimGroup');
                const grpMap = {
                    'VS_L':['VS1_L','VS2_L','VS3_L','VS4_L'], 'VS_R':['VS1_R','VS2_R','VS3_R','VS4_R'],
                    'HS_L':['HSN_L','HSE_L','HSS_L'], 'HS_R':['HSN_R','HSE_R','HSS_R']
                };
                return {
                    amp:    mcpf('mcAmp', 10),
                    tstart: mcpf('mcTstart', 90),
                    tend:   mcpf('mcTend', 590),
                    tmax:   mcpf('mcTmax', 1500),
                    noise:  mcpf('mcNoise', 3),
                    gaxL:   mcpf('mcGaxL', 1.0),
                    gaxM:   mcpf('mcGaxM', 0.5),
                    cdend:  mcpf('mcCdend', 0.5),
                    csoma:  mcpf('mcCsoma', 0.3),
                    caxon:  mcpf('mcCaxon', 0.5),
                    tauMuscle: mcpf('mcTauMuscle', 50),
                    targets: Array.from(sel.selectedOptions).flatMap(o => grpMap[o.value] || [o.value]),
                };
            }

            // ── Main MC simulation ──
            function buildAndRunMC(p) {
                const dt = 0.05;
                const NT = Math.round(p.tmax / dt);
                const N  = N_CELLS;
                const isLPTCa = CELL_NAMES.map(n => isLPTC_(n));
                const isMNa   = CELL_NAMES.map(n => isMN_(n));

                // Rev potentials
                const ENa=55, EK=-80, ECa=120;

                // Cell parameters (from Tier 1 defaults)
                const pGNa=120, pGK_m=36, pGL_m=0.3, pVLm=-65, pGNaP=0.3;
                const pGVT_l=0.5, pGK_l=2.0, pGL_l=0.05;
                const pRinVS1=150, pVrVS1=-40, pRinHS=150, pVrHS=-45;
                const pGlptc=0.05, pClptc=0.05;
                const pGvsmos=0.1, pGhsmos=0.1, pGhsmot=0.1, pCmn=0.8;
                const pGgrad=0.005, pGspike=0.02, pTauSyn=5;
                const pVthresh=-40, pVscale=20;
                const pRinM=300;

                const Vr = CELL_NAMES.map(n => n.startsWith('HS') ? pVrHS : n.startsWith('VS') ? pVrVS1 : pVLm);
                const Rin= CELL_NAMES.map(n => n.startsWith('HS') ? pRinHS : n.startsWith('VS') ? pRinVS1 : pRinM);

                // Capacitances
                const Cd = p.cdend, Cs = p.csoma, Ca = p.caxon;

                // State: V[c][0=dend, 1=soma, 2=axon]
                const V = CELL_NAMES.map((n,c) => [Vr[c], Vr[c], Vr[c]]);

                // HH gates for MN axon
                function aM(v){ return Math.abs(v+40)<1e-4 ? 1.0 : 0.1*(v+40)/(1-Math.exp(-(v+40)/10)); }
                function bM(v){ return 4*Math.exp(-(v+65)/18); }
                function aH(v){ return 0.07*Math.exp(-(v+65)/20); }
                function bH(v){ return 1/(1+Math.exp(-(v+35)/10)); }
                function aN(v){ return Math.abs(v+55)<1e-4 ? 0.1 : 0.01*(v+55)/(1-Math.exp(-(v+55)/10)); }
                function bN(v){ return 0.125*Math.exp(-(v+65)/80); }

                const mHH=[], hHH=[], nHH=[];
                CELL_NAMES.forEach((n,c) => {
                    const v=V[c][2];
                    mHH.push(aM(v)/(aM(v)+bM(v))); hHH.push(aH(v)/(aH(v)+bH(v))); nHH.push(aN(v)/(aN(v)+bN(v)));
                });

                // T-Ca / K gating for LPTC axon
                const tMinf=v=>1/(1+Math.exp(-(v+50)/3));
                const tHinf=v=>1/(1+Math.exp((v+68)/3.75));
                const tHtau=v=>1/(Math.exp((v+160)/30)+Math.exp(-(v+84)/7.3))+22.7;
                const kMinf=v=>1/(1+Math.exp(-(v+20)/8));
                const mCa=[], hCa=[], nKl=[];
                CELL_NAMES.forEach((n,c) => {
                    const v=V[c][2];
                    mCa.push(tMinf(v)); hCa.push(tHinf(v)); nKl.push(kMinf(v));
                });

                // GJ structures
                // Axo-axonal within LPTC chains: {a, b, g, tau, Vfa, Vfb}
                const gjAx = [];
                function mkAxGJ(na, nb, g, tau) {
                    return { a: CI[na], b: CI[nb], g, tau, Vfa: V[CI[na]][2], Vfb: V[CI[nb]][2] };
                }
                ['L','R'].forEach(s => {
                    for (let k=1;k<=3;k++) gjAx.push(mkAxGJ('VS'+k+'_'+s,'VS'+(k+1)+'_'+s, pGlptc, pClptc));
                    gjAx.push(mkAxGJ('HSN_'+s,'HSE_'+s, pGlptc, pClptc));
                    gjAx.push(mkAxGJ('HSE_'+s,'HSS_'+s, pGlptc, pClptc));
                });

                // LPTC-axon <-> MN-dend GJ: {cl, cm, g, tau, VfL, VfM}
                const gjLD = [];
                function mkLDgj(nl, nm, g, tau) {
                    return { cl: CI[nl], cm: CI[nm], g, tau, VfL: V[CI[nl]][2], VfM: V[CI[nm]][0] };
                }
                ['L','R'].forEach(s => {
                    for (let k=1;k<=4;k++) gjLD.push(mkLDgj('VS'+k+'_'+s,'MOS_'+s, pGvsmos, pCmn));
                    ['HSN','HSE','HSS'].forEach(h => {
                        gjLD.push(mkLDgj(h+'_'+s,'MOS_'+s, pGhsmos, pCmn));
                        gjLD.push(mkLDgj(h+'_'+s,'MOT_'+s, pGhsmot, pCmn));
                    });
                });

                // Graded syn lookup: LPTC axon -> MN dend
                const gradSyns = [];
                for (let pi=0;pi<N;pi++) {
                    if (!isLPTCa[pi]) continue;
                    for (let qi=0;qi<N;qi++) {
                        if (!isMNa[qi]) continue;
                        const cnt=RAW_COUNTS[pi][qi]; if (!cnt) continue;
                        gradSyns.push({ pre:pi, post:qi, cnt, Erev:SYN_ESYN[pi][qi] });
                    }
                }

                // Alpha synapses: MN axon -> LPTC dend
                const alphaSyns = [];
                for (let pi=0;pi<N;pi++) {
                    if (!isMNa[pi]) continue;
                    for (let qi=0;qi<N;qi++) {
                        if (!isLPTCa[qi]) continue;
                        const cnt=RAW_COUNTS[pi][qi]; if (!cnt) continue;
                        alphaSyns.push({ pre:pi, post:qi, g:0, dg:0, tau:pTauSyn, Erev:SYN_ESYN[pi][qi] });
                    }
                }

                // Spike detection prev state
                const prevVax = new Float64Array(N);

                // Muscle tension (leaky integrator)
                const muscTens = new Float64Array(N);

                // Stim set
                const stimSet = new Set(p.targets.map(n=>CI[n]).filter(i=>i!=null&&i>=0));

                // Recording (downsample 4x)
                const recStep = 4, NR = Math.ceil(NT/recStep);
                const tRec = new Float32Array(NR);
                const recAxon  = CELL_NAMES.map(()=>new Float32Array(NR));
                const recDend  = CELL_NAMES.map(()=>new Float32Array(NR));
                const recSoma  = CELL_NAMES.map(()=>new Float32Array(NR));
                const mnIdxs   = CELL_NAMES.map((n,i)=>isMNa[i]?i:-1).filter(i=>i>=0);
                const recMuscle= mnIdxs.map(()=>new Float32Array(NR));

                let ri = 0;

                for (let t=0; t<NT; t++) {
                    const tms = t * dt;
                    const stim = (tms >= p.tstart && tms < p.tend);

                    // dV contributions (will be divided by C inside each section)
                    const dVd = new Float64Array(N);  // dend
                    const dVs = new Float64Array(N);  // soma
                    const dVa = new Float64Array(N);  // axon

                    // 1) Intrinsic currents
                    for (let c=0; c<N; c++) {
                        const vd=V[c][0], vs_=V[c][1], va=V[c][2];
                        if (isLPTCa[c]) {
                            const gl = 1/Rin[c], vr = Vr[c];
                            dVd[c] += gl*(vr-vd)/Cd;
                            dVs[c] += gl*(vr-vs_)/Cs;
                            const ITca = pGVT_l*mCa[c]*mCa[c]*hCa[c]*(ECa-va);
                            const IKl  = pGK_l *nKl[c]*nKl[c]*nKl[c]*nKl[c]*(EK-va);
                            const ILl  = pGL_l *(vr-va);
                            dVa[c] += (ITca+IKl+ILl)/Ca;
                            // Update T-Ca/K gates
                            mCa[c] += (tMinf(va)-mCa[c])/2 * dt;
                            hCa[c] += (tHinf(va)-hCa[c])/tHtau(va) * dt;
                            nKl[c] += (kMinf(va)-nKl[c])/20 * dt;
                        } else if (isMNa[c]) {
                            const gl = 1/Rin[c];
                            dVd[c] += gl*(pVLm-vd)/Cd;
                            dVs[c] += gl*(pVLm-vs_)/Cs;
                            const INa  = pGNa *mHH[c]*mHH[c]*mHH[c]*hHH[c]*(ENa-va);
                            const IKhh = pGK_m*nHH[c]*nHH[c]*nHH[c]*nHH[c]*(EK-va);
                            const INaP = pGNaP*(1/(1+Math.exp(-(va+52)/5)))*(ENa-va);
                            const ILm  = pGL_m*(pVLm-va);
                            dVa[c] += (INa+IKhh+INaP+ILm)/Ca;
                            // Update HH gates
                            mHH[c] += (aM(va)*(1-mHH[c])-bM(va)*mHH[c])*dt;
                            hHH[c] += (aH(va)*(1-hHH[c])-bH(va)*hHH[c])*dt;
                            nHH[c] += (aN(va)*(1-nHH[c])-bN(va)*nHH[c])*dt;
                        }
                    }

                    // 2) Axial coupling dend<->soma<->axon
                    for (let c=0; c<N; c++) {
                        const gax = isLPTCa[c] ? p.gaxL : p.gaxM;
                        const ids = gax*(V[c][1]-V[c][0]);
                        dVd[c] += ids/Cd; dVs[c] -= ids/Cs;
                        const isa = gax*(V[c][2]-V[c][1]);
                        dVs[c] += isa/Cs; dVa[c] -= isa/Ca;
                    }

                    // 3) Axo-axonal GJs (LPTC chains)
                    for (const gj of gjAx) {
                        const tauk = gj.tau/dt+1;
                        gj.Vfa += (V[gj.a][2]-gj.Vfa)/tauk;
                        gj.Vfb += (V[gj.b][2]-gj.Vfb)/tauk;
                        dVa[gj.a] += gj.g*(gj.Vfb-V[gj.a][2])/Ca;
                        dVa[gj.b] += gj.g*(gj.Vfa-V[gj.b][2])/Ca;
                    }

                    // 4) LPTC-axon <-> MN-dend GJs
                    for (const gj of gjLD) {
                        const tauk = gj.tau/dt+1;
                        gj.VfL += (V[gj.cl][2]-gj.VfL)/tauk;
                        gj.VfM += (V[gj.cm][0]-gj.VfM)/tauk;
                        dVa[gj.cl] += gj.g*(gj.VfM-V[gj.cl][2])/Ca;
                        dVd[gj.cm] += gj.g*(gj.VfL-V[gj.cm][0])/Cd;
                    }

                    // 5) Graded chem syn: LPTC axon -> MN dend
                    for (const s of gradSyns) {
                        const Spre = 1/(1+Math.exp(-(V[s.pre][2]-pVthresh)/pVscale));
                        dVd[s.post] += -pGgrad*s.cnt*Spre*(V[s.post][0]-s.Erev)/Cd;
                    }

                    // 6) Alpha syn: MN axon -> LPTC dend (update conductance)
                    for (const s of alphaSyns) {
                        s.g  += s.dg*dt;
                        s.dg -= s.dg/s.tau*dt;
                        s.g   = Math.max(0, s.g);
                        dVd[s.post] += -s.g*pGspike*(V[s.post][0]-s.Erev)/Cd;
                    }

                    // 7) Spike detection on MN axon -> trigger alpha syn
                    for (let c=0; c<N; c++) {
                        if (!isMNa[c]) continue;
                        if (prevVax[c] < 0 && V[c][2] >= 0) {
                            for (const s of alphaSyns) { if (s.pre===c) s.dg += 1.0/s.tau; }
                            muscTens[c] += 1.0;
                        }
                        prevVax[c] = V[c][2];
                    }

                    // 8) Muscle tension (leaky integrator)
                    for (let c=0; c<N; c++) {
                        if (isMNa[c]) muscTens[c] -= muscTens[c]*dt/p.tauMuscle;
                    }

                    // 9) Stim (into dendrite of stimulated cells)
                    if (stim) { for (const ci of stimSet) dVd[ci] += p.amp/Cd; }

                    // 10) Noise (into axon of each cell)
                    for (let c=0; c<N; c++) { dVa[c] += p.noise*(Math.random()-0.5)*2/Ca; }

                    // Euler step
                    for (let c=0; c<N; c++) {
                        V[c][0] += dVd[c]*dt;
                        V[c][1] += dVs[c]*dt;
                        V[c][2] += dVa[c]*dt;
                    }

                    // Record
                    if (t%recStep===0 && ri<NR) {
                        tRec[ri] = tms;
                        for (let c=0; c<N; c++) { recDend[c][ri]=V[c][0]; recSoma[c][ri]=V[c][1]; recAxon[c][ri]=V[c][2]; }
                        for (let mi=0; mi<mnIdxs.length; mi++) recMuscle[mi][ri] = muscTens[mnIdxs[mi]];
                        ri++;
                    }
                }

                return {
                    t: Array.from(tRec.subarray(0,ri)),
                    dend:   recDend.map(a=>Array.from(a.subarray(0,ri))),
                    soma:   recSoma.map(a=>Array.from(a.subarray(0,ri))),
                    axon:   recAxon.map(a=>Array.from(a.subarray(0,ri))),
                    muscle: recMuscle.map(a=>Array.from(a.subarray(0,ri))),
                    mnIdxs,
                };
            }

            // ── Plotting ──
            function plotMCResults(res, p) {
                const t = res.t;

                // LPTC soma traces
                const lptcTraces = CELL_NAMES
                    .map((n,i) => ({ n, i }))
                    .filter(({n}) => isLPTC_(n))
                    .map(({n,i}) => ({
                        x: t, y: res.soma[i], name: n, type: 'scatter', mode: 'lines',
                        line: { color: neuronColors[n]||'#aaa', width: 1.0 },
                    }));
                lptcTraces.push({ x:[p.tstart,p.tend,p.tend,p.tstart], y:[-80,-80,20,20],
                    fill:'toself', fillcolor:'rgba(255,140,0,0.18)', line:{width:0},
                    showlegend:false, hoverinfo:'skip', type:'scatter', mode:'lines' });
                Plotly.react('mcPlotLPTC', lptcTraces, {
                    title: { text:'LPTC Soma Vm (VS + HS)', font:{size:10,color:'#333'} },
                    xaxis: { title:'ms', color:'#444', gridcolor:'#ddd' },
                    yaxis: { title:'mV', color:'#444', gridcolor:'#ddd', range:[-80,20] },
                    paper_bgcolor:'#ffffff', plot_bgcolor:'#ffffff',
                    legend:{ font:{size:8,color:'#333'}, bgcolor:'rgba(255,255,255,0.85)', bordercolor:'#ccc', borderwidth:1 },
                    margin:{l:40,r:10,t:24,b:28},
                }, {responsive:true});

                // MN compartments (per cell: dend=dotted, soma=dashed, axon=solid)
                const mnTraces = [];
                res.mnIdxs.forEach(ci => {
                    const n = CELL_NAMES[ci], col = neuronColors[n]||'#ff7043';
                    mnTraces.push({ x:t, y:res.dend[ci], name:n+' dend', type:'scatter', mode:'lines', line:{color:col, width:0.8, dash:'dot'} });
                    mnTraces.push({ x:t, y:res.soma[ci], name:n+' soma', type:'scatter', mode:'lines', line:{color:col, width:0.8, dash:'dash'} });
                    mnTraces.push({ x:t, y:res.axon[ci], name:n+' axon',  type:'scatter', mode:'lines', line:{color:col, width:1.5} });
                });
                mnTraces.push({ x:[p.tstart,p.tend,p.tend,p.tstart], y:[-80,-80,70,70],
                    fill:'toself', fillcolor:'rgba(255,140,0,0.18)', line:{width:0},
                    showlegend:false, hoverinfo:'skip', type:'scatter', mode:'lines' });
                Plotly.react('mcPlotMN', mnTraces, {
                    title: { text:'MN Compartments  (dotted=dend, dashed=soma, solid=axon)', font:{size:10,color:'#333'} },
                    xaxis: { title:'ms', color:'#444', gridcolor:'#ddd' },
                    yaxis: { title:'mV', color:'#444', gridcolor:'#ddd', range:[-80,70] },
                    paper_bgcolor:'#ffffff', plot_bgcolor:'#ffffff',
                    legend:{ font:{size:8,color:'#333'}, bgcolor:'rgba(255,255,255,0.85)', bordercolor:'#ccc', borderwidth:1 },
                    margin:{l:40,r:10,t:24,b:28},
                }, {responsive:true});

                // Muscle tension
                const muscTraces = res.mnIdxs.map((ci,mi) => {
                    const n = CELL_NAMES[ci];
                    const isMOS = n.startsWith('MOS');
                    return { x:t, y:res.muscle[mi], name:n + (isMOS ? ' (horiz)' : ' (vert)'), type:'scatter', mode:'lines',
                             line:{ color: neuronColors[n]||'#ff8a65', width:1.5, dash: isMOS ? 'solid' : 'dash' } };
                });
                Plotly.react('mcPlotMuscle', muscTraces, {
                    title: { text:'Muscle Tension \u2014 MOS=horizontal axis (solid), MOT=vertical axis (dashed)', font:{size:10,color:'#333'} },
                    xaxis: { title:'ms', color:'#444', gridcolor:'#ddd' },
                    yaxis: { title:'AU', color:'#444', gridcolor:'#ddd', rangemode:'tozero' },
                    paper_bgcolor:'#ffffff', plot_bgcolor:'#ffffff',
                    legend:{ font:{size:8,color:'#333'}, bgcolor:'rgba(255,255,255,0.85)', bordercolor:'#ccc', borderwidth:1 },
                    margin:{l:40,r:10,t:24,b:28},
                }, {responsive:true});

                // 2D eye direction: MOS = horizontal (x), MOT = vertical (y)
                // Compute L-R difference for each MN type as net drive per axis
                const mosLi = res.mnIdxs.findIndex(ci => CELL_NAMES[ci]==='MOS_L');
                const mosRi = res.mnIdxs.findIndex(ci => CELL_NAMES[ci]==='MOS_R');
                const motLi = res.mnIdxs.findIndex(ci => CELL_NAMES[ci]==='MOT_L');
                const motRi = res.mnIdxs.findIndex(ci => CELL_NAMES[ci]==='MOT_R');
                const nPts = t.length;
                const eyeX = new Array(nPts), eyeY = new Array(nPts);
                for (let i=0; i<nPts; i++) {
                    eyeX[i] = (mosLi>=0 ? res.muscle[mosLi][i] : 0) - (mosRi>=0 ? res.muscle[mosRi][i] : 0);
                    eyeY[i] = (motLi>=0 ? res.muscle[motLi][i] : 0) - (motRi>=0 ? res.muscle[motRi][i] : 0);
                }
                const eyeTraces = [
                    { x:t, y:eyeX, name:'MOS axis (L\u2212R)', type:'scatter', mode:'lines',
                      line:{ color:'#4CAF50', width:1.5 } },
                    { x:t, y:eyeY, name:'MOT axis (L\u2212R)', type:'scatter', mode:'lines',
                      line:{ color:'#7C5AB7', width:1.5, dash:'dash' } },
                ];
                // Also add a small polar-like trace: instantaneous direction
                // Sample every 20 points for direction arrows
                const step = Math.max(1, Math.floor(nPts/40));
                const arrowX = [], arrowY = [], arrowText = [];
                for (let i=0; i<nPts; i+=step) {
                    const mag = Math.sqrt(eyeX[i]*eyeX[i]+eyeY[i]*eyeY[i]);
                    if (mag > 0.01) {
                        const ang = Math.atan2(eyeY[i], eyeX[i]) * 180/Math.PI;
                        arrowText.push('t='+t[i].toFixed(0)+'ms  \u03b8='+ang.toFixed(0)+'\u00b0  |F|='+mag.toFixed(2));
                    }
                }
                Plotly.react('mcPlotEyeDir', eyeTraces, {
                    title: { text:'Eye Movement Vector (MOS=horiz, MOT=vert) \u2014 L minus R', font:{size:10,color:'#333'} },
                    xaxis: { title:'ms', color:'#444', gridcolor:'#ddd' },
                    yaxis: { title:'L\u2212R (AU)', color:'#444', gridcolor:'#ddd', zeroline:true, zerolinecolor:'#bbb' },
                    paper_bgcolor:'#ffffff', plot_bgcolor:'#ffffff',
                    legend:{ font:{size:8,color:'#333'}, bgcolor:'rgba(255,255,255,0.85)', bordercolor:'#ccc', borderwidth:1 },
                    margin:{l:40,r:10,t:24,b:28},
                }, {responsive:true});
            }

            // ── Run button ──
            document.getElementById('mcRun').addEventListener('click', function() {
                this.disabled = true; this.textContent = '\u23f3 Running...';
                setTimeout(() => {
                    try {
                        const params = readMCParams();
                        const result = buildAndRunMC(params);
                        plotMCResults(result, params);
                    } catch(e) { console.error('MC model error:', e); alert('MC model error: ' + e.message); }
                    this.disabled = false; this.textContent = '\u25b6 Run';
                }, 20);
            });

            // Auto-run on first open
            setTimeout(() => {
                try {
                    const params = readMCParams();
                    const result = buildAndRunMC(params);
                    plotMCResults(result, params);
                } catch(e) { console.error('MC init error:', e); }
            }, 80);
        }

        // ── Resizable panels ────────────────────────────────────────
        let isResizing = false, curResizer = null;
        function initResize(resizer, leftPanel, rightPanel) {
            resizer.addEventListener('mousedown', function(e) {
                isResizing = true;
                curResizer = { resizer, leftPanel, rightPanel };
                document.body.style.cursor = 'col-resize';
                e.preventDefault();
            });
        }
        document.addEventListener('mousemove', function(e) {
            if (!isResizing || !curResizer) return;
            const rect = document.querySelector('.container').getBoundingClientRect();
            if (curResizer.leftPanel === sidebar) {
                const w = e.clientX - rect.left;
                if (w >= 100 && w <= 300) sidebar.style.width = w + 'px';
            } else if (curResizer.rightPanel === emPanel) {
                const w = rect.right - e.clientX;
                if (w >= 400 && w <= 1000) emPanel.style.width = w + 'px';
            }
            try { if (plotDiv) Plotly.Plots.resize(plotDiv); } catch(e) {}
        });
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false; curResizer = null;
                document.body.style.cursor = 'default';
            }
        });
        initResize(document.getElementById('resizer1'), sidebar, centerCol);
        initResize(document.getElementById('resizer2'), centerCol, emPanel);

        itemInfo.textContent = '';
        updateZValue(0);
    </script>
</body>
</html>"""


# ── Data loading functions ────────────────────────────────────────────

def load_contacts(results_dir):
    """Load contact patches using EXACTLY the same logic as generate_em_stacks.py.

    Iterates all_results_combined.csv Has_Contact rows, expands Top1..TopN
    patches with sequential indices matching EM snapshot filenames:
        contact_0_segmented.png, contact_1_segmented.png, ...
    """
    csv_file = os.path.join(results_dir, 'all_results_combined.csv')
    df = pd.read_csv(csv_file)
    df = df[df['Has_Contact'] == True]

    n_top = _cfg.get('top_patches', 10)

    all_patches = []
    patch_idx = 0

    for _, row in df.iterrows():
        source = row['Source_Neuron']
        target = row['Target_Neuron']

        for pn in range(1, n_top + 1):
            x_col = f'Top{pn}_Patch_Centroid_X'
            y_col = f'Top{pn}_Patch_Centroid_Y'
            z_col = f'Top{pn}_Patch_Centroid_Z'
            a_col = f'Top{pn}_Patch_Area_um2'

            if all(c in df.columns for c in [x_col, y_col, z_col]):
                if not pd.isna(row[x_col]):
                    area = 0.0
                    if a_col in df.columns and not pd.isna(row.get(a_col, np.nan)):
                        area = float(row[a_col])
                    all_patches.append({
                        'idx': patch_idx,
                        'x': float(row[x_col]),
                        'y': float(row[y_col]),
                        'z': float(row[z_col]),
                        'source': source,
                        'target': target,
                        'patch_num': pn,
                        'patch_area': area,
                    })
                    patch_idx += 1

    result_df = pd.DataFrame(all_patches)
    # Ensure expected columns exist even if empty
    if result_df.empty:
        result_df = pd.DataFrame(columns=['idx','x','y','z','source','target','patch_num','patch_area'])
    print(f"[contacts] Loaded {len(result_df)} patches from "
          f"{len(df)} contact pairs (Top1-{n_top})")
    return result_df


def load_synapses(results_dir):
    """Load synapses for configured synapse_groups."""
    csv_file = os.path.join(results_dir, 'synapses.csv')
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['x', 'y', 'z'])

    if 'pre_type' in df.columns:
        df['source'] = df['pre_type']
        df['target'] = df['post_type']
    else:
        df['source'] = df['pre']
        df['target'] = df['post']

    df = df[df['source'].isin(SYNAPSE_NEURONS) | df['target'].isin(SYNAPSE_NEURONS)]
    print(f"[synapses] Loaded {len(df)} synapses "
          f"(groups: {', '.join(sorted(_synapse_groups))})")
    return df[['x', 'y', 'z', 'source', 'target']].reset_index()


def load_overlap_vertices(results_dir):
    """Load overlap midpoints, organized per neuron AND per pair.

    Also merges overlap_em_meta.json (written by generate_em_stacks.py)
    so each pair carries its EM snapshot idx, z_lo, z_hi.

    Returns:
        per_neuron: {neuron: {x, y, z}} — all overlap vertices involving neuron
        per_pair:   {neuron: [{other, idx, z_lo, z_hi, x[], y[], z[]}, ...]}
    """
    csv_file = os.path.join(results_dir, 'geometric_data', 'contact_vertices.csv')
    if not os.path.exists(csv_file):
        print("[overlaps] No contact_vertices.csv found, skipping overlap viz")
        return {}, {}

    print("[overlaps] Loading contact_vertices.csv ...")
    df = pd.read_csv(csv_file)

    # Load EM metadata if available (for idx / z_lo / z_hi)
    # Now supports multiple sub-clusters per (source, target) pair
    meta_file = os.path.join(results_dir, 'overlap_em_meta.json')
    em_meta_list = {}  # (source, target) -> [item, ...]  (list of sub-clusters)
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            for item in json.load(f):
                a, b = item['source'], item['target']
                if 'slice_detail' in item:
                    item['valid_z'] = sorted([s['z_offset'] for s in item['slice_detail']])
                else:
                    item['valid_z'] = list(range(item.get('z_lo', -20), item.get('z_hi', 20) + 1))
                em_meta_list.setdefault((a, b), []).append(item)
                em_meta_list.setdefault((b, a), []).append(item)
        n_entries = sum(len(v) for v in em_meta_list.values()) // 2
        n_pairs = len(set(tuple(sorted(k)) for k in em_meta_list.keys()))
        print(f"[overlaps] Loaded EM metadata: {n_entries} sub-clusters across {n_pairs} pairs")

        # Ensure every meta entry has z_base_nm (absolute Z of first EM slice).
        # Older overlap_em_meta.json files may lack this field; compute from faces.
        needs_zbase = any(
            item.get('z_base_nm') is None
            for items in em_meta_list.values() for item in items
        )
        if needs_zbase:
            faces_csv = os.path.join(results_dir, 'geometric_data', 'contact_faces.csv')
            if os.path.exists(faces_csv):
                from scipy.cluster.hierarchy import linkage, fcluster
                fdf = pd.read_csv(faces_csv)
                fdf['pair_key'] = fdf.apply(
                    lambda r: tuple(sorted([r['neuron_a'], r['neuron_b']])), axis=1)
                _zb_lookup = {}
                for pk, grp in fdf.groupby('pair_key'):
                    cents = grp[['centroid_x', 'centroid_y', 'centroid_z']].values
                    if len(cents) > 1:
                        _labels = fcluster(linkage(cents, method='single'),
                                           t=10000, criterion='distance')
                    else:
                        _labels = np.array([1])
                    for cl in range(1, int(_labels.max()) + 1):
                        cz = grp['centroid_z'].values[_labels == cl]
                        _zb_lookup[(*pk, cl)] = int(np.round(cz / 40).min()) * 40
                patched = 0
                for items in em_meta_list.values():
                    for item in items:
                        if item.get('z_base_nm') is not None:
                            continue
                        key = (item['source'], item['target'], item.get('cluster_label', 1))
                        zb = _zb_lookup.get(key) or _zb_lookup.get(
                            (item['target'], item['source'], item.get('cluster_label', 1)))
                        if zb is not None:
                            item['z_base_nm'] = zb
                            patched += 1
                if patched:
                    print(f"[overlaps] Computed z_base_nm for {patched // 2} entries from contact_faces.csv")

    # per_neuron: all overlap vertices for a neuron (for 'All overlaps')
    per_neuron = {}
    for neuron in pd.concat([df['neuron_a'], df['neuron_b']]).unique():
        mask = (df['neuron_a'] == neuron) | (df['neuron_b'] == neuron)
        sub = df[mask]
        per_neuron[neuron] = {
            'x': sub['mid_x'].tolist(),
            'y': sub['mid_y'].tolist(),
            'z': sub['mid_z'].tolist(),
        }

    # per_pair: breakdown by (neuron, other) for 'Curr overlaps'
    # With spatial clustering, each sub-cluster becomes a separate entry.
    # Vertices are assigned to their nearest sub-cluster centroid.
    per_pair = {}  # {neuron: [{other, idx, z_lo, z_hi, x[], y[], z[]}, ...]}

    # Collect undirected pairs
    pair_groups = {}
    for (na, nb), grp in df.groupby(['neuron_a', 'neuron_b']):
        ukey = tuple(sorted([na, nb]))
        if ukey not in pair_groups:
            pair_groups[ukey] = grp
        else:
            pair_groups[ukey] = pd.concat([pair_groups[ukey], grp])

    _auto_idx_counter = 0  # auto-assign overlap idx when EM metadata is absent

    for (na, nb), grp in pair_groups.items():
        mx = grp['mid_x'].values
        my = grp['mid_y'].values
        mz = grp['mid_z'].values

        meta_items = em_meta_list.get((na, nb), [])

        if len(meta_items) <= 1:
            # Single cluster (or no metadata) — same as before
            meta = meta_items[0] if meta_items else {}
            em_idx = meta.get('idx', -1)
            # Auto-assign a positive idx when EM metadata is absent
            # so that overlapList gets populated for face clicking
            if em_idx < 0:
                em_idx = _auto_idx_counter
                _auto_idx_counter += 1
            z_lo = meta.get('z_lo', -20)
            z_hi = meta.get('z_hi', 20)
            valid_z = meta.get('valid_z', [])
            slice_detail = meta.get('slice_detail', [])
            area_um2 = meta.get('total_area_um2', 0)
            n_slices = meta.get('n_slices', len(valid_z) or 1)
            for neuron, other in [(na, nb), (nb, na)]:
                per_pair.setdefault(neuron, []).append({
                    'other': other,
                    'x': mx.tolist(), 'y': my.tolist(), 'z': mz.tolist(),
                    'idx': em_idx, 'z_lo': z_lo, 'z_hi': z_hi,
                    'valid_z': valid_z,
                    'slice_detail': slice_detail,
                    'area_um2': area_um2,
                    'orig_n_slices': n_slices,
                    'source': na, 'target': nb,
                    'meta_x': meta.get('x', None),
                    'meta_y': meta.get('y', None),
                    'meta_z': meta.get('z', None),
                    'z_base_nm': meta.get('z_base_nm', None),
                })
        else:
            # Multiple sub-clusters — assign each vertex to nearest centroid
            cluster_centroids = np.array([
                [m['x'], m['y'], m['z']] for m in meta_items
            ])
            pts = np.column_stack([mx, my, mz])
            # Compute distances from each point to each cluster centroid
            dists = np.linalg.norm(
                pts[:, None, :] - cluster_centroids[None, :, :], axis=2
            )
            assignments = dists.argmin(axis=1)

            for ci, meta in enumerate(meta_items):
                cmask = assignments == ci
                if cmask.sum() == 0:
                    continue
                cl_idx = meta.get('idx', -1)
                if cl_idx < 0:
                    cl_idx = _auto_idx_counter
                    _auto_idx_counter += 1
                for neuron, other in [(na, nb), (nb, na)]:
                    per_pair.setdefault(neuron, []).append({
                        'other': other,
                        'x': mx[cmask].tolist(),
                        'y': my[cmask].tolist(),
                        'z': mz[cmask].tolist(),
                        'idx': cl_idx,
                        'z_lo': meta.get('z_lo', -20),
                        'z_hi': meta.get('z_hi', 20),
                        'valid_z': meta.get('valid_z', []),
                        'slice_detail': meta.get('slice_detail', []),
                        'area_um2': meta.get('total_area_um2', 0),
                        'orig_n_slices': meta.get('n_slices', len(meta.get('valid_z', [])) or 1),
                        'source': na, 'target': nb,
                        'meta_x': meta.get('x', None),
                        'meta_y': meta.get('y', None),
                        'meta_z': meta.get('z', None),
                        'z_base_nm': meta.get('z_base_nm', None),
                    })

    total = sum(len(v['x']) for v in per_neuron.values())
    n_auto = _auto_idx_counter if not em_meta_list else 0
    print(f"[overlaps] {len(df)} overlap midpoints across "
          f"{len(per_neuron)} neurons ({total} trace points)")
    if n_auto:
        print(f"[overlaps] Auto-assigned {n_auto} overlap indices (no EM metadata)")
    return per_neuron, per_pair


def load_overlap_faces(results_dir):
    """Load overlap face triangles from contact_faces.csv for Mesh3d rendering.

    Returns:
        per_neuron_faces: {neuron: {x, y, z, i, j, k, pair_labels}}
            Mesh3d-ready data with deduplicated vertices per neuron.
            x/y/z are vertex coordinate arrays; i/j/k are face index arrays.
            pair_labels[face_idx] = 'neuronA ↔ neuronB' for hover.
        per_pair_faces: {neuron: [{other, source, target, x, y, z, i, j, k}, ...]}
            Same structure broken out per pair for "Curr overlap faces" filtering.
    """
    csv_file = os.path.join(results_dir, 'geometric_data', 'contact_faces.csv')
    if not os.path.exists(csv_file):
        print("[overlap faces] No contact_faces.csv found, skipping face viz")
        return {}, {}

    print("[overlap faces] Loading contact_faces.csv ...")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[overlap faces] Fast CSV read failed ({e}); retrying with python engine...")
        df = pd.read_csv(csv_file, engine='python')

    decimate_nm = _cfg.get('face_decimation_nm', 80)  # 0 = no decimation

    # Detect degenerate faces (all 3 vertices identical = recycling bug)
    degen = (
        (df['vertex1_x'] == df['vertex2_x']) &
        (df['vertex1_y'] == df['vertex2_y']) &
        (df['vertex1_z'] == df['vertex2_z']) &
        (df['vertex2_x'] == df['vertex3_x']) &
        (df['vertex2_y'] == df['vertex3_y']) &
        (df['vertex2_z'] == df['vertex3_z'])
    )
    n_degen = degen.sum()
    if n_degen > 0:
        print(f"[overlap faces] WARNING: {n_degen}/{len(df)} degenerate faces "
              f"(all vertices identical) — skipping those")
        df = df[~degen].copy()

    if df.empty:
        print("[overlap faces] No valid (non-degenerate) faces to render")
        return {}, {}

    def _build_mesh3d_arrays(sub_df):
        """Convert a sub-DataFrame of faces into Mesh3d x/y/z/i/j/k arrays.

        If decimate_nm > 0, rounds vertex coordinates to the nearest step
        before deduplication, merging nearby vertices to reduce triangle count.
        """
        xs, ys, zs = [], [], []
        ii, jj, kk = [], [], []
        vert_map = {}  # (x,y,z) -> index for dedup
        labels = []
        step = decimate_nm if decimate_nm > 0 else 0

        for _, row in sub_df.iterrows():
            raw_verts = [
                (row['vertex1_x'], row['vertex1_y'], row['vertex1_z']),
                (row['vertex2_x'], row['vertex2_y'], row['vertex2_z']),
                (row['vertex3_x'], row['vertex3_y'], row['vertex3_z']),
            ]
            # Apply decimation: round to nearest step
            if step:
                verts = [
                    (round(v[0] / step) * step,
                     round(v[1] / step) * step,
                     round(v[2] / step) * step)
                    for v in raw_verts
                ]
            else:
                verts = raw_verts

            idxs = []
            for v in verts:
                if v not in vert_map:
                    vert_map[v] = len(xs)
                    xs.append(v[0])
                    ys.append(v[1])
                    zs.append(v[2])
                idxs.append(vert_map[v])
            # Skip degenerate triangles (2+ vertices merged to same point)
            if idxs[0] == idxs[1] or idxs[1] == idxs[2] or idxs[0] == idxs[2]:
                continue
            ii.append(idxs[0])
            jj.append(idxs[1])
            kk.append(idxs[2])
            labels.append(f"{row['neuron_a']} \u2194 {row['neuron_b']}")

        return xs, ys, zs, ii, jj, kk, labels

    # Build per-neuron aggregate (all faces involving each neuron)
    per_neuron_faces = {}
    all_neurons = set(df['neuron_a'].unique()) | set(df['neuron_b'].unique())
    for neuron in all_neurons:
        mask = (df['neuron_a'] == neuron) | (df['neuron_b'] == neuron)
        sub = df[mask]
        if sub.empty:
            continue
        xs, ys, zs, ii, jj, kk, labels = _build_mesh3d_arrays(sub)
        per_neuron_faces[neuron] = {
            'x': xs, 'y': ys, 'z': zs,
            'i': ii, 'j': jj, 'k': kk,
            'labels': labels,
        }

    # Build per-pair breakdown for curr-overlap-faces filtering
    per_pair_faces = {}
    for (na, nb), grp in df.groupby(['neuron_a', 'neuron_b']):
        xs, ys, zs, ii, jj, kk, labels = _build_mesh3d_arrays(grp)
        entry = {
            'x': xs, 'y': ys, 'z': zs,
            'i': ii, 'j': jj, 'k': kk,
            'source': na, 'target': nb,
        }
        for neuron, other in [(na, nb), (nb, na)]:
            if neuron not in per_pair_faces:
                per_pair_faces[neuron] = []
            per_pair_faces[neuron].append({**entry, 'other': other})

    total_faces = sum(len(v['i']) for v in per_neuron_faces.values())
    total_verts = sum(len(v['x']) for v in per_neuron_faces.values())
    # Each face appears in two per-neuron traces (for neuron_a and neuron_b),
    # so the "original" count for comparison is 2 * len(df) (without decimation,
    # each trace would have 3 verts per face = len(df)*3 verts per neuron).
    orig_tris_in_traces = len(df) * 2  # one copy per involved neuron
    reduction = 100 * (1 - total_faces / orig_tris_in_traces) if orig_tris_in_traces > 0 else 0
    print(f"[overlap faces] {len(df)} CSV faces across "
          f"{len(per_neuron_faces)} neurons "
          f"({total_verts} vertices, {total_faces} triangles in traces)")
    if decimate_nm > 0:
        print(f"[overlap faces] Decimation at {decimate_nm}nm: "
              f"{orig_tris_in_traces} -> {total_faces} trace triangles "
              f"({reduction:.1f}% reduction, "
              f"{total_verts} deduplicated vertices)")
    return per_neuron_faces, per_pair_faces


def load_overlap_table(results_dir):
    """Build overlap summary table from all_results_combined.csv.

    Counts Top1-N patches per pair for patch count column.
    """
    csv_file = os.path.join(results_dir, 'all_results_combined.csv')
    df = pd.read_csv(csv_file)
    contacts = df[df['Has_Contact'] == True]

    n_top = _cfg.get('top_patches', 10)

    table = []
    for _, row in contacts.iterrows():
        src = row['Source_Neuron']
        tgt = row['Target_Neuron']

        n_patches = 0
        for pn in range(1, n_top + 1):
            col = f'Top{pn}_Patch_Centroid_X'
            if col in df.columns and not pd.isna(row.get(col, np.nan)):
                n_patches += 1

        area = float(row['Contact_Area_um2']) \
            if not pd.isna(row.get('Contact_Area_um2', np.nan)) else 0.0
        table.append({
            'source': src,
            'target': tgt,
            'area': area,
            'patches': n_patches,
            'status': 'active',
        })

    print(f"[table] Built overlap table with {len(table)} pairs")
    return table


def load_mesh_skeleton(neuron_name, mesh_dir, max_points=15000):
    """Load a neuron surface mesh and return a subsampled point cloud.

    The mesh file is looked up by numeric neuron ID (from ``NEURON_IDS``), so
    the file must be named ``<neuron_id>.obj`` inside *mesh_dir*.

    Args:
        neuron_name:  Canonical neuron name (e.g. ``'VS1_L'``).
        mesh_dir:     Directory containing ``.obj`` files.
        max_points:   Maximum vertices to keep.  If the mesh has more,
                      every ``N//max_points``-th vertex is kept so the Plotly
                      Scatter3d trace stays responsive (default 15 000).

    Returns:
        ``(x, y, z)`` numpy float32 arrays, or ``(None, None, None)`` if the
        neuron is not in ``NEURON_IDS`` or the file does not exist.
    """
    neuron_id = {v: k for k, v in NEURON_IDS.items()}.get(neuron_name)
    if neuron_id is None:
        return None, None, None
    mesh_file = os.path.join(mesh_dir, f"{neuron_id}.obj")
    if not os.path.exists(mesh_file):
        return None, None, None

    try:
        mesh = trimesh.load(mesh_file)
        vertices = mesh.vertices
        if len(vertices) > max_points:
            step = max(1, len(vertices) // max_points)
            vertices = vertices[::step]
        return (np.array(vertices[:, 0], dtype=np.float32),
                np.array(vertices[:, 1], dtype=np.float32),
                np.array(vertices[:, 2], dtype=np.float32))
    except Exception as e:
        print(f"  [!] Error loading mesh for {neuron_name}: {e}")
        return None, None, None


# ── Build Plotly figure ───────────────────────────────────────────────

def build_figure(mesh_dir):
    """Build the Plotly 3D figure, loading all data sources.

    Creates one group of traces per neuron, in this order:

    1. ``<name>_mesh``              — Scatter3d point cloud (subsampled mesh)
    2. ``<name>_contacts``          — Scatter3d contact patch centroids (red)
    3. ``<name>_contacts_hl``       — empty highlight Scatter3d (JS populates on click)
    4. ``<name>_synapses``          — Scatter3d synapses (yellow=excitatory, blue=inhibitory)
    5. ``<name>_synapses_hl``       — empty highlight Scatter3d
    6. ``<name>_alloverlaps``       — Scatter3d all overlap midpoints
    7. ``<name>_curoverlaps``       — Scatter3d current-pair overlap midpoints
    8. ``<name>_alloverlapfaces``   — Mesh3d aggregate overlap triangles
    9. ``<name>_curroverlapfaces``  — Mesh3d current-pair triangles

    Plus three global traces at the end:
    * ``_pos_indicator``  — diamond at the currently selected EM location
    * ``_gap_junctions``  — cyan spheres at annotated GJ sites

    Returns:
        ``(fig, contacts, synapses, trace_info, overlap_pairs, overlap_pair_faces)``
        where ``trace_info`` maps ``'<name>_<kind>'`` keys to integer Plotly trace
        indices, and the overlap dicts are passed straight through to
        ``generate_html()``.
    """
    contacts = load_contacts(RESULTS_DIR)
    synapses = load_synapses(RESULTS_DIR)
    overlap_verts, overlap_pairs = load_overlap_vertices(RESULTS_DIR)
    overlap_faces, overlap_pair_faces = load_overlap_faces(RESULTS_DIR)

    traces = []
    trace_info = {}

    print("\n[meshes] Loading neuron data...")
    for neuron_name in sorted(VIEWER_NEURONS):
        color = NEURON_COLORS.get(neuron_name, '#888888')

        # ── 1. Mesh (point cloud) ────────────────────────
        x, y, z = load_mesh_skeleton(neuron_name, mesh_dir)
        if x is not None:
            trace_info[f"{neuron_name}_mesh"] = len(traces)
            traces.append(go.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                name=f"{neuron_name}_mesh",
                marker=dict(size=1, color=color, opacity=0.15),
                visible=False,
                hovertemplate=f'{neuron_name}<extra></extra>',
                legendgroup=neuron_name
            ))
            print(f"  {neuron_name}: mesh ({len(x)} pts)")

        # ── 2. Contact traces ────────────────────────────
        neuron_contacts = contacts[
            (contacts['source'] == neuron_name) |
            (contacts['target'] == neuron_name)
        ]
        if not neuron_contacts.empty:
            N = len(neuron_contacts)
            trace_info[f"{neuron_name}_contacts"] = len(traces)
            traces.append(go.Scatter3d(
                x=neuron_contacts['x'],
                y=neuron_contacts['y'],
                z=neuron_contacts['z'],
                mode='markers',
                name=f'{neuron_name}_contacts',
                visible=False,
                marker=dict(size=7, color='white', opacity=0.98,
                            symbol='circle-open',
                            line=dict(color='red', width=3)),
                customdata=np.column_stack([
                    neuron_contacts['x'],
                    neuron_contacts['y'],
                    neuron_contacts['z'],
                    np.full(N, 'contact'),
                    neuron_contacts['source'],
                    neuron_contacts['target'],
                    neuron_contacts['idx'],
                    neuron_contacts['patch_num']]),
                hovertemplate=(
                    'Contact #%{customdata[6]}<br>'
                    '%{customdata[4]} \u2192 %{customdata[5]}'
                    '<extra></extra>'),
                legendgroup=neuron_name
            ))
            # Highlight trace
            trace_info[f"{neuron_name}_contacts_highlight"] = len(traces)
            traces.append(go.Scatter3d(
                x=[], y=[], z=[], mode='markers',
                name=f'{neuron_name}_contacts_highlight',
                visible=False,
                marker=dict(size=12, color='red', opacity=1.0,
                            symbol='circle'),
                hovertemplate='SELECTED Contact<extra></extra>',
                legendgroup=neuron_name, showlegend=False
            ))
            print(f"  {neuron_name}: {N} contacts")

        # ── 3. Synapse traces ────────────────────────────
        neuron_synapses = synapses[
            (synapses['source'] == neuron_name) |
            (synapses['target'] == neuron_name)
        ]
        if not neuron_synapses.empty:
            M = len(neuron_synapses)
            synapse_indices = neuron_synapses['index'].values
            # Per-point colors: blue for inhibitory (GABA), yellow for excitatory
            syn_colors = [
                '#4488ff' if (row['source'], row['target']) in INH_PAIRS
                else 'yellow'
                for _, row in neuron_synapses.iterrows()
            ]
            syn_labels = [
                'Inhibitory (GABA)' if (row['source'], row['target']) in INH_PAIRS
                else 'Excitatory (ACh/Glut)'
                for _, row in neuron_synapses.iterrows()
            ]
            trace_info[f"{neuron_name}_synapses"] = len(traces)
            traces.append(go.Scatter3d(
                x=neuron_synapses['x'],
                y=neuron_synapses['y'],
                z=neuron_synapses['z'],
                mode='markers',
                name=f'{neuron_name}_synapses',
                visible=False,
                marker=dict(size=4, color=syn_colors, opacity=0.8),
                customdata=np.column_stack([
                    neuron_synapses['x'],
                    neuron_synapses['y'],
                    neuron_synapses['z'],
                    np.full(M, 'synapse'),
                    neuron_synapses['source'],
                    neuron_synapses['target'],
                    synapse_indices,
                    syn_labels]),
                hovertemplate=(
                    'Synapse<br>%{customdata[4]} \u2192 %{customdata[5]}'
                    '<br>%{customdata[7]}'
                    '<extra></extra>'),
                legendgroup=neuron_name
            ))
            # Highlight trace
            trace_info[f"{neuron_name}_synapses_highlight"] = len(traces)
            traces.append(go.Scatter3d(
                x=[], y=[], z=[], mode='markers',
                name=f'{neuron_name}_synapses_highlight',
                visible=False,
                marker=dict(size=15, color='yellow', opacity=1.0,
                            symbol='circle'),
                hovertemplate='SELECTED Synapse<extra></extra>',
                legendgroup=neuron_name, showlegend=False
            ))
            print(f"  {neuron_name}: {M} synapses")

        # ── 4. Overlap vertex trace (red) ────────────────
        #   all-overlaps: shows all overlap vertices for this neuron
        #   curr-overlaps: empty initially, JS fills via overlap_pairs data
        if neuron_name in overlap_verts:
            # Build customdata for all overlap vertices of this neuron.
            # Each vertex gets its pair's EM idx, source, target, z_lo, z_hi.
            pairs_for_neuron = overlap_pairs.get(neuron_name, [])
            all_x, all_y, all_z = [], [], []
            all_cd = []
            for p in pairs_for_neuron:
                n_pts = len(p['x'])
                all_x.extend(p['x'])
                all_y.extend(p['y'])
                all_z.extend(p['z'])
                for i in range(n_pts):
                    all_cd.append([
                        p['x'][i], p['y'][i], p['z'][i],
                        'overlap',
                        p['source'], p['target'],
                        p['idx'], p.get('z_lo', -20), p.get('z_hi', 20)
                    ])

            trace_info[f"{neuron_name}_alloverlaps"] = len(traces)
            traces.append(go.Scatter3d(
                x=all_x, y=all_y, z=all_z,
                mode='markers',
                name=f'{neuron_name}_alloverlaps',
                visible=False,
                marker=dict(size=2,
                            color='rgba(255,40,40,0.7)',
                            opacity=0.7),
                customdata=all_cd,
                hovertemplate=(
                    'Overlap %{customdata[4]} ↔ %{customdata[5]}'
                    '<extra></extra>'),
                legendgroup=neuron_name, showlegend=False
            ))
            # Curr overlaps trace (empty, populated by JS with customdata)
            trace_info[f"{neuron_name}_curoverlaps"] = len(traces)
            traces.append(go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                name=f'{neuron_name}_curoverlaps',
                visible=False,
                marker=dict(size=2,
                            color='rgba(255,80,80,0.9)',
                            opacity=0.9),
                hovertemplate=(
                    'Overlap %{customdata[4]} ↔ %{customdata[5]}'
                    '<extra></extra>'),
                legendgroup=neuron_name, showlegend=False
            ))
            print(f"  {neuron_name}: {len(all_x)} overlap vertices (clickable)")

        # ── 5. Overlap face surfaces (Mesh3d, semi-transparent red) ────
        #   all-overlapfaces: shows all overlap face triangles for this neuron
        #   curr-overlapfaces: empty initially, JS fills from overlapPairFaces data
        if neuron_name in overlap_faces:
            fdata = overlap_faces[neuron_name]
            trace_info[f"{neuron_name}_alloverlapfaces"] = len(traces)
            traces.append(go.Mesh3d(
                x=fdata['x'], y=fdata['y'], z=fdata['z'],
                i=fdata['i'], j=fdata['j'], k=fdata['k'],
                color='rgba(255, 0, 0, 1.0)',
                opacity=1.0,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.3, specular=0.1),
                lightposition=dict(x=0, y=0, z=100000),
                name=f'{neuron_name}_alloverlapfaces',
                visible=False,
                hovertemplate='Overlap face<extra></extra>',
                showlegend=False,
            ))
            # Curr overlap faces trace (empty Mesh3d, JS fills it)
            trace_info[f"{neuron_name}_curroverlapfaces"] = len(traces)
            traces.append(go.Mesh3d(
                x=[], y=[], z=[],
                i=[], j=[], k=[],
                color='rgba(255, 0, 0, 1.0)',
                opacity=1.0,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.3, specular=0.1),
                lightposition=dict(x=0, y=0, z=100000),
                name=f'{neuron_name}_curroverlapfaces',
                visible=False,
                hovertemplate='Overlap face<extra></extra>',
                showlegend=False,
            ))
            print(f"  {neuron_name}: {len(fdata['i'])} overlap face triangles")

    # ── 6. Position indicator (single point, updated by JS) ──────
    trace_info['_pos_indicator'] = len(traces)
    traces.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        name='_pos_indicator',
        visible=False,
        marker=dict(size=16, color='rgba(0,0,0,0)',
                    symbol='diamond-open',
                    line=dict(color='black', width=3)),
        hovertemplate='Current EM location<extra></extra>',
        showlegend=False
    ))
    print(f"  [pos_indicator] 3D position marker trace added")

    # ── 7. Gap-junction markers (empty, JS fills) ───────────────
    trace_info['_gap_junctions'] = len(traces)
    traces.append(go.Scatter3d(
        x=[], y=[], z=[],
        mode='markers',
        name='_gap_junctions',
        visible=True,
        marker=dict(size=4, color='#39FF14', symbol='circle',
                    opacity=0.9),
        hovertemplate='Gap junction<br>%{text}<extra></extra>',
        text=[],
        showlegend=False
    ))
    print(f"  [gap_junctions] marker trace added")

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (nm)', showbackground=False),
            yaxis=dict(title='Y (nm)', showbackground=False),
            zaxis=dict(title='Z (nm)', showbackground=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.1))
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=800
    )

    return fig, contacts, synapses, trace_info, overlap_pairs, overlap_pair_faces


# ── Snapshot indexing ─────────────────────────────────────────────────

def index_em_snapshots(em_snap_dir, contacts, synapses, results_dir):
    """Index EM snapshots as relative file paths (not base64, to keep HTML small)."""
    import re
    snapshot_map = {'contact': {}, 'synapse': {}, 'overlap': {}}
    zstack_map = {'contact': {}, 'synapse': {}, 'overlap': {}}
    em_files = os.listdir(em_snap_dir) if os.path.isdir(em_snap_dir) else []

    def _coord_tag_xyz(x, y, z):
        return f"x{int(round(x))}_y{int(round(y))}_z{int(round(z))}"

    def _find_segmented(kind, idx, expected_coord=None):
        idx = int(idx)
        if expected_coord is not None:
            expected = f"{kind}_{idx}_{expected_coord}_segmented.png"
            if expected in em_files:
                return expected
        legacy = f"{kind}_{idx}_segmented.png"
        if legacy in em_files:
            return legacy
        pat = re.compile(rf"^{re.escape(kind)}_{idx}_.*_segmented\.png$")
        for fname in em_files:
            if pat.match(fname):
                return fname
        return None

    # Load cluster mapping (generate_em_stacks.py now clusters nearby patches)
    cluster_map_file = os.path.join(results_dir, 'contact_cluster_map.json')
    contact_cluster_map = {}  # patch_idx -> cluster_id
    if os.path.exists(cluster_map_file):
        with open(cluster_map_file, 'r') as cmf:
            cluster_data = json.load(cmf)
        for p in cluster_data.get('patches', []):
            contact_cluster_map[p['idx']] = p['cluster_id']
        print(f"[snapshots] Loaded cluster map: {len(contact_cluster_map)} patches -> "
              f"{cluster_data.get('n_clusters', '?')} clusters")

    # Index contact center images as relative paths
    indexed_clusters = set()
    for idx in contacts['idx'].unique():
        idx_int = int(idx)
        cid = contact_cluster_map.get(idx_int, None)
        if cid is not None:
            fname = f"cluster_{cid}_segmented.png"
            if os.path.exists(os.path.join(em_snap_dir, fname)):
                snapshot_map['contact'][idx_int] = f"em_snaps/{fname}"
                indexed_clusters.add(cid)
        else:
            fname = _find_segmented('contact', idx_int)
            if fname:
                snapshot_map['contact'][idx_int] = f"em_snaps/{fname}"
    print(f"[snapshots] Indexed {len(snapshot_map['contact'])} contacts "
          f"({len(indexed_clusters)} unique clusters) [file paths]")

    for idx in synapses['index'].unique():
        idx_int = int(idx)
        fname = _find_segmented('synapse', idx_int)
        if fname:
            snapshot_map['synapse'][idx_int] = f"em_snaps/{fname}"
    print(f"[snapshots] Indexed {len(snapshot_map['synapse'])} synapses [file paths]")

    meta_file = os.path.join(results_dir, 'overlap_em_meta.json')
    overlap_meta_by_idx = {}
    overlap_expected_zfiles = {}  # idx -> {z_off: filename}
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as mf:
            overlap_meta = json.load(mf)
        for item in overlap_meta:
            idx = int(item['idx'])
            overlap_meta_by_idx[idx] = item

            # Center image uses z_offset==0 slice center when present.
            center_sd = None
            for sd in item.get('slice_detail', []):
                if int(sd.get('z_offset', 999999)) == 0:
                    center_sd = sd
                    break
            if center_sd is not None:
                center_coord = _coord_tag_xyz(center_sd.get('cx', item.get('x', 0)),
                                              center_sd.get('cy', item.get('y', 0)),
                                              item.get('z_base_nm', item.get('z', 0)))
            else:
                center_coord = _coord_tag_xyz(item.get('x', 0), item.get('y', 0),
                                              item.get('z_base_nm', item.get('z', 0)))

            fname = _find_segmented('overlap', idx, center_coord)
            if fname:
                snapshot_map['overlap'][idx] = f"em_snaps/{fname}"

            # Build expected coordinate-tagged z-stack files from slice_detail.
            zfiles = {}
            z_base_nm = item.get('z_base_nm', 0)
            for sd in item.get('slice_detail', []):
                rel = int(sd.get('z_offset', 0))
                coord = _coord_tag_xyz(sd.get('cx', item.get('x', 0)),
                                       sd.get('cy', item.get('y', 0)),
                                       z_base_nm + rel * 40)
                sign = '+' if rel >= 0 else '-'
                zname = f"overlap_{idx}_{coord}_z{sign}{abs(rel):03d}.png"
                zfiles[rel] = zname
            overlap_expected_zfiles[idx] = zfiles
    print(f"[snapshots] Indexed {len(snapshot_map['overlap'])} overlaps [file paths]")

    # Index z-stack images for dynamic filename resolution in JS.
    # Supports both legacy names (kind_idx_z+NNN.png) and coordinate-tagged names
    # (kind_idx_x..._y..._z..._z+NNN.png).
    zpat = re.compile(r'^(contact|synapse|overlap)_(\d+)(?:_.*)?_z([+-])(\d+)\.png$')

    # Prefer overlap z-stack filenames that exactly match current overlap metadata.
    for idx, zmap in overlap_expected_zfiles.items():
        for zoff, fname in zmap.items():
            if fname in em_files:
                zstack_map['overlap'].setdefault(idx, {})[zoff] = f"em_snaps/{fname}"

    for fname in em_files:
        m = zpat.match(fname)
        if not m:
            continue
        kind = m.group(1)
        idx = int(m.group(2))
        sign = 1 if m.group(3) == '+' else -1
        zoff = sign * int(m.group(4))
        # Preserve preferred overlap mapping if already set from metadata match.
        if kind == 'overlap' and zoff in zstack_map['overlap'].get(idx, {}):
            continue
        zstack_map.setdefault(kind, {}).setdefault(idx, {})[zoff] = f"em_snaps/{fname}"

    print(f"[snapshots] Indexed z-stacks: "
          f"contact={sum(len(v) for v in zstack_map['contact'].values())}, "
          f"synapse={sum(len(v) for v in zstack_map['synapse'].values())}, "
          f"overlap={sum(len(v) for v in zstack_map['overlap'].values())}")

    return snapshot_map, zstack_map


# ── HTML generation ───────────────────────────────────────────────────

def generate_html(fig, contacts, synapses, trace_info,
                   overlap_pairs, overlap_pair_faces, em_snap_dir):
    """Serialise all data into the HTML template and return the full HTML string.

    Template placeholders substituted (all in ``HTML_TEMPLATE``):

    * ``{SNAPSHOT_JSON}``          — ``{kind: {idx: 'em_snaps/...png'}}`` paths
    * ``{SNAPSHOT_ZMAP_JSON}``     — ``{kind: {idx: {z_off: 'em_snaps/...png'}}}``
    * ``{CLUSTER_MAP_JSON}``       — ``{patch_idx: cluster_id}`` mapping
    * ``{NEURON_NAMES_JSON}``      — ordered list of neuron names
    * ``{TRACE_INFO_JSON}``        — ``{name_kind: trace_index}`` lookup
    * ``{CONTACT_LIST_JSON}``      — list of contact patch records
    * ``{SYNAPSE_LIST_JSON}``      — list of synapse records with isInh flag
    * ``{OVERLAP_LIST_JSON}``      — list of overlap records with valid_z, area etc.
    * ``{OVERLAP_TABLE_JSON}``     — summary rows used by the heatmap matrix
    * ``{OVERLAP_PAIRS_JSON}``     — per-neuron list of overlap pairs with coords
    * ``{OVERLAP_PAIR_FACES_JSON}``— per-neuron Mesh3d face data per pair
    * ``{NEURON_COLORS_JSON}``     — ``{name: '#rrggbb'}`` color map
    * ``{CHECKBOXES_HTML}``        — sidebar neuron control checkboxes
    * ``{PLOT_DIV}``               — Plotly figure HTML + inline JSON

    Returns:
        Complete self-contained HTML string (~4 MB typical).
    """
    snapshot_mapping, zstack_mapping = index_em_snapshots(
        em_snap_dir, contacts, synapses, RESULTS_DIR)
    overlap_table = load_overlap_table(RESULTS_DIR)

    plot_div = fig.to_html(
        full_html=False, include_plotlyjs=False,
        div_id='plotly3d',
        config={'displayModeBar': True, 'displaylogo': False}
    )

    # Build sidebar checkboxes
    neuron_list = sorted(VIEWER_NEURONS)
    neuron_checkboxes = []
    for neuron in neuron_list:
        color = NEURON_COLORS.get(neuron, '#888888')
        neuron_checkboxes.append(
            f'<div class="neuron-group">'
            f'<span class="neuron-name" style="color:{color}">{neuron}</span>'
            f'<div class="neuron-controls">'
            f'<label><input type="checkbox" id="mesh_{neuron}"> Mesh</label>'
            f'<label><input type="checkbox" id="allcontacts_{neuron}">'
            f' All contacts</label>'
            f'<label><input type="checkbox" id="curcontacts_{neuron}">'
            f' Curr contacts</label>'
            f'<label><input type="checkbox" id="alloverlaps_{neuron}">'
            f' All overlaps</label>'
            f'<label><input type="checkbox" id="curoverlaps_{neuron}">'
            f' Curr overlaps</label>'
            f'<label><input type="checkbox" id="alloverlapfaces_{neuron}">'
            f' All putative GJs</label>'
            f'<label><input type="checkbox" id="curroverlapfaces_{neuron}">'
            f' Curr putative GJs</label>'
            f'<label><input type="checkbox" id="allsynapses_{neuron}">'
            f' All synapses</label>'
            f'<label><input type="checkbox" id="cursynapses_{neuron}">'
            f' Curr synapses</label>'
            f'</div></div>'
        )

    global_controls = (
        '<div class="neuron-group" style="margin-bottom:12px;">'
        '<span class="neuron-name" style="color:#FFD400">Display</span>'
        '<div class="neuron-controls">'
        '<label><input type="checkbox" id="toggleAxes" checked>'
        ' Axis labels</label>'
        '</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;">'
        '<button id="btnAllMesh" style="font-size:9px;padding:2px 6px;background:#444;'
        'color:#ccc;border:1px solid #666;border-radius:3px;cursor:pointer;"'
        ' title="Toggle all neuron meshes">All Neurons</button>'
        '<button id="btnAllOverlaps" style="font-size:9px;padding:2px 6px;background:#444;'
        'color:#ccc;border:1px solid #666;border-radius:3px;cursor:pointer;"'
        ' title="Toggle all overlap vertices">All Overlaps</button>'
        '<button id="btnAllOvFaces" style="font-size:9px;padding:2px 6px;background:#444;'
        'color:#ccc;border:1px solid #666;border-radius:3px;cursor:pointer;"'
        ' title="Toggle all putative gap junctions">All Put. GJs</button>'
        '<button id="btnAllSynapses" style="font-size:9px;padding:2px 6px;background:#444;'
        'color:#ccc;border:1px solid #666;border-radius:3px;cursor:pointer;"'
        ' title="Toggle all synapses">All Synapses</button>'
        '</div>'
        '</div>'
    )

    checkboxes_html = global_controls + '\n'.join(neuron_checkboxes)

    # Build contact list with patch_area
    contact_list = contacts[[
        'idx', 'x', 'y', 'z', 'source', 'target', 'patch_area'
    ]].to_dict('records')
    # Inhibitory pairs: add isInh field to synapse list entries
    synapse_list = synapses[['index', 'x', 'y', 'z', 'source', 'target']] \
        .rename(columns={'index': 'idx'}).to_dict('records')
    for s in synapse_list:
        s['isInh'] = 1 if (s['source'], s['target']) in INH_PAIRS else 0

    # Build overlap_pairs JSON for JS curr-overlap filtering
    # {neuron: [{other, idx, z_lo, z_hi, source, target, x[], y[], z[]}, ...]}
    # Also build a flat overlapList for item navigation
    overlap_pairs_json = {}
    seen_overlap_idxs = set()
    overlap_list = []
    for neuron, pairs in overlap_pairs.items():
        overlap_pairs_json[neuron] = [
            {'other': p['other'],
             'idx': p.get('idx', -1),
             'z_lo': p.get('z_lo', -20),
             'z_hi': p.get('z_hi', 20),
             'valid_z': p.get('valid_z', []),
             'source': p.get('source', ''),
             'target': p.get('target', ''),
             'x': p['x'], 'y': p['y'], 'z': p['z']}
            for p in pairs
        ]
        for p in pairs:
            ov_idx = p.get('idx', -1)
            if ov_idx >= 0 and ov_idx not in seen_overlap_idxs:
                seen_overlap_idxs.add(ov_idx)
                # Build per-slice coordinate lookup from slice_detail.
                # Each EM slice was downloaded at its own (cx, cy) center,
                # so the diamond must track that position per Z-offset.
                sd_list = p.get('slice_detail', [])
                slice_coords = {}
                for sd in sd_list:
                    zo = sd.get('z_offset')
                    if zo is not None and 'cx' in sd and 'cy' in sd:
                        slice_coords[str(zo)] = [round(sd['cx'], 1), round(sd['cy'], 1)]
                # z_base_nm = absolute Z (nm) of the first EM slice (z_offset=0).
                # This replaces meta_z as the base for Z-offset calculations.
                z_base = p.get('z_base_nm')
                if z_base is None:
                    # Fallback: use meta_z (less accurate)
                    meta_z = p.get('meta_z')
                    z_base = meta_z if meta_z is not None else (
                        sum(p['z']) / len(p['z']) if p['z'] else 0)
                overlap_list.append({
                    'idx': ov_idx,
                    'x': slice_coords.get('0', [0, 0])[0] if slice_coords else (
                        sum(p['x']) / len(p['x']) if p['x'] else 0),
                    'y': slice_coords.get('0', [0, 0])[1] if slice_coords else (
                        sum(p['y']) / len(p['y']) if p['y'] else 0),
                    'z': z_base,
                    'source': p.get('source', ''),
                    'target': p.get('target', ''),
                    'z_lo': p.get('z_lo', -20),
                    'z_hi': p.get('z_hi', 20),
                    'valid_z': p.get('valid_z', []),
                    'area_um2': p.get('area_um2', 0),
                    'orig_n_slices': p.get('orig_n_slices', len(p.get('valid_z', [])) or 1),
                    'slice_coords': slice_coords,
                })
    overlap_list.sort(key=lambda o: o['idx'])

    # ── Populate valid_z from actual files on disk ───────────────
    # overlap_em_meta.json may have empty valid_z; scan em_snaps/ for real files
    import re as _re
    _ov_file_re = _re.compile(r'^overlap_(\d+)(?:_.*)?_z([+-])(\d+)\.png$')
    _ov_center_legacy_re = _re.compile(r'^overlap_(\d+)_segmented\.png$')
    _ov_center_coord_re = _re.compile(r'^overlap_(\d+)_.*_segmented\.png$')
    _ov_files_by_idx = {}  # idx -> set of z-offsets
    if os.path.isdir(em_snap_dir):
        for fname in os.listdir(em_snap_dir):
            m = _ov_file_re.match(fname)
            if m:
                ov_idx = int(m.group(1))
                sign = 1 if m.group(2) == '+' else -1
                offset = int(m.group(3))
                _ov_files_by_idx.setdefault(ov_idx, set()).add(sign * offset)
            # Also check for center image (z=0)
            else:
                m0 = _ov_center_legacy_re.match(fname) or _ov_center_coord_re.match(fname)
                if m0:
                    ov_idx = int(m0.group(1))
                    _ov_files_by_idx.setdefault(ov_idx, set()).add(0)
    # Patch overlap_list and overlap_pairs_json with file-derived valid_z
    for ov in overlap_list:
        file_zs = _ov_files_by_idx.get(ov['idx'], set())
        if file_zs:
            ov['valid_z'] = sorted(file_zs)
            ov['z_lo'] = min(file_zs)
            ov['z_hi'] = max(file_zs)
        elif not ov.get('valid_z'):
            ov['valid_z'] = [0]  # fallback: at least center
    for neuron, pairs in overlap_pairs_json.items():
        for p in pairs:
            ov_idx = p.get('idx', -1)
            file_zs = _ov_files_by_idx.get(ov_idx, set())
            if file_zs:
                p['valid_z'] = sorted(file_zs)
                p['z_lo'] = min(file_zs)
                p['z_hi'] = max(file_zs)
            elif not p.get('valid_z'):
                p['valid_z'] = [0]
    _patched_count = sum(1 for ov in overlap_list if len(ov.get('valid_z', [])) > 1)
    print(f"[valid_z] Patched {_patched_count}/{len(overlap_list)} overlaps "
          f"with file-derived valid_z from {em_snap_dir}")

    # Assemble final HTML
    html = HTML_TEMPLATE
    html = html.replace('{CHECKBOXES_HTML}', checkboxes_html)
    html = html.replace('{PLOT_DIV}', plot_div)
    # Build contact cluster map (patch_idx -> cluster_id) for JS
    cluster_map_file = os.path.join(RESULTS_DIR, 'contact_cluster_map.json')
    contact_cluster_js = {}
    if os.path.exists(cluster_map_file):
        with open(cluster_map_file, 'r') as cmf:
            cluster_data = json.load(cmf)
        for p in cluster_data.get('patches', []):
            contact_cluster_js[p['idx']] = p['cluster_id']

    html = html.replace('{SNAPSHOT_JSON}', json.dumps(snapshot_mapping))
    html = html.replace('{SNAPSHOT_ZMAP_JSON}', json.dumps(zstack_mapping))
    html = html.replace('{CLUSTER_MAP_JSON}', json.dumps(contact_cluster_js))
    html = html.replace('{NEURON_NAMES_JSON}', json.dumps(neuron_list))
    html = html.replace('{TRACE_INFO_JSON}', json.dumps(trace_info))
    html = html.replace('{CONTACT_LIST_JSON}', json.dumps(contact_list))
    html = html.replace('{SYNAPSE_LIST_JSON}', json.dumps(synapse_list))
    html = html.replace('{OVERLAP_LIST_JSON}', json.dumps(overlap_list))
    html = html.replace('{OVERLAP_TABLE_JSON}', json.dumps(overlap_table))
    html = html.replace('{OVERLAP_PAIRS_JSON}', json.dumps(overlap_pairs_json))

    # Build overlap pair faces JSON for Mesh3d curr-overlap-faces filtering
    # {neuron: [{other, source, target, x[], y[], z[], i[], j[], k[]}, ...]}
    overlap_pair_faces_json = {}
    for neuron, pairs in overlap_pair_faces.items():
        overlap_pair_faces_json[neuron] = [
            {'other': p['other'],
             'source': p.get('source', ''),
             'target': p.get('target', ''),
             'x': p['x'], 'y': p['y'], 'z': p['z'],
             'i': p['i'], 'j': p['j'], 'k': p['k']}
            for p in pairs
        ]
    html = html.replace('{OVERLAP_PAIR_FACES_JSON}',
                         json.dumps(overlap_pair_faces_json))
    html = html.replace('{NEURON_COLORS_JSON}', json.dumps(NEURON_COLORS))

    return html


# ── Main ──────────────────────────────────────────────────────────────

def main():
    """Entry point: build and write the interactive HTML viewer.

    Pipeline:
        1. Build Plotly figure (loads all CSVs and mesh OBJs).
        2. Generate HTML (serialise data, substitute template placeholders).
        3. Write ``skeleton_em_viewer.html`` into ``RESULTS_DIR``.

    The output file is completely self-contained — it requires only that the
    browser can resolve relative paths ``em_snaps/*.png`` from the same folder.
    """
    print("=" * 70)
    print("EM Viewer - Generating Interactive Viewer")
    print("=" * 70)
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Neurons.json: {len(NEURON_CFG)} neurons, "
          f"{len(VIEWER_NEURONS)} in viewer")

    mesh_dir = os.path.join(RESULTS_DIR, 'neuron_meshes')
    em_snap_dir = os.path.join(RESULTS_DIR, 'em_snaps')

    print("\n[1/3] Building 3D figure with highlighting...")
    fig, contacts, synapses, trace_info, overlap_pairs, overlap_pair_faces = \
        build_figure(mesh_dir)

    print("\n[2/3] Generating HTML viewer...")
    html = generate_html(fig, contacts, synapses, trace_info,
                         overlap_pairs, overlap_pair_faces, em_snap_dir)

    print("\n[3/3] Writing output...")
    output_file = os.path.join(RESULTS_DIR, 'skeleton_em_viewer.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n{'=' * 70}")
    print(f"Complete! ({size_mb:.1f} MB)")
    print(f"Open: {output_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()


