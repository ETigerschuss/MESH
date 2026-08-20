"""
Skeleton EM Viewer — interactive 3D + EM contact proofreading
============================================================

A self-contained, interactive 3D + 2D viewer for inspecting and proofreading
neuron-neuron contacts in electron microscopy.  One Python run
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

SCIENTIFIC PROVENANCE
---------------------
Dataset provenance follows the FlyWire reconstruction on FAFB v141. The chemical
synapse table is read from FlyWire/CAVE via ``fafbseg.flywire.synapses``
(automated predictions of Buhmann et al. 2021, cleft score >= 50).

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
* Measurement tool — trace an apposed membrane for length / area (8 nm/px)

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

# ── Publication: pre-selected putative gap-junction sites ──────────────
# Marked on load with the green putative-GJ marker style. The confirmed site is
# the MOT_R <-> HSN_R junction (FlyWire voxel 154698, 66954, 5068 @ 4x4x40 nm =
# the nm below). Additional curated example sites are loaded from the GJ figure
# pipeline output (gj_figures/gj_sites.json) when present.
CONFIRMED_GJ_NM = (618792, 267816, 202720)   # MOT_R <-> HSN_R (confirmed)


# Minimum spacing between distinct putative-GJ markers of the same pair (nm).
# Collapses the many small validated patches of one contact into a few
# representative sites so the viewer isn't flooded with near-duplicate markers.
_GJ_MARKER_SEP_NM = 3000.0


def _load_preselected_gj(results_dir):
    """[(x_nm, y_nm, z_nm, label), ...]: the confirmed GJ first (drawn larger),
    then putative gap-junction sites.

    Preferred source is ``gj_candidates.csv`` (Phase A): seg-validated
    motor<->LPTC appositions from which contacts coinciding with an annotated
    chemical synapse have been removed, i.e. the chemically-filtered
    gap-junction candidates, already deduplicated to distinct sites and ranked
    by area. Falls back to ``validated_patches.csv`` (all real appositions,
    deduplicated here) and then to the motor<->LPTC subset of
    ``overlap_em_meta.json``. Motor/descending neurons never enter the lobula
    plate, so all of these lie on the LPTC *axon* where MOT/MOS gap junctions
    are expected."""
    import csv
    sites = [(CONFIRMED_GJ_NM[0], CONFIRMED_GJ_NM[1], CONFIRMED_GJ_NM[2],
              "Confirmed GJ (MOT_R <-> HSN_R)")]

    def _is_motor_partner(a, b):
        names = (str(a), str(b))
        has_motor = any(n.startswith(("MOT", "MOS")) for n in names)
        has_lptc = any(n.startswith(("HS", "VS")) for n in names)
        return has_motor and has_lptc

    # Preferred: chemically-filtered candidates from phase_a_gj_candidates.py.
    cand = os.path.join(results_dir, "gj_candidates.csv")
    if os.path.exists(cand):
        try:
            rows = list(csv.DictReader(open(cand, newline="")))
            rows.sort(key=lambda r: -float(r.get("area_um2", 0.0)))
            for r in rows:
                a, b = r.get("neuron_a"), r.get("neuron_b")
                chem = float(r.get("nearest_chem_nm", "inf") or "inf")
                tag = "no chem synapse" if chem == float("inf") else \
                      "chem {:.1f} um".format(chem / 1000.0)
                sites.append((float(r["x"]), float(r["y"]), float(r["z"]),
                              "{} <-> {} (GJ candidate; area {} um2; {})".format(
                                  a, b, r.get("area_um2", "?"), tag)))
            return sites
        except Exception:
            pass

    val = os.path.join(results_dir, "validated_patches.csv")
    if os.path.exists(val):
        try:
            by_pair = {}
            with open(val, newline="") as _f:
                for row in csv.DictReader(_f):
                    a, b = row.get("neuron_a"), row.get("neuron_b")
                    if not _is_motor_partner(a, b):
                        continue
                    key = tuple(sorted((a, b)))
                    by_pair.setdefault(key, []).append(
                        (float(row["x"]), float(row["y"]), float(row["z"]),
                         float(row.get("area_um2", 0.0))))
            for (a, b), pts in by_pair.items():
                pts.sort(key=lambda p: -p[3])          # largest patch first
                kept = []
                for x, y, z, area in pts:
                    if all((x - kx) ** 2 + (y - ky) ** 2 + (z - kz) ** 2
                           >= _GJ_MARKER_SEP_NM ** 2 for kx, ky, kz in kept):
                        kept.append((x, y, z))
                        sites.append((x, y, z, "{} <-> {} (axonal)".format(a, b)))
            return sites
        except Exception:
            pass

    # Fallback: motor<->LPTC subset of the overlap metadata (still axonal).
    meta = os.path.join(results_dir, "overlap_em_meta.json")
    if os.path.exists(meta):
        try:
            with open(meta) as _f:
                for it in json.load(_f):
                    if not _is_motor_partner(it.get("source"), it.get("target")):
                        continue
                    x, y = it.get("x"), it.get("y")
                    z = it.get("z_base_nm", it.get("z"))
                    if x is not None and y is not None and z is not None:
                        sites.append((x, y, z, "{} <-> {} (overlap {})".format(
                            it.get("source", "?"), it.get("target", "?"), it.get("idx", ""))))
        except Exception:
            pass
    return sites

# Inhibitory pairs used for synapse color/label classification.
INH_PAIRS = frozenset([
    ('VS1_L', 'VS2_L'),
    ('VS1_R', 'VS2_R'),
    ('VS1_R', 'VS3_R'),
    ('VS2_L', 'VS3_L'),
    ('VS3_L', 'VS4_L'),
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
        #measureCanvas {
            position: absolute; left: 0; top: 0; width: 100%; height: 100%;
            pointer-events: none; z-index: 6;
        }
        #measureCanvas.active { pointer-events: auto; cursor: crosshair; }
        .measure-readout {
            color: #7fe3c8; font-size: 11px; font-family: 'Consolas', monospace;
            flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        }
        #btnMeasure.active { background: #2E7D32; border-color: #66BB6A; }
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
                <canvas id="measureCanvas"></canvas>
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
                <div class="control-row" style="gap:8px;flex-wrap:wrap;">
                    <button id="btnMeasure" title="Trace the membrane apposition to measure contact length/area (8 nm/px)">&#128207; Measure</button>
                    <button id="btnMeasureMode" title="Toggle Line (apposition length &times; 40 nm/slice) vs Area (polygon cross-section)" disabled>Mode: Line</button>
                    <button id="btnMeasureUndo" title="Undo last point" disabled>&#8630; Undo</button>
                    <button id="btnMeasureClear" title="Clear this slice's trace" disabled>Clear</button>
                    <button id="btnMeasureExport" title="Export all measurements as CSV" disabled>&#128190; CSV</button>
                    <span id="measureReadout" class="measure-readout">Measure: off</span>
                </div>
                <div class="control-row" style="justify-content: center; gap: 12px; flex-wrap: wrap;">
                    <span id="zNote" style="color: #888; font-size: 10px;">&#177;800nm depth range</span>
                    <button id="btnDeleteSlice" title="Remove this single slice (contact or overlap Z-slice)">&#128465; Delete Slice</button>
                    <button id="btnDeleteAll" title="Remove entire overlap pair (all slices)">&#128465; Delete All</button>
                    <button id="btnMarkGJ" title="Mark current location as putative gap junction">&#9889; Putative Gap-Junc</button>
                    <button id="btnRemoveGJ" title="Remove gap junction at current location" style="display:none;">&#10006; Remove GJ</button>
                    <button id="btnAllGJ" title="Show/hide all putative gap-junction sites (axonal motor&harr;LPTC contacts)">&#9889; All Putative GJ</button>
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
                <div class="modal-tab" data-tab="summary">Overlap Summary</div>
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
            <div class="modal-tab-content" id="tabSummary">
                <div id="summaryContainer" style="padding:12px;overflow:auto;"></div>
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
        // Automatic (geometric) contact area per pair, um2 — for comparing the
        // hand-traced measurement against the pipeline's own number.
        const autoAreas     = {AUTO_AREAS_JSON};
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
        const btnAllGJ      = document.getElementById('btnAllGJ');
        const gjContainer   = document.getElementById('gjContainer');
        const modalTitle    = document.getElementById('modalTitle');
        const connectivityContainer = document.getElementById('connectivityContainer');

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
        // electrophysiology and the anatomical contacts.  For each known pair
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

        // ── EM measurement tool (apposition length / cross-section area) ─────
        // EM images are 8 nm/px in-plane, 40 nm between Z-slices. LINE mode
        // traces the apposed membrane on each slice; contact area is then
        // Σ(length × 40 nm) across the slices spanned. AREA mode traces a closed
        // polygon → cross-sectional area. The scale is exact, so both are direct
        // pixel→nm conversions.
        const NM_PER_PX = 8.0;
        const NM_PER_SLICE = 40.0;
        const measureCanvas = document.getElementById('measureCanvas');
        const mctx = measureCanvas.getContext('2d');
        const btnMeasure = document.getElementById('btnMeasure');
        const btnMeasureMode = document.getElementById('btnMeasureMode');
        const btnMeasureUndo = document.getElementById('btnMeasureUndo');
        const btnMeasureClear = document.getElementById('btnMeasureClear');
        const btnMeasureExport = document.getElementById('btnMeasureExport');
        const measureReadout = document.getElementById('measureReadout');

        let measureOn = false;
        let measureMode = 'line';       // 'line' | 'area'
        // 'kind:idx:zoff' -> {mode, source, target, absZ, zoff, traces:[[[x,y],...], ...]}
        // A slice can hold SEVERAL traces: right-click finishes the current one
        // so the next click starts a new trace (several appositions per image).
        const measureStore = {};
        let hoverPt = null;

        function measureKey() { return currentKind + ':' + currentIdx + ':' + currentZ; }
        function pairKeyOf(s, t) { return [s, t].sort().join(' <-> '); }

        // traces of the current slice; the LAST one is the one being drawn
        function tracesAt(key) {
            const m = measureStore[key];
            return (m && m.traces) ? m.traces : [];
        }
        function activeTrace(key) {
            const t = tracesAt(key);
            return t.length ? t[t.length - 1] : null;
        }

        // object-fit:contain image rect (client px) → scale + letterbox offsets.
        function containedImgRect() {
            const r = emImage.getBoundingClientRect();
            const natW = emImage.naturalWidth || 512, natH = emImage.naturalHeight || 512;
            const scale = Math.min(r.width / natW, r.height / natH);
            return { scale, offX: r.left + (r.width - natW * scale) / 2,
                     offY: r.top + (r.height - natH * scale) / 2, natW, natH };
        }
        function clientToNative(cx, cy) {
            const c = containedImgRect();
            return [(cx - c.offX) / c.scale, (cy - c.offY) / c.scale];
        }
        function nativeToCanvas(nx, ny) {
            const c = containedImgRect();
            const canR = measureCanvas.getBoundingClientRect();
            return [(c.offX - canR.left) + nx * c.scale, (c.offY - canR.top) + ny * c.scale];
        }
        function polylineLenNm(pts) {
            let L = 0;
            for (let i = 1; i < pts.length; i++)
                L += Math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1]);
            return L * NM_PER_PX;
        }
        function polygonAreaUm2(pts) {
            if (pts.length < 3) return 0;
            let a = 0;
            for (let i = 0; i < pts.length; i++) {
                const p = pts[i], q = pts[(i + 1) % pts.length];
                a += p[0] * q[1] - q[0] * p[1];
            }
            return Math.abs(a) / 2 * NM_PER_PX * NM_PER_PX / 1e6;
        }
        // Apposition area over every traced slice of the CURRENT PAIR (line mode).
        // Each slice contributes sum(trace lengths) x 40 nm section thickness.
        function appositionAreaUm2() {
            let area = 0, n = 0, nTraces = 0;
            const pair = pairKeyOf(currentSource, currentTarget);
            for (const k in measureStore) {
                const m = measureStore[k];
                if (m.mode !== 'line') continue;
                if (pairKeyOf(m.source, m.target) !== pair) continue;
                let sliceLen = 0, used = 0;
                (m.traces || []).forEach(t => {
                    if (t.length >= 2) { sliceLen += polylineLenNm(t); used++; }
                });
                if (used) { area += sliceLen * NM_PER_SLICE / 1e6; n++; nTraces += used; }
            }
            return { area, n, nTraces };
        }
        function updateMeasureReadout() {
            if (!measureOn) { measureReadout.textContent = 'Measure: off'; return; }
            const traces = tracesAt(measureKey());
            if (measureMode === 'line') {
                let sliceLen = 0;
                traces.forEach(t => { if (t.length >= 2) sliceLen += polylineLenNm(t); });
                const acc = appositionAreaUm2();
                const auto = autoAreas[pairKeyOf(currentSource, currentTarget)];
                let txt = 'Line | slice ' + sliceLen.toFixed(0) + ' nm';
                if (traces.length > 1) txt += ' (' + traces.length + ' traces)';
                txt += ' | manual ' + acc.area.toFixed(4) + ' µm² (' + acc.n + ' sl)';
                if (auto) txt += ' | auto ' + auto.toFixed(3)
                    + ' µm² → ' + (100 * acc.area / auto).toFixed(1) + '%';
                measureReadout.textContent = txt;
            } else {
                let a = 0;
                traces.forEach(t => { a += polygonAreaUm2(t); });
                measureReadout.textContent = 'Area | ' + a.toFixed(4) + ' µm² ('
                    + traces.length + ' polygon' + (traces.length === 1 ? '' : 's') + ')';
            }
        }
        function redrawMeasure() {
            const canR = measureCanvas.getBoundingClientRect();
            if (measureCanvas.width !== Math.round(canR.width)) measureCanvas.width = Math.round(canR.width);
            if (measureCanvas.height !== Math.round(canR.height)) measureCanvas.height = Math.round(canR.height);
            mctx.clearRect(0, 0, measureCanvas.width, measureCanvas.height);
            if (!measureOn) return;
            const key = measureKey();
            const traces = tracesAt(key);
            const lastIdx = traces.length - 1;
            traces.forEach((pts, ti) => {
                const draw = pts.map(p => nativeToCanvas(p[0], p[1]));
                const preview = draw.slice();
                // rubber-band only on the trace currently being drawn
                if (ti === lastIdx && hoverPt && draw.length)
                    preview.push(nativeToCanvas(hoverPt[0], hoverPt[1]));
                // finished traces are dimmed so the active one stands out
                mctx.lineWidth = 2;
                mctx.strokeStyle = (ti === lastIdx) ? '#39FF14' : '#1f9c12';
                mctx.fillStyle = 'rgba(57,255,20,0.15)';
                if (preview.length) {
                    mctx.beginPath();
                    mctx.moveTo(preview[0][0], preview[0][1]);
                    for (let i = 1; i < preview.length; i++) mctx.lineTo(preview[i][0], preview[i][1]);
                    if (measureMode === 'area' && preview.length > 2) { mctx.closePath(); mctx.fill(); }
                    mctx.stroke();
                }
                mctx.fillStyle = (ti === lastIdx) ? '#FFD400' : '#b38f00';
                draw.forEach(p => {
                    mctx.beginPath(); mctx.arc(p[0], p[1], 3, 0, 2 * Math.PI); mctx.fill();
                });
            });
            updateMeasureReadout();
        }
        function setMeasureButtons() {
            [btnMeasureMode, btnMeasureUndo, btnMeasureClear].forEach(b => b.disabled = !measureOn);
            btnMeasureExport.disabled = Object.keys(measureStore).length === 0;
        }

        btnMeasure.addEventListener('click', function() {
            measureOn = !measureOn;
            btnMeasure.classList.toggle('active', measureOn);
            measureCanvas.classList.toggle('active', measureOn);
            setMeasureButtons(); hoverPt = null; redrawMeasure();
        });
        btnMeasureMode.addEventListener('click', function() {
            measureMode = (measureMode === 'line') ? 'area' : 'line';
            btnMeasureMode.textContent = 'Mode: ' + (measureMode === 'line' ? 'Line' : 'Area');
            redrawMeasure();
        });
        btnMeasureUndo.addEventListener('click', function() {
            const k = measureKey(), m = measureStore[k];
            if (!m || !m.traces.length) return;
            const t = m.traces[m.traces.length - 1];
            t.pop();
            // an emptied trace is dropped, exposing the previous one for editing
            if (!t.length) m.traces.pop();
            if (!m.traces.length) delete measureStore[k];
            setMeasureButtons(); redrawMeasure();
        });
        btnMeasureClear.addEventListener('click', function() {
            delete measureStore[measureKey()]; setMeasureButtons(); redrawMeasure();
        });
        measureCanvas.addEventListener('mousedown', function(e) {
            if (!measureOn || currentIdx === null || e.button !== 0) return;
            const [nx, ny] = clientToNative(e.clientX, e.clientY);
            const c = containedImgRect();
            if (nx < 0 || ny < 0 || nx > c.natW || ny > c.natH) return;  // outside image
            const k = measureKey();
            if (!measureStore[k]) measureStore[k] = {
                mode: measureMode, source: currentSource, target: currentTarget,
                absZ: Math.round(curItemZnm + currentZ * 40), zoff: currentZ, traces: [[]]
            };
            const m = measureStore[k];
            m.mode = measureMode;
            if (!m.traces.length) m.traces.push([]);
            m.traces[m.traces.length - 1].push([nx, ny]);
            setMeasureButtons(); redrawMeasure();
        });

        // Right-click finishes the current trace, so the next left-click starts a
        // NEW one — for images with several separate appositions.
        measureCanvas.addEventListener('contextmenu', function(e) {
            if (!measureOn) return;
            e.preventDefault();
            const m = measureStore[measureKey()];
            if (!m || !m.traces.length) return;
            const t = m.traces[m.traces.length - 1];
            if (t.length >= 2) {
                m.traces.push([]);           // start a fresh trace
                hoverPt = null;
                infoText.textContent = '✓ Trace ' + (m.traces.length - 1)
                    + ' finished (' + polylineLenNm(t).toFixed(0)
                    + ' nm) — click to start the next one';
            } else if (t.length) {
                t.length = 0;                // discard a stray single point
            }
            redrawMeasure();
        });
        measureCanvas.addEventListener('mousemove', function(e) {
            if (!measureOn) return;
            hoverPt = clientToNative(e.clientX, e.clientY); redrawMeasure();
        });
        measureCanvas.addEventListener('mouseleave', function() { hoverPt = null; redrawMeasure(); });
        window.addEventListener('resize', redrawMeasure);
        // Redraw whenever a new EM slice/image loads (persists across loadImage's
        // own onload reassignment).
        emImage.addEventListener('load', redrawMeasure);

        btnMeasureExport.addEventListener('click', function() {
            const rows = [['kind','idx','source','target','pair','z_offset','abs_z_nm','mode',
                           'trace','n_points','trace_length_nm','trace_area_um2']];
            const pairAcc = {};
            for (const k in measureStore) {
                const m = measureStore[k];
                const parts = k.split(':'), kind = parts[0], idx = parts[1];
                const pk = pairKeyOf(m.source, m.target);
                let sliceLen = 0;
                (m.traces || []).forEach((t, ti) => {
                    if (!t.length) return;
                    const len = (m.mode === 'line') ? polylineLenNm(t) : 0;
                    const ar  = (m.mode === 'area') ? polygonAreaUm2(t) : 0;
                    rows.push([kind, idx, m.source || '', m.target || '', pk, m.zoff, m.absZ,
                               m.mode, ti + 1, t.length, len.toFixed(1), ar.toFixed(6)]);
                    if (m.mode === 'line' && t.length >= 2) sliceLen += len;
                });
                if (m.mode === 'line' && sliceLen > 0) {
                    pairAcc[pk] = pairAcc[pk] || {area: 0, n: 0};
                    pairAcc[pk].area += sliceLen * NM_PER_SLICE / 1e6;
                    pairAcc[pk].n++;
                }
            }
            rows.push([]);
            rows.push(['# Manual vs automatic contact area per pair.']);
            rows.push(['# manual = sum over traced slices of (traced length x 40 nm section).']);
            rows.push(['# automatic = geometric contact area from the pipeline (post-proofreading).']);
            rows.push(['# NOTE: manual covers only the slices actually traced, so it is a lower']);
            rows.push(['#       bound unless every slice of the contact was traced.']);
            rows.push(['pair','n_slices_traced','manual_area_um2','automatic_area_um2','manual_pct_of_auto']);
            for (const pk in pairAcc) {
                const a = pairAcc[pk];
                const auto = autoAreas[pk];
                rows.push([pk, a.n, a.area.toFixed(6),
                           (auto !== undefined ? auto.toFixed(6) : ''),
                           (auto ? (100 * a.area / auto).toFixed(1) : '')]);
            }
            const csv = rows.map(r => r.join(',')).join('\n');
            const blob = new Blob([csv], {type: 'text/csv'});
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url; link.download = 'em_measurements.csv'; link.click();
            URL.revokeObjectURL(url);
        });
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

        // Toggle the full set of pre-selected putative-GJ markers. Hidden on
        // load; the user reveals them on demand with this button.
        let _allGJVisible = false;
        const _preselGJIdx = traceInfo['_preselected_gj'];
        btnAllGJ.addEventListener('click', function() {
            if (_preselGJIdx === undefined) return;
            _allGJVisible = !_allGJVisible;
            safeRestyle(plotDiv, { visible: [_allGJVisible] }, [_preselGJIdx]);
            btnAllGJ.style.background = _allGJVisible ? '#2E7D32' : '';
            btnAllGJ.style.borderColor = _allGJVisible ? '#66BB6A' : '';
            infoText.textContent = _allGJVisible
                ? '⚡ Showing all putative gap-junction sites (axonal motor↔LPTC contacts)'
                : 'Putative gap-junction markers hidden';
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

        // ── Overlap Summary matrix (publication panel j) ─────────────
        // Binary MOT/MOS x HS/VS overlap. Rows = MOS/MOT x right/left;
        // columns = cell TYPE (L/R collapsed). A cell is filled in the
        // motor color when the row's motor neuron overlaps the same-
        // hemisphere LPTC of that column type. Mirrors reduced_matrix.py.
        function renderSummaryMatrix() {
            const C_MOT = '#5E3C99', C_MOS = '#4D9221';
            const C_HS = '#C51B7D', C_VS14 = '#D14900', C_VS58 = '#007F5F';
            const MIN_AREA = 0.5;
            const rows = [
                { motor: 'MOS_R', color: C_MOS, label: 'MOS', hemi: 'right' },
                { motor: 'MOT_R', color: C_MOT, label: 'MOT', hemi: 'right' },
                { motor: 'MOS_L', color: C_MOS, label: 'MOS', hemi: 'left' },
                { motor: 'MOT_L', color: C_MOT, label: 'MOT', hemi: 'left' },
            ];
            const cols = [
                { t: 'HSN', g: 'HS', c: C_HS }, { t: 'HSS', g: 'HS', c: C_HS }, { t: 'HSE', g: 'HS', c: C_HS },
                { t: 'VS1', g: 'VS1-4', c: C_VS14 }, { t: 'VS2', g: 'VS1-4', c: C_VS14 },
                { t: 'VS3', g: 'VS1-4', c: C_VS14 }, { t: 'VS4', g: 'VS1-4', c: C_VS14 },
                { t: 'VS5', g: 'VS5-8', c: C_VS58 }, { t: 'VS6', g: 'VS5-8', c: C_VS58 },
                { t: 'VS7', g: 'VS5-8', c: C_VS58 }, { t: 'VS8', g: 'VS5-8', c: C_VS58 },
            ];
            const area = {};
            overlapTable.forEach(r => {
                const k = r.source + '|' + r.target;
                if (area[k] === undefined || r.area > area[k]) area[k] = r.area;
            });
            function ov(a, b) {
                return Math.max(area[a + '|' + b] || 0, area[b + '|' + a] || 0) >= MIN_AREA;
            }
            const cs = 34, x0 = 72, y0 = 58;
            const W = x0 + cols.length * cs + 90;
            const H = y0 + rows.length * cs + 70;
            let svg = '<svg width="' + W + '" height="' + H + '" xmlns="http://www.w3.org/2000/svg" style="background:#fff;font-family:sans-serif;">';
            // column-group color bars + group labels
            let j = 0;
            while (j < cols.length) {
                const g = cols[j].g; let k = j;
                while (k < cols.length && cols[k].g === g) k++;
                const bx = x0 + j * cs, bw = (k - j) * cs;
                svg += '<rect x="' + bx + '" y="' + (y0 - 20) + '" width="' + bw + '" height="13" fill="' + cols[j].c + '"/>';
                svg += '<text x="' + (bx + bw / 2) + '" y="' + (y0 - 25) + '" text-anchor="middle" font-size="12" font-weight="bold" fill="' + cols[j].c + '">' + g + '</text>';
                j = k;
            }
            // cells + row labels
            rows.forEach((row, i) => {
                cols.forEach((col, jj) => {
                    const x = x0 + jj * cs, y = y0 + i * cs;
                    const filled = ov(row.motor, col.t + (row.hemi === 'right' ? '_R' : '_L'));
                    svg += '<rect x="' + x + '" y="' + y + '" width="' + cs + '" height="' + cs + '" fill="' + (filled ? row.color : '#ffffff') + '" stroke="#444" stroke-width="0.8"/>';
                });
                svg += '<text x="' + (x0 - 8) + '" y="' + (y0 + i * cs + cs / 2 + 4) + '" text-anchor="end" font-size="13" font-weight="bold" fill="' + row.color + '">' + row.label + '</text>';
            });
            // column labels (rotated, tinted by group)
            cols.forEach((col, jj) => {
                const cx = x0 + jj * cs + cs / 2, cy = y0 + rows.length * cs + 6;
                svg += '<text x="' + cx + '" y="' + cy + '" font-size="11" fill="' + col.c + '" transform="rotate(90 ' + cx + ' ' + cy + ')">' + col.t + '</text>';
            });
            // right-side hemisphere brackets
            const bxr = x0 + cols.length * cs + 10;
            [['right', 0], ['left', 2]].forEach(pair => {
                const top = y0 + pair[1] * cs + 3, bot = y0 + (pair[1] + 2) * cs - 3;
                svg += '<path d="M' + bxr + ' ' + top + ' h8 V' + bot + ' h-8" fill="none" stroke="#222" stroke-width="1.4"/>';
                const mid = (top + bot) / 2;
                svg += '<text x="' + (bxr + 24) + '" y="' + (mid + 4) + '" text-anchor="middle" font-size="12" fill="#111" transform="rotate(90 ' + (bxr + 24) + ' ' + mid + ')">' + pair[0] + '</text>';
            });
            svg += '</svg>';
            const caption = '<div style="margin-top:10px;color:#333;font-size:11px;max-width:560px;line-height:1.4;">'
                + '<b>Overlap summary (publication panel).</b> Filled = the motor neuron (row) overlaps the '
                + 'same-hemisphere LPTC of that type (column) with contact area &ge; ' + MIN_AREA + ' &micro;m&sup2;. '
                + 'MOS rows in MOS color, MOT rows in MOT color; VS5-8 is the control group.</div>';
            document.getElementById('summaryContainer').innerHTML = svg + caption;
        }

        // ── Matrix modal ────────────────────────────────────────────
        // LAZY TAB RENDERING: only render a tab's content when it is first
        // clicked (or when data changes).  The three static-data tabs
        // (overlaps, gapjunctions, connectivity) are rendered once on first
        // open; tabs re-render on first visit because
        // they contain live simulation state.
        const _tabRendered = { overlaps: false, gapjunctions: false, connectivity: false, summary: false };

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
            summary:      { el: 'tabSummary',         title: 'Overlap Summary (MOT/MOS × HS/VS)' },
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
                    if (target === 'summary' && !_tabRendered.summary) {
                        renderSummaryMatrix();
                        _tabRendered.summary = true;
                    }
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
            // e.g. "VS1_L" → "VS1", "HSN_R" → "HSN"
            return name.replace(/_[LR]$/, '');
        }

        function renderHeatmap() {
            // ─────────────────────────────────────────────────────────────────
            // OVERLAP AREA HEATMAP  (4 panels)
            // Panel 1 — Full matrix: every neuron pair, raw area (µm²).
            // Panel 2 — L/R pair mean: base names (e.g. VS1), L and R averaged.
            // Panel 3 — Group mean: functional groups (VS, HS, MOT, MOS).
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
            // Group neurons by functional group (MOT, MOS, VS1..VS4, HSN/HSE/HSS)
            // Then average the pair means
            function _groupName(baseName) {
                // VS1→VS, VS2→VS, etc. ; HSN→HS, HSE→HS, HSS→HS
                if (baseName.match(/^VS\d/)) return 'VS';
                if (baseName.match(/^HS[NES]/)) return 'HS';
                return baseName;  // MOT, MOS
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

        // ── Cell names & index mapping (connectivity matrix) ──
        const CELL_NAMES = [
            'MOT_L','MOT_R','MOS_L','MOS_R',
            'VS1_L','VS1_R','VS2_L','VS2_R',
            'VS3_L','VS3_R','VS4_L','VS4_R',
            'HSN_L','HSN_R','HSE_L','HSE_R',
            'HSS_L','HSS_R'
        ];
        const N_CELLS = CELL_NAMES.length;
        const CI = {}; CELL_NAMES.forEach((n,i) => { CI[n] = i; });
        const SPIKING = new Set(['MOT_L','MOT_R','MOS_L','MOS_R']);

        // ── RAW_COUNTS: rows=pre, cols=post  [from synapses.csv] ──
        const RAW_COUNTS = [
         // MOT_L MOT_R MOS_L MOS_R VS1_L VS1_R VS2_L VS2_R VS3_L VS3_R VS4_L VS4_R HSN_L HSN_R HSE_L HSE_R HSS_L HSS_R
            [   0,    0,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOT_L
            [   0,    0,    0,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOT_R
            [   3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOS_L
            [   0,    6,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOS_R
            [   0,    0,    0,    0,    0,    0,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS1_L
            [   0,    0,    0,    0,    0,    0,    0,    4,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0],  // VS1_R
            [   0,    0,    6,    0,    1,    0,    0,    0,    4,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS2_L
            [   0,    0,    0,    0,    0,    6,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS2_R
            [   0,    0,    6,    0,    0,    0,   12,    0,    0,    0,    2,    0,    0,    0,    0,    0,    0,    0],  // VS3_L
            [   0,    0,    0,    5,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS3_R
            [   0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS4_L
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    0,    0],  // VS4_R
            [   2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0],  // HSN_L
            [   0,    9,    0,    4,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    7,    0,    3],  // HSN_R
            [   4,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    5,    0,    0,    0,    0,    0],  // HSE_L
            [   0,    6,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    2,    0,    0,    0,    0],  // HSE_R
            [   4,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    0,    0,    0,    0,    0],  // HSS_L
            [   0,    3,    0,    2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // HSS_R
        ];

        // ── SYN_ESYN: per-connection reversal potential (mV) ──
        // FlyWire NT predictions: ACh/Glut → excitatory (0 mV), GABA → inhibitory (-80 mV)
        // VS chain fwd (VS1→VS2, VS2→VS3, VS3→VS4) = GABA(-80)
        // VS chain back (VS3→VS2, VS2→VS1, VS4→VS2) = ACh(0)
        // HS→MN, VS→MOS, HS chain, MN↔MN = excitatory(0)
        const E_EXC = 0, E_INH = -80;
        const SYN_ESYN = [
         // MOT_L MOT_R MOS_L MOS_R VS1_L VS1_R VS2_L VS2_R VS3_L VS3_R VS4_L VS4_R HSN_L HSN_R HSE_L HSE_R HSS_L HSS_R
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOT_L
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOT_R
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOS_L
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOS_R
            [   0,    0,    0,    0,    0,    0,  -80,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS1_L  fwd→VS2
            [   0,    0,    0,    0,    0,    0,    0,  -80,    0,  -80,    0,    0,    0,    0,    0,    0,    0,    0],  // VS1_R  fwd→VS2,VS3
            [   0,    0,    0,    0,    0,    0,    0,    0,  -80,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS2_L  fwd→VS3
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS2_R
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,  -80,    0,    0,    0,    0,    0,    0,    0],  // VS3_L  fwd→VS4
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS3_R
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS4_L
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS4_R
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // HSN_L
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // HSN_R
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // HSE_L
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // HSE_R
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // HSS_L
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // HSS_R
        ];

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

    # BOTH partners must be in the active configuration. With an OR here a
    # synapse onto a cell outside the config leaks that cell's name into the
    # viewer (partner labels, connectivity matrix), so the viewer would show
    # neurons the user never asked for — e.g. left-over rows in a synapses.csv
    # fetched for a wider neuron set.
    n_all = len(df)
    df = df[df['source'].isin(SYNAPSE_NEURONS) & df['target'].isin(SYNAPSE_NEURONS)]
    dropped = n_all - len(df)
    print(f"[synapses] Loaded {len(df)} synapses "
          f"(groups: {', '.join(sorted(_synapse_groups))})"
          + (f"; dropped {dropped} involving cells outside the config" if dropped else ""))
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

    # ── 7b. Pre-selected putative gap-junction sites (always visible) ──
    # The confirmed MOT_R<->HSN_R junction (larger marker) plus curated example
    # sites from gj_figures/gj_sites.json, marked on load with the green
    # putative-GJ marker style (the interactive GJ trace is checkbox-gated).
    trace_info['_preselected_gj'] = len(traces)
    _sites = _load_preselected_gj(RESULTS_DIR)
    _pg_x = [s[0] for s in _sites]
    _pg_y = [s[1] for s in _sites]
    _pg_z = [s[2] for s in _sites]
    _pg_txt = ['{}<br>{} nm'.format(s[3], (int(s[0]), int(s[1]), int(s[2]))) for s in _sites]
    _pg_size = [13] + [6] * (len(_sites) - 1)   # confirmed (first) is larger
    traces.append(go.Scatter3d(
        x=_pg_x, y=_pg_y, z=_pg_z,
        mode='markers',
        name='Putative GJ (large = confirmed)',
        visible=False,   # hidden on load; revealed via the "All Putative GJ" button
        marker=dict(size=_pg_size, color='#39FF14', symbol='circle',
                    line=dict(color='#0a3d0a', width=1), opacity=1.0),
        hovertext=_pg_txt,
        hovertemplate='%{hovertext}<extra></extra>',
        showlegend=True,
    ))
    print(f"  [preselected_gj] {len(_sites)} pre-selected GJ site(s) marked (confirmed + curated)")

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
        # Voxel coordinates at 4x4x40 nm, matching generate_em_stacks._coord_tag
        return f"vx{int(round(x/4.0))}_vy{int(round(y/4.0))}_vz{int(round(z/40.0))}"

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

    # Defensive: every overlap that has z-stack files must also have a center
    # image mapped — even when overlap_em_meta.json is absent/incomplete or the
    # files use coordinate-tagged names. Without this, overlaps show no EM in the
    # viewer (while synapses, which keep plain names, still work).
    for idx in list(zstack_map.get('overlap', {}).keys()):
        if int(idx) not in snapshot_map['overlap']:
            fn = _find_segmented('overlap', int(idx))
            if fn:
                snapshot_map['overlap'][int(idx)] = f"em_snaps/{fn}"

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

    # Automatic per-pair contact area (um2) from the EM cluster metadata, so the
    # viewer can show hand-traced vs automatic side by side. Reads the *current*
    # metadata, i.e. it already reflects any applied proofreading.
    _auto_areas = {}
    _meta_f = os.path.join(RESULTS_DIR, 'overlap_em_meta.json')
    if os.path.exists(_meta_f):
        try:
            for _m in json.load(open(_meta_f)):
                _k = ' <-> '.join(sorted((str(_m.get('source')), str(_m.get('target')))))
                _auto_areas[_k] = _auto_areas.get(_k, 0.0) + float(_m.get('total_area_um2', 0.0))
        except Exception:
            _auto_areas = {}
    html = html.replace('{AUTO_AREAS_JSON}', json.dumps(_auto_areas))

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


