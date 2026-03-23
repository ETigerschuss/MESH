"""
Skeleton EM Viewer - Config-Driven Neuron Viewer
=================================================
Self-contained EM viewer reading neuron config from neurons.json.

Features:
- Left sidebar: Neuron controls (mesh / all contacts / curr contacts / overlaps / synapses)
- Center top: 3D point-cloud viewer with contact/synapse/overlap highlighting
- Center bottom: Overlap summary table (live-updating)
- Right: EM snapshot viewer with Z-stack navigation + delete contact button
- Delete button removes contact from viewer + marks it; table updates in real-time

Usage:
    python skeleton_em_viewer.py

Output:
    comprehensive_overlap_results_<date>/skeleton_em_viewer.html
"""

import os
import json
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh

# ── Config from neurons.json ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, 'neurons.json'), 'r') as f:
    _cfg = json.load(f)

NEURON_CFG     = _cfg['neurons']
NEURON_IDS     = {info['id']: name for name, info in NEURON_CFG.items()}
NEURON_COLORS  = {name: info['color_hex'] for name, info in NEURON_CFG.items()}
VIEWER_NEURONS = _cfg.get('viewer_neurons', sorted(NEURON_CFG.keys()))

_synapse_groups = set(_cfg.get('synapse_groups', []))
SYNAPSE_NEURONS = [n for n, info in NEURON_CFG.items() if info['group'] in _synapse_groups]


def _default_results_dir():
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
            background: #2a2a2a; border: 1px solid #555;
            border-radius: 8px; max-width: 90vw; max-height: 85vh;
            overflow: auto; padding: 16px; position: relative;
        }
        .modal-content h3 { color: #FFD400; margin: 0 0 12px 0; }
        .modal-close {
            position: absolute; top: 8px; right: 12px;
            background: none; border: none; color: #aaa;
            font-size: 22px; cursor: pointer;
        }
        .modal-close:hover { color: #fff; }

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
        .modal-tabs { display: flex; gap: 0; margin-bottom: 12px; border-bottom: 2px solid #555; }
        .modal-tab {
            padding: 6px 16px; cursor: pointer; background: #333; color: #aaa;
            border: 1px solid #555; border-bottom: none; border-radius: 6px 6px 0 0;
            font-size: 12px; font-weight: bold;
        }
        .modal-tab:hover { background: #444; color: #fff; }
        .modal-tab.active { background: #2a2a2a; color: #FFD400; border-bottom: 2px solid #2a2a2a; margin-bottom: -2px; }
        .modal-tab-content { display: none; }
        .modal-tab-content.active { display: block; }
        .gj-table { border-collapse: collapse; font-size: 11px; width: 100%; }
        .gj-table th { background: #333; color: #FFD400; padding: 4px 8px; border: 1px solid #444; text-align: left; font-size: 10px; }
        .gj-table td { padding: 4px 8px; border: 1px solid #333; font-size: 10px; color: #ccc; }
        .gj-table tr:hover { background: #333; }
        .gj-table .gj-delete { cursor: pointer; color: #B00; }
        .gj-table .gj-delete:hover { color: #F44; }

        /* ── Heatmap matrix ─────────────────────── */
        .heatmap-container {
            max-height: 350px; min-height: 120px;
            overflow: auto; background: #1a1a1a;
            border-top: 2px solid #444;
            padding: 8px;
        }
        .heatmap-container table {
            border-collapse: collapse;
            font-size: 11px;
        }
        .heatmap-container th {
            background: #333; color: #FFD400;
            padding: 3px 6px; text-align: center;
            border: 1px solid #444;
            position: sticky; top: 0; z-index: 2;
            font-size: 10px;
        }
        .heatmap-container th.row-header {
            position: sticky; left: 0; z-index: 3;
            background: #333;
        }
        .heatmap-container th.corner {
            position: sticky; left: 0; top: 0; z-index: 4;
            background: #333;
        }
        .heatmap-container td {
            padding: 2px 4px; text-align: center;
            border: 1px solid #333; cursor: pointer;
            font-size: 10px; font-family: monospace;
            min-width: 46px;
        }
        .heatmap-container td:hover { outline: 2px solid #FFD400; }
        .heatmap-container td.diagonal { background: #1a1a1a; cursor: default; }
        .heatmap-container td.no-data { color: #555; }
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
                <div class="control-row" style="justify-content: center; gap: 12px; flex-wrap: wrap;">
                    <span id="zNote" style="color: #888; font-size: 10px;">&#177;800nm depth range</span>
                    <button id="btnDeleteSlice" title="Remove this single slice (contact or overlap Z-slice)">&#128465; Delete Slice</button>
                    <button id="btnDeleteAll" title="Remove entire overlap pair (all slices)">&#128465; Delete All</button>
                    <button id="btnMarkGJ" title="Mark current location as putative gap junction">&#9889; Mark Gap Junction</button>
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
                <div class="modal-tab" data-tab="circuit">Circuit Model</div>
            </div>
            <div class="modal-tab-content active" id="tabOverlaps">
                <div class="heatmap-container" id="heatmapContainer"></div>
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
        </div>
    </div>

    <script>
        // ── Embedded data ───────────────────────────────────────────
        const snapshotMap   = {SNAPSHOT_JSON};
        const contactClusterMap = {CLUSTER_MAP_JSON};
        const neuronNames   = {NEURON_NAMES_JSON};
        const traceInfo     = {TRACE_INFO_JSON};
        let   contactList   = {CONTACT_LIST_JSON};
        const synapseList   = {SYNAPSE_LIST_JSON};
        const overlapList   = {OVERLAP_LIST_JSON};
        let   overlapTable  = {OVERLAP_TABLE_JSON};
        const overlapPairs  = {OVERLAP_PAIRS_JSON};
        const overlapPairFaces = {OVERLAP_PAIR_FACES_JSON};
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
        // Plotly.restyle on gl3d traces can hang the browser (Promise never
        // resolves, or triggers a full scene recompute that takes minutes).
        // Instead we mutate plotDiv.data[idx] in-place and batch a single
        // Plotly.redraw via requestAnimationFrame.
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
        function safeRestyle(div, update, indices) {
            // Mutate trace data in-place, then schedule a single redraw
            try {
                const idxArr = Array.isArray(indices) ? indices : [indices];
                idxArr.forEach((trIdx, i) => {
                    const trace = plotDiv.data[trIdx];
                    if (!trace) return;
                    for (const key in update) {
                        const val = update[key];
                        if (Array.isArray(val) && val.length === idxArr.length) {
                            trace[key] = val[i];
                        } else if (Array.isArray(val) && val.length === 1 && idxArr.length === 1) {
                            trace[key] = val[0];
                        } else {
                            trace[key] = val;
                        }
                    }
                });
                scheduleRedraw();
            } catch(e) { console.warn('safeRestyle error:', e.message); }
        }
        function safeRelayout(div, update) {
            try { Plotly.relayout(div, update); }
            catch(e) { console.warn('relayout error:', e.message); }
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
                    // Also include from overlap face checkboxes
                    ['alloverlapfaces_','curroverlapfaces_'].forEach(pre => {
                        const cb2 = document.getElementById(pre + neuron);
                        if (cb2 && cb2.checked) {
                            itemList.filter(it => (it.source === neuron || it.target === neuron) && !it._eliminated)
                                    .forEach(it => { if (it.idx >= 0) seen.add(it.idx); });
                        }
                    });
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
            safeRestyle(plotDiv, {
                x: [filtered.map(d => d.x)],
                y: [filtered.map(d => d.y)],
                z: [filtered.map(d => d.z)],
                visible: [show],
                customdata: [filtered.map(d => [d.x, d.y, d.z,
                    kind === 'contacts' ? 'contact' : 'synapse',
                    d.source, d.target, d.idx, d.patch_num || 0])]
            }, [traceIdx]);
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

        // ── Overlap FACE trace toggle (Mesh3d surfaces) ──────────────
        function rebuildAllOverlapFacesTrace(neuron) {
            const cb = document.getElementById('alloverlapfaces_' + neuron);
            const traceIdx = traceInfo[neuron + '_alloverlapfaces'];
            if (traceIdx === undefined) return;
            if (!cb || !cb.checked) {
                safeRestyle(plotDiv, { visible: [false] }, [traceIdx]);
                return;
            }
            // Rebuild from pair faces data, filtering eliminated
            const pairs = overlapPairFaces[neuron] || [];
            const xs = [], ys = [], zs = [];
            const ii = [], jj = [], kk = [];
            let vertOffset = 0;
            pairs.forEach(p => {
                const pairOv = overlapList.find(o =>
                    (o.source === p.source && o.target === p.target) ||
                    (o.source === p.target && o.target === p.source));
                if (pairOv && pairOv._eliminated) return;
                for (let v = 0; v < p.x.length; v++) {
                    xs.push(p.x[v]); ys.push(p.y[v]); zs.push(p.z[v]);
                }
                for (let f = 0; f < p.i.length; f++) {
                    ii.push(p.i[f] + vertOffset);
                    jj.push(p.j[f] + vertOffset);
                    kk.push(p.k[f] + vertOffset);
                }
                vertOffset += p.x.length;
            });
            safeRestyle(plotDiv, {
                x: [xs], y: [ys], z: [zs],
                i: [ii], j: [jj], k: [kk], visible: [xs.length > 0]
            }, [traceIdx]);
        }
        function rebuildCurrOverlapFacesTrace(neuron) {
            const cb = document.getElementById('curroverlapfaces_' + neuron);
            const traceIdx = traceInfo[neuron + '_curroverlapfaces'];
            if (traceIdx === undefined) return;
            if (!cb || !cb.checked) {
                safeRestyle(plotDiv, { visible: [false] }, [traceIdx]);
                return;
            }
            const mv = getVisibleNeurons();
            const pairs = overlapPairFaces[neuron] || [];
            const xs = [], ys = [], zs = [];
            const ii = [], jj = [], kk = [];
            let vertOffset = 0;
            pairs.forEach(p => {
                // Skip eliminated overlap pairs
                const pairOv = overlapList.find(o =>
                    (o.source === p.source && o.target === p.target) ||
                    (o.source === p.target && o.target === p.source));
                if (pairOv && pairOv._eliminated) return;
                if (mv.includes(p.other)) {
                    for (let v = 0; v < p.x.length; v++) {
                        xs.push(p.x[v]); ys.push(p.y[v]); zs.push(p.z[v]);
                    }
                    for (let f = 0; f < p.i.length; f++) {
                        ii.push(p.i[f] + vertOffset);
                        jj.push(p.j[f] + vertOffset);
                        kk.push(p.k[f] + vertOffset);
                    }
                    vertOffset += p.x.length;
                }
            });
            safeRestyle(plotDiv, {
                x: [xs], y: [ys], z: [zs],
                i: [ii], j: [jj], k: [kk], visible: [xs.length > 0]
            }, [traceIdx]);
        }
        function recalcAllCurrOverlapFaces() {
            neuronNames.forEach(n => {
                const cb = document.getElementById('curroverlapfaces_' + n);
                if (cb && cb.checked) rebuildCurrOverlapFacesTrace(n);
            });
        }

        // ── Checkbox handlers ───────────────────────────────────────
        neuronNames.forEach(neuron => {
            const meshCb = document.getElementById('mesh_' + neuron);
            if (meshCb) meshCb.addEventListener('change', function() {
                const idx = traceInfo[neuron + '_mesh'];
                if (idx !== undefined)
                    safeRestyle(plotDiv, {visible: [this.checked]}, [idx]);
                recalcAllCurrentTraces();
                recalcAllCurrOverlaps();
                recalcAllCurrOverlapFaces();
            });
            ['contacts', 'synapses'].forEach(kind => {
                const allCb = document.getElementById('all' + kind + '_' + neuron);
                if (allCb) allCb.addEventListener('change', () => rebuildTraceData(neuron, kind));
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
            currentZ = 0;
            currentSource = source;
            currentTarget = target;
            zSlider.value = 0;
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

            // Dynamic Z-slider range
            if (kind === 'overlap') {
                const ov = overlapList.find(o => o.idx === idx);
                const zLo = ov ? ov.z_lo : -20;
                const zHi = ov ? ov.z_hi : 20;
                const vz = (ov && ov.valid_z && ov.valid_z.length) ? ov.valid_z : null;
                zSlider.min = zLo;
                zSlider.max = zHi;
                zSlider.dataset.validZ = vz ? JSON.stringify(vz) : '';
                const nSlices = vz ? vz.length : (zHi - zLo + 1);
                zNote.textContent = nSlices + ' EM slices  (Z: ' + zLo + ' to ' + zHi + ')';
            } else {
                zSlider.min = -20;
                zSlider.max = 20;
                zSlider.dataset.validZ = '';
            }

            // Set base Z and initial diamond position
            curItemZnm = z;  // default: use the passed-in Z
            if (kind === 'overlap') {
                const ov = overlapList.find(o => o.idx === idx);
                if (ov) {
                    // ov.z = z_base_nm (absolute Z of the first EM slice)
                    curItemZnm = ov.z;
                    // Start diamond at EM-slice-0 center coords
                    const sc0 = ov.slice_coords && ov.slice_coords['0'];
                    if (sc0) { x = sc0[0]; y = sc0[1]; }
                }
            }
            curItemX = x; curItemY = y;
            update3DIndicator(x, y, curItemZnm);

            loadImage(kind, idx, 0);
        }

        // ── Image loading ───────────────────────────────────────────
        function loadImage(kind, idx, zOffset) {
            const imgData = snapshotMap[kind] && snapshotMap[kind][idx];
            if (!imgData) {
                emImage.style.display = 'none';
                emPlaceholder.style.display = 'block';
                emPlaceholder.textContent = 'No snapshot for ' + kind + ' ' + idx;
                return;
            }
            currentZ = zOffset;  // sync so stepValidZ always sees latest offset
            if (zOffset === 0) {
                emImage.src = imgData;
                emImage.style.display = 'block';
                emPlaceholder.style.display = 'none';
                updateZValue(0);
            } else {
                loadZStackImage(kind, idx, zOffset);
            }
            // For overlaps: update X,Y from per-slice EM center coords
            if (kind === 'overlap') {
                const ov = overlapList.find(o => o.idx === idx);
                if (ov && ov.slice_coords) {
                    const sc = ov.slice_coords[String(zOffset)];
                    if (sc) { curItemX = sc[0]; curItemY = sc[1]; }
                }
            }
            // Update 3D diamond to match current EM slice position
            const newZ = curItemZnm + zOffset * 40;
            update3DIndicator(curItemX, curItemY, newZ);
            // Update location display with diamond coords
            emLocation.textContent = currentSource + ' \u2192 ' + currentTarget
                + ' at (' + Math.round(curItemX) + ', ' + Math.round(curItemY) + ', ' + Math.round(curItemZ) + ')';
        }

        function loadZStackImage(kind, idx, zOffset) {
            const sign = zOffset >= 0 ? '+' : '-';
            const zStr = 'z' + sign + String(Math.abs(zOffset)).padStart(3, '0');
            emImage.onerror = function() {
                const center = snapshotMap[kind] && snapshotMap[kind][idx];
                if (center && zOffset !== 0) emImage.src = center;
            };
            emImage.onload = function() {
                emImage.style.display = 'block';
                emPlaceholder.style.display = 'none';
                updateZValue(zOffset);
            };
            let filePrefix = kind + '_' + idx;
            if (kind === 'contact' && contactClusterMap[idx] !== undefined) {
                filePrefix = 'cluster_' + contactClusterMap[idx];
            }
            emImage.src = 'em_snaps/' + filePrefix + '_' + zStr + '.png';
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
                return cur;
            } else {
                for (let i = 0; i < vz.length; i++) {
                    if (vz[i] > cur) return vz[i];
                }
                return cur;
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
            }
            itemInfo.textContent = currentKind.charAt(0).toUpperCase()
                + currentKind.slice(1) + ' '
                + (newPos + 1) + '/' + currentList.length + ' (idx: ' + newIdx + ')';
            deletedBanner.style.display = 'none';
            btnDeleteSlice.style.display = (currentKind === 'contact' || currentKind === 'overlap')
                ? 'inline-block' : 'none';
            btnDeleteAll.style.display = (currentKind === 'overlap')
                ? 'inline-block' : 'none';

            if (currentKind === 'overlap') {
                const ov = overlapList.find(o => o.idx === newIdx);
                const zLo = ov ? ov.z_lo : -20;
                const zHi = ov ? ov.z_hi : 20;
                const vz = (ov && ov.valid_z && ov.valid_z.length) ? ov.valid_z : null;
                zSlider.min = zLo; zSlider.max = zHi;
                zSlider.dataset.validZ = vz ? JSON.stringify(vz) : '';
            } else {
                zSlider.min = -20; zSlider.max = 20;
                zSlider.dataset.validZ = '';
            }
            currentZ = 0; zSlider.value = 0;
            loadImage(currentKind, newIdx, 0);
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
            // Find all sub-clusters for this undirected pair
            const subs = overlapList.filter(
                o => (o.source === source && o.target === target)
                  || (o.source === target && o.target === source));
            let totalOrigArea = 0;
            let totalRemainArea = 0;
            let allEliminated = true;
            subs.forEach(sub => {
                const origN = sub.orig_n_slices || 1;
                const curN = (sub.valid_z || []).length;
                totalOrigArea += sub.area_um2;
                if (curN > 0) {
                    totalRemainArea += sub.area_um2 * (curN / origN);
                    allEliminated = false;
                }
            });
            const fraction = totalOrigArea > 0
                ? totalRemainArea / totalOrigArea : 0;
            // Update both directions in overlapTable
            const tableRows = findTableRows(source, target);
            tableRows.forEach(row => {
                if (!row._orig_area) row._orig_area = row.area;
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
            // Re-render the matrix if the modal is open
            if (matrixModal.classList.contains('active')) {
                renderHeatmap();
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
                // Update matrix if open
                if (matrixModal.classList.contains('active')) renderHeatmap();

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
            updateGJTrace();
            updateRemoveGJButton();
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
            if (gjTraceIdx === undefined) return;
            const allX = [], allY = [], allZ = [], allText = [];
            gapJunctions.forEach((gj, i) => {
                allX.push(gj.x);
                allY.push(gj.y);
                allZ.push(gj.z);
                allText.push('GJ #' + (i + 1) + ': ' + gj.source + ' \u2194 ' + gj.target);
            });
            const trace = plotDiv.data[gjTraceIdx];
            if (trace) {
                trace.x = allX; trace.y = allY; trace.z = allZ;
                trace.text = allText;
                trace.visible = allX.length > 0;
                scheduleRedraw();
            }
        }

        // ── Matrix modal ────────────────────────────────────────────
        btnMatrix.addEventListener('click', function() {
            renderHeatmap();
            renderGJTab();
            renderConnectivityMatrix();
            matrixModal.classList.add('active');
        });
        document.getElementById('matrixClose').addEventListener('click', () => {
            matrixModal.classList.remove('active');
        });
        matrixModal.addEventListener('click', function(e) {
            if (e.target === matrixModal) matrixModal.classList.remove('active');
        });
        // Tab switching
        const tabMap = {
            overlaps:     { el: 'tabOverlaps',      title: 'Overlap Area Matrix (\u00b5m\u00b2)' },
            gapjunctions: { el: 'tabGapJunctions',   title: 'Putative Gap Junctions' },
            connectivity: { el: 'tabConnectivity',   title: 'Connectivity Matrix (GJ + Chemical Synapses)' },
            circuit:      { el: 'tabCircuit',         title: 'Tier 1 Circuit Model' },
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
                    if (target === 'circuit') renderCircuitModel();
                }
            });
        });

        function renderHeatmap() {
            const lookup = {};
            overlapTable.forEach(row => { lookup[row.source + '|' + row.target] = row; });
            const allNames = new Set();
            overlapTable.forEach(r => { allNames.add(r.source); allNames.add(r.target); });
            const names = Array.from(allNames).sort();
            let maxArea = 0;
            overlapTable.forEach(r => { if (r.area > maxArea) maxArea = r.area; });
            if (maxArea === 0) maxArea = 1;

            let html = '<table><thead><tr><th class="corner"></th>';
            names.forEach(n => { html += '<th>' + n.replace('_', '<br>') + '</th>'; });
            html += '</tr></thead><tbody>';
            names.forEach(src => {
                html += '<tr><th class="row-header" style="text-align:right;">' + src + '</th>';
                names.forEach(tgt => {
                    if (src === tgt) { html += '<td class="diagonal">\u2014</td>'; return; }
                    const row = lookup[src + '|' + tgt];
                    if (!row || row.area <= 0) {
                        html += '<td class="no-data" data-src="' + src + '" data-tgt="' + tgt + '">-</td>';
                        return;
                    }
                    const frac = Math.min(1, row.area / maxArea);
                    const r = Math.round(40 + 215 * frac);
                    const g = Math.round(40 * (1 - frac));
                    const b = Math.round(40 * (1 - frac));
                    const style = 'background:rgb(' + r + ',' + g + ',' + b + ');color:'
                        + (frac > 0.5 ? '#fff' : '#ddd') + ';'
                        + (row.status === 'eliminated' ? 'text-decoration:line-through;opacity:0.4;' : '');
                    html += '<td style="' + style + '" data-src="' + src + '" data-tgt="' + tgt
                        + '" title="' + src + ' \u2192 ' + tgt + ': ' + row.area.toFixed(3)
                        + ' \u00b5m\u00b2, ' + row.patches + ' patches">'
                        + row.area.toFixed(2) + '</td>';
                });
                html += '</tr>';
            });
            html += '</tbody></table>';
            heatmapDiv.innerHTML = html;

            heatmapDiv.querySelectorAll('td[data-src]').forEach(td => {
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

        function renderGJTab() {
            if (gapJunctions.length === 0) {
                gjContainer.innerHTML = '<p style="color:#888;font-size:12px;">No putative gap junctions marked yet.<br>'
                    + 'Select an overlap or contact, navigate to the location, then click \u26a1 Mark Gap Junction.</p>';
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
            // Collect all neuron names from overlaps + synapses + gap junctions
            const allNames = new Set();
            overlapTable.forEach(r => { allNames.add(r.source); allNames.add(r.target); });
            synapseList.forEach(s => { allNames.add(s.source); allNames.add(s.target); });
            gapJunctions.forEach(gj => { allNames.add(gj.source); allNames.add(gj.target); });
            const names = Array.from(allNames).sort();

            // Build chemical synapse count: source -> target (directional)
            const chemCount = {};
            synapseList.forEach(s => {
                const key = s.source + '|' + s.target;
                chemCount[key] = (chemCount[key] || 0) + 1;
            });

            // Build GJ count: undirected pair
            const gjCount = {};
            gapJunctions.forEach(gj => {
                const a = gj.source, b = gj.target;
                const k1 = a + '|' + b, k2 = b + '|' + a;
                gjCount[k1] = (gjCount[k1] || 0) + 1;
                gjCount[k2] = (gjCount[k2] || 0) + 1;
            });

            let html = '<p style="color:#aaa;font-size:10px;margin:0 0 6px;">'
                + 'Rows \u2192 Columns. '
                + '<span style="color:#39FF14;">\u25cf GJ</span> &nbsp; '
                + '<span style="color:#FFD700;">\u25cf Chem. Syn.</span> &nbsp; '
                + '<span style="color:#00BFFF;">\u25cf Both</span></p>';
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
                    let bg, fg;
                    if (nGJ > 0 && nChem > 0) { bg = '#00BFFF'; fg = '#000'; }
                    else if (nGJ > 0)          { bg = '#39FF14'; fg = '#000'; }
                    else                       { bg = '#FFD700'; fg = '#000'; }
                    const parts = [];
                    if (nGJ > 0) parts.push(nGJ + ' GJ');
                    if (nChem > 0) parts.push(nChem + ' syn');
                    const cellText = parts.join('+');
                    const tip = src + ' \u2192 ' + tgt + ': ' + parts.join(', ');
                    html += '<td style="background:' + bg + ';color:' + fg
                        + ';font-size:9px;font-weight:bold;" title="' + tip + '">'
                        + cellText + '</td>';
                });
                html += '</tr>';
            });
            html += '</tbody></table>';
            connectivityContainer.innerHTML = html;
        }

        // ── Tier 1 Circuit Model ────────────────────────────────────
        let circuitInitialized = false;
        function renderCircuitModel() {
            if (circuitInitialized) return;
            circuitInitialized = true;

            const dt = 0.01;  // ms (match user's code)
            const Cm = 1.0;
            const VCa = 120, V_Na = 50, V_K = -77, E_SYN = 0;

            // ── Cell names & index mapping (match RAW_COUNTS order) ──
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

            // ── RAW_COUNTS: rows=pre, cols=post ──
            const RAW_COUNTS = [
             // MOT_L MOT_R MOS_L MOS_R VS1_L VS1_R VS2_L VS2_R VS3_L VS3_R VS4_L VS4_R HSN_L HSN_R HSE_L HSE_R HSS_L HSS_R
                [0,    0,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOT_L
                [0,    0,    0,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOT_R
                [3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOS_L
                [0,    6,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // MOS_R
                [0,    0,    0,    0,    0,    0,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS1_L
                [0,    0,    0,    0,    0,    0,    0,    4,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0],  // VS1_R
                [0,    0,    6,    0,    1,    0,    0,    0,    4,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS2_L
                [0,    0,    0,    0,    0,    6,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS2_R
                [0,    0,    6,    0,    0,    0,   12,    0,    0,    0,    2,    0,    0,    0,    0,    0,    0,    0],  // VS3_L
                [0,    0,    0,    5,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS3_R
                [0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // VS4_L
                [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    0,    0],  // VS4_R
                [2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0],  // HSN_L
                [0,    9,    0,    4,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    7,    0,    3],  // HSN_R
                [4,    0,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    5,    0,    0,    0,    0,    0],  // HSE_L
                [0,    6,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    2,    0,    0,    0,    0],  // HSE_R
                [4,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    2,    0,    0,    0,    0,    0],  // HSS_L
                [0,    3,    0,    2,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0],  // HSS_R
            ];

            // ── Gate kinetics ──
            function alphaM(V) { const x = V+40; return Math.abs(x)<1e-7 ? 1 : 0.1*x/(1-Math.exp(-x/10)); }
            function betaM(V)  { return 4*Math.exp(-(V+65)/18); }
            function alphaH(V) { return 0.07*Math.exp(-(V+65)/20); }
            function betaH(V)  { return 1/(1+Math.exp(-(V+35)/10)); }
            function alphaN(V) { const x = V+55; return Math.abs(x)<1e-7 ? 0.1 : 0.01*x/(1-Math.exp(-x/10)); }
            function betaN(V)  { return 0.125*Math.exp(-(V+65)/80); }
            function ss(af,bf,V) { const a=af(V),b=bf(V); return a/(a+b); }
            function mInfCa(V)  { return 1/(1+Math.exp((-61-V)/4.2)); }
            function hInfCa(V)  { return 1/(1+Math.exp((V+85.5)/8.6)); }
            function tauHCa(V)  { return 40+30/(1+Math.exp((V+84)/7.3))*Math.exp((V+160)/30); }
            function mNaPinf(V) { return 1/(1+Math.exp(-(V+52)/5)); }

            // ── LPTC cell (graded, non-spiking) ──
            function createLPTC(name, Rin, VL, gVT, gL, gK) {
                return {
                    name, type: 'LPTC',
                    V: VL, gVT, gL, Rin, gK, VL,
                    hCa: hInfCa(VL),
                    mNa: ss(alphaM,betaM,VL), hNa: ss(alphaH,betaH,VL),
                    n: ss(alphaN,betaN,VL),
                };
            }
            function stepLPTC(c, Itot) {
                const v = c.V;
                c.hCa += (hInfCa(v)-c.hCa)/tauHCa(v)*dt;
                const iT = c.gVT * Math.pow(mInfCa(v),3) * c.hCa * (v-VCa);
                c.n += (alphaN(v)*(1-c.n)-betaN(v)*c.n)*dt;
                const iK = c.gK * Math.pow(c.n,4) * (v-V_K);
                const iL = (c.gL + 1/c.Rin) * (v - c.VL);
                c.V = v + (-iT - iK - iL + Itot)/Cm*dt;
                return c.V;
            }

            // ── MN cell (HH spiking) ──
            function createMN(name, Rin, VL, gVT, gL, gNa, gK, gNaP) {
                return {
                    name, type: 'MN',
                    V: VL, gVT, gL, Rin, gNa, gK, VL, gNaP,
                    hCa: hInfCa(VL),
                    mNa: ss(alphaM,betaM,VL), hNa: ss(alphaH,betaH,VL),
                    n: ss(alphaN,betaN,VL),
                };
            }
            function stepMN(c, Itot) {
                const v = c.V;
                c.hCa += (hInfCa(v)-c.hCa)/tauHCa(v)*dt;
                const iT = c.gVT * Math.pow(mInfCa(v),3) * c.hCa * (v-VCa);
                c.mNa += (alphaM(v)*(1-c.mNa)-betaM(v)*c.mNa)*dt;
                c.hNa += (alphaH(v)*(1-c.hNa)-betaH(v)*c.hNa)*dt;
                c.n   += (alphaN(v)*(1-c.n)  -betaN(v)*c.n  )*dt;
                const iNa  = c.gNa  * Math.pow(c.mNa,3)*c.hNa * (v-V_Na);
                const iK   = c.gK   * Math.pow(c.n,4)          * (v-V_K);
                const iNaP = c.gNaP * mNaPinf(v)                * (v-V_Na);
                const iL   = (c.gL + 1/c.Rin) * (v - c.VL);
                c.V = v + (-iT - iNa - iK - iNaP - iL + Itot)/Cm*dt;
                return c.V;
            }

            // ── LP-filtered bidirectional gap junction ──
            function createGJ(G, C) {
                return { G, tau: (G>1e-9 ? C/G : 0), Vf: 0 };
            }
            function gjPair(gj, Va, Vb) {
                const raw = Vb - Va;
                if (gj.tau > 1e-9) gj.Vf += (raw - gj.Vf)/gj.tau*dt;
                else gj.Vf = raw;
                const IA = gj.G * gj.Vf;
                return [IA, -IA];  // [into A, into B]
            }

            // ── Graded synapse (LPTC pre) ──
            function createGradedSyn(nSyn, gPerSyn) {
                return { gMax: nSyn*gPerSyn, Vthresh: -60, Vscale: 30 };
            }
            function gradedCurrent(s, Vpre, Vpost) {
                if (s.gMax < 1e-15) return 0;
                const rel = Math.max(0, Math.min(1, (Vpre-s.Vthresh)/s.Vscale));
                return -s.gMax * rel * (Vpost - E_SYN);
            }

            // ── Alpha synapse (MN pre, spiking) ──
            function createAlphaSyn(nSyn, gPerSyn, tau) {
                return { gMax: nSyn*gPerSyn, tau, g: 0, dg: 0, prevV: -65, thresh: 0 };
            }
            function alphaStep(s, Vpre, Vpost) {
                if (Vpre > s.thresh && s.prevV <= s.thresh) s.dg += s.gMax/s.tau;
                s.prevV = Vpre;
                s.g  += s.dg * dt;
                s.dg -= s.dg / s.tau * dt;
                s.g   = Math.max(0, s.g);
                if (s.gMax < 1e-15) return 0;
                return -s.g * (Vpost - E_SYN);
            }

            // ── Default parameters ──
            let pGVT_l = 0.5, pGL_l = 0.05, pGK_l = 2.0;
            let pRinVS1 = 150, pVrVS1 = -40, pRinHS = 150, pVrHS = -45;
            let pGVT_m = 0.3, pGL_m = 0.3, pRinM = 300;
            let pGNa = 120, pGK_m = 36, pVLm = -65, pGNaP = 0.5;
            let pGlptc = 0.05, pClptc = 0.05, pGmn = 0.1, pCmn = 0.8;
            let pGgrad = 0.005, pGspike = 0.02, pTauSyn = 5;

            function buildAndRun() {
                // ── Instantiate cells ──
                const cells = [];
                const VS_Vr  = [pVrVS1, pVrVS1-5, pVrVS1-10, pVrVS1-15];
                const VS_Rin = [pRinVS1, pRinVS1-10, pRinVS1-20, pRinVS1-30];
                CELL_NAMES.forEach(n => {
                    if (n.startsWith('VS')) {
                        const k = parseInt(n[2]) - 1;
                        cells.push(createLPTC(n, VS_Rin[k], VS_Vr[k], pGVT_l, pGL_l, pGK_l));
                    } else if (n.startsWith('HS')) {
                        cells.push(createLPTC(n, pRinHS, pVrHS, pGVT_l, pGL_l, pGK_l));
                    } else {
                        cells.push(createMN(n, pRinM, pVLm, pGVT_m, pGL_m, pGNa, pGK_m, pGNaP));
                    }
                });

                // ── Chemical synapses from RAW_COUNTS ──
                const synapses = [];
                for (let pi = 0; pi < N_CELLS; pi++) {
                    for (let qi = 0; qi < N_CELLS; qi++) {
                        const cnt = RAW_COUNTS[pi][qi];
                        if (cnt === 0) continue;
                        if (SPIKING.has(CELL_NAMES[pi])) {
                            synapses.push({ pre: pi, post: qi, obj: createAlphaSyn(cnt, pGspike, pTauSyn) });
                        } else {
                            synapses.push({ pre: pi, post: qi, obj: createGradedSyn(cnt, pGgrad) });
                        }
                    }
                }

                // ── Gap junctions (bidirectional, LP-filtered) ──
                const gjList = [];
                // Within VS chains
                ['L','R'].forEach(s => {
                    for (let k = 1; k <= 3; k++)
                        gjList.push({ a: CI['VS'+k+'_'+s], b: CI['VS'+(k+1)+'_'+s], gj: createGJ(pGlptc, pClptc) });
                    // Within HS chains
                    gjList.push({ a: CI['HSN_'+s], b: CI['HSE_'+s], gj: createGJ(pGlptc, pClptc) });
                    gjList.push({ a: CI['HSE_'+s], b: CI['HSS_'+s], gj: createGJ(pGlptc, pClptc) });
                    // LPTC axon \u2194 MN dendrite (bidirectional)
                    const mos = CI['MOS_'+s], mot = CI['MOT_'+s];
                    for (let k = 1; k <= 4; k++)
                        gjList.push({ a: CI['VS'+k+'_'+s], b: mos, gj: createGJ(pGmn, pCmn) });
                    ['HSN','HSE','HSS'].forEach(h => {
                        gjList.push({ a: CI[h+'_'+s], b: mos, gj: createGJ(pGmn, pCmn) });
                        gjList.push({ a: CI[h+'_'+s], b: mot, gj: createGJ(pGmn, pCmn) });
                    });
                });

                // ── Simulation ──
                const nSteps = Math.round(simTime / dt);
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
                    if (t >= stimStart && t <= stimEnd) {
                        stimTargets.forEach(sn => {
                            if (CI[sn] !== undefined) extI[CI[sn]] += stimAmp;
                        });
                    }
                    // Noise
                    for (let n = 0; n < N_CELLS; n++)
                        extI[n] += noiseLevel * (Math.random()*2-1);

                    // 4. Step all cells
                    for (let n = 0; n < N_CELLS; n++) {
                        const Itot = gjI[n] + chemI[n] + extI[n];
                        if (cells[n].type === 'LPTC') stepLPTC(cells[n], Itot);
                        else stepMN(cells[n], Itot);
                        rec[CELL_NAMES[n]][step] = cells[n].V;
                    }
                }
                return { t: tArr, records: rec };
            }

            // ── Simulation state ──
            let simTime = 1500, stimStart = 90, stimEnd = 590, stimAmp = 10;
            let noiseLevel = 3;
            let stimTargets = ['VS1_L','VS2_L','VS3_L','VS4_L'];

            // ── Build UI ──
            let html = '<div style="display:flex;flex-direction:column;gap:6px;">';

            // Wiring diagram (SVG)
            html += '<div id="wiringDiagram" style="background:#1a1a2e;border:1px solid #444;'
                + 'border-radius:4px;padding:8px;overflow-x:auto;"></div>';

            // Controls
            html += '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:4px 0;">';
            html += '<label style="color:#ccc;font-size:11px;">Stim: '
                + '<select id="circStimGroup" style="background:#333;color:#fff;border:1px solid #555;font-size:11px;">'
                + '<option value="VS_L">VS1-4 Left</option>'
                + '<option value="VS_R">VS1-4 Right</option>'
                + '<option value="HS_L">HSN/E/S Left</option>'
                + '<option value="HS_R">HSN/E/S Right</option>'
                + '<option value="ALL_L">All Left LPTCs</option>'
                + '<option value="MN_L">MOS+MOT Left</option>'
                + '</select></label>';
            html += '<label style="color:#ccc;font-size:10px;">Amp: '
                + '<input id="circStimAmp" type="number" value="10" step="1" min="-100" max="100" '
                + 'style="width:45px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ccc;font-size:10px;">t<sub>start</sub>: '
                + '<input id="circStimStart" type="number" value="90" step="10" min="0" max="8000" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ccc;font-size:10px;">t<sub>end</sub>: '
                + '<input id="circStimEnd" type="number" value="590" step="10" min="0" max="8000" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ccc;font-size:10px;">T<sub>max</sub>: '
                + '<input id="circSimTime" type="number" value="1500" step="100" min="100" max="10000" '
                + 'style="width:55px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ccc;font-size:10px;">Noise: '
                + '<input id="circNoise" type="number" value="3" step="0.5" min="0" max="10" '
                + 'style="width:40px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<button id="circRun" style="background:#2E7D32;color:#fff;border:1px solid #4CAF50;'
                + 'padding:4px 14px;border-radius:3px;cursor:pointer;font-size:11px;font-weight:bold;">'
                + '\u25b6 Run</button>';
            html += '</div>';

            // GJ/Synapse parameter row
            html += '<details style="color:#aaa;font-size:10px;"><summary style="cursor:pointer;color:#aed581;">GJ &amp; Synapse Parameters</summary>';
            html += '<div style="display:flex;gap:8px;flex-wrap:wrap;padding:4px 0;">';
            html += '<label style="color:#aed581;font-size:10px;">G<sub>LPTC-GJ</sub>: '
                + '<input id="pGlptc" type="number" value="0.05" step="0.005" min="0" max="1" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#aed581;font-size:10px;">C<sub>LPTC-GJ</sub>: '
                + '<input id="pClptc" type="number" value="0.05" step="0.005" min="0" max="1" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#aed581;font-size:10px;">G<sub>MN-GJ</sub>: '
                + '<input id="pGmn" type="number" value="0.1" step="0.005" min="0" max="1" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#aed581;font-size:10px;">C<sub>MN-GJ</sub>: '
                + '<input id="pCmn" type="number" value="0.8" step="0.01" min="0" max="2" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ffcc02;font-size:10px;">g<sub>grad</sub>: '
                + '<input id="pGgrad" type="number" value="0.005" step="0.001" min="0" max="0.05" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ffcc02;font-size:10px;">g<sub>spike</sub>: '
                + '<input id="pGspike" type="number" value="0.02" step="0.005" min="0" max="0.2" '
                + 'style="width:50px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '<label style="color:#ffcc02;font-size:10px;">\u03c4<sub>syn</sub>: '
                + '<input id="pTauSyn" type="number" value="5" step="0.5" min="0.5" max="50" '
                + 'style="width:45px;background:#333;color:#fff;border:1px solid #555;font-size:10px;"></label>';
            html += '</div></details>';

            // Plot areas
            html += '<div id="circPlotLPTC" style="width:100%;height:180px;background:#1a1a1a;border:1px solid #444;"></div>';
            html += '<div id="circPlotMN" style="width:100%;height:180px;background:#1a1a1a;border:1px solid #444;"></div>';
            html += '</div>';

            circuitContainer.innerHTML = html;

            // ── Draw SVG wiring diagram ──
            (function drawWiring() {
                const W = 880, H = 340;
                const CVS='#ce93d8', CHS='#4fc3f7', CMOS='#ef5350', CMOT='#ff7043';
                const CGJ='#aed581', CCHEM='#ffcc02', CT='#ccc', CBG='#0d1b2e';

                let svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+W+' '+H+'" '
                    + 'style="width:100%;max-height:320px;font-family:sans-serif;">';
                svg += '<rect width="'+W+'" height="'+H+'" fill="#1a1a2e"/>';

                // Title
                svg += '<text x="'+W/2+'" y="20" text-anchor="middle" fill="'+CT+'" font-size="13" font-weight="bold">'
                    + 'Circuit Wiring Diagram</text>';
                svg += '<text x="'+W/2+'" y="34" text-anchor="middle" fill="#888" font-size="9">'
                    + '\u27f7 green dashed = bidirectional GJ (LP-filtered) &nbsp; '
                    + '\u2192 yellow = chemical synapse (n = count)</text>';

                // Positions  {name: [cx, cy]}
                const pos = {};
                // Left hemisphere
                const lx_vs = 55, lx_hs = 165, lx_mos = 290, lx_mot = 290;
                const vs_ys = [70, 125, 180, 235];
                const hs_ys = [70, 125, 180];
                for (let k = 0; k < 4; k++) pos['VS'+(k+1)+'_L'] = [lx_vs, vs_ys[k]];
                ['HSN','HSE','HSS'].forEach((h,i) => { pos[h+'_L'] = [lx_hs, hs_ys[i]]; });
                pos['MOS_L'] = [lx_mos, 97];
                pos['MOT_L'] = [lx_mot, 210];

                // Right hemisphere (mirrored)
                const rx_vs = W-55, rx_hs = W-165, rx_mos = W-290, rx_mot = W-290;
                for (let k = 0; k < 4; k++) pos['VS'+(k+1)+'_R'] = [rx_vs, vs_ys[k]];
                ['HSN','HSE','HSS'].forEach((h,i) => { pos[h+'_R'] = [rx_hs, hs_ys[i]]; });
                pos['MOS_R'] = [rx_mos, 97];
                pos['MOT_R'] = [rx_mot, 210];

                // Draw boxes
                function neuronBox(name, color) {
                    const p = pos[name], bw = 72, bh = 30;
                    svg += '<rect x="'+(p[0]-bw/2)+'" y="'+(p[1]-bh/2)+'" width="'+bw+'" height="'+bh+'" '
                        + 'rx="5" fill="'+CBG+'" stroke="'+color+'" stroke-width="1.5"/>';
                    svg += '<text x="'+p[0]+'" y="'+(p[1]+4)+'" text-anchor="middle" fill="'+color+'" font-size="9" font-weight="bold">'
                        + name + '</text>';
                }
                CELL_NAMES.forEach(n => {
                    let c = CMOS;
                    if (n.startsWith('VS')) c = CVS;
                    else if (n.startsWith('HS')) c = CHS;
                    else if (n.startsWith('MOT')) c = CMOT;
                    neuronBox(n, c);
                });

                // GJ lines (dashed green, bidirectional)
                function gjLine(a, b) {
                    const p1 = pos[a], p2 = pos[b];
                    svg += '<line x1="'+p1[0]+'" y1="'+p1[1]+'" x2="'+p2[0]+'" y2="'+p2[1]+'" '
                        + 'stroke="'+CGJ+'" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.7"/>';
                    const mx = (p1[0]+p2[0])/2, my = (p1[1]+p2[1])/2;
                    svg += '<line x1="'+(mx-4)+'" y1="'+my+'" x2="'+(mx+4)+'" y2="'+my+'" '
                        + 'stroke="'+CGJ+'" stroke-width="2.5"/>';
                }
                // VS chain GJ
                ['L','R'].forEach(s => {
                    for (let k = 1; k <= 3; k++) gjLine('VS'+k+'_'+s, 'VS'+(k+1)+'_'+s);
                    gjLine('HSN_'+s, 'HSE_'+s);
                    gjLine('HSE_'+s, 'HSS_'+s);
                });

                // LPTC \u2194 MN GJ (bidirectional arrows)
                function gjArrow(a, b) {
                    const p1 = pos[a], p2 = pos[b];
                    const dx = p2[0]-p1[0], dy = p2[1]-p1[1];
                    const len = Math.sqrt(dx*dx+dy*dy);
                    const ux = dx/len, uy = dy/len;
                    const x1 = p1[0]+ux*38, y1 = p1[1]+uy*16;
                    const x2 = p2[0]-ux*38, y2 = p2[1]-uy*16;
                    svg += '<line x1="'+x1+'" y1="'+y1+'" x2="'+x2+'" y2="'+y2+'" '
                        + 'stroke="'+CGJ+'" stroke-width="1" stroke-dasharray="4,3" opacity="0.5"/>';
                }
                ['L','R'].forEach(s => {
                    for (let k = 1; k <= 4; k++) gjArrow('VS'+k+'_'+s, 'MOS_'+s);
                    ['HSN','HSE','HSS'].forEach(h => {
                        gjArrow(h+'_'+s, 'MOS_'+s);
                        gjArrow(h+'_'+s, 'MOT_'+s);
                    });
                });

                // Chemical synapse arrows (yellow, directional)
                function chemArrow(pre, post, n, dy) {
                    const p1 = pos[pre], p2 = pos[post];
                    if (!p1 || !p2) return;
                    const dx2 = p2[0]-p1[0], dy2 = p2[1]-p1[1];
                    const len = Math.sqrt(dx2*dx2+dy2*dy2);
                    if (len < 1) return;
                    const ux = dx2/len, uy = dy2/len;
                    const x1 = p1[0]+ux*38, y1 = p1[1]+uy*16 + (dy||0);
                    const x2 = p2[0]-ux*38, y2 = p2[1]-uy*16 + (dy||0);
                    const mid = 'M'+x1+','+y1+' L'+x2+','+y2;
                    const aid = 'ca_'+pre+'_'+post;
                    svg += '<defs><marker id="'+aid+'" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">'
                        + '<path d="M0,0 L6,3 L0,6" fill="'+CCHEM+'"/></marker></defs>';
                    svg += '<path d="'+mid+'" stroke="'+CCHEM+'" stroke-width="'+Math.max(0.5,n*0.3)+'" '
                        + 'fill="none" marker-end="url(#'+aid+')" opacity="0.7"/>';
                    const mx = (x1+x2)/2, my = (y1+y2)/2 - 4;
                    svg += '<text x="'+mx+'" y="'+my+'" text-anchor="middle" fill="'+CCHEM+'" font-size="7" font-weight="bold">'
                        + n + '</text>';
                }
                // Draw chemical synapses from RAW_COUNTS
                for (let pi = 0; pi < N_CELLS; pi++) {
                    for (let qi = 0; qi < N_CELLS; qi++) {
                        const cnt = RAW_COUNTS[pi][qi];
                        if (cnt === 0) continue;
                        chemArrow(CELL_NAMES[pi], CELL_NAMES[qi], cnt, (pi%2===0?-3:3));
                    }
                }

                // Legend
                svg += '<text x="10" y="'+( H-30)+'" fill="#888" font-size="8">'
                    + 'GJ: LPTC chains (within-type) + LPTC axon \u2194 MN dendrite (bidirectional, LP-filtered)</text>';
                svg += '<text x="10" y="'+(H-18)+'" fill="#888" font-size="8">'
                    + 'Chem syn: Graded (LPTC pre, Manor 1997) / Alpha (MN pre, Dayan\u0026Abbott 2001) / E_syn=0mV</text>';
                svg += '<text x="10" y="'+(H-6)+'" fill="#888" font-size="8">'
                    + 'MOT\u2194MOS: dendrodendritic chemical synapses &middot; '
                    + 'LPTC\u2192MN: axon\u2192dendrite &middot; Baines \u0026 Bate 1998</text>';

                // Legend symbols
                svg += '<line x1="'+(W-200)+'" y1="'+(H-28)+'" x2="'+(W-170)+'" y2="'+(H-28)+'" '
                    + 'stroke="'+CGJ+'" stroke-width="2" stroke-dasharray="5,3"/>';
                svg += '<text x="'+(W-165)+'" y="'+(H-25)+'" fill="'+CGJ+'" font-size="8">Bidirectional GJ</text>';
                svg += '<line x1="'+(W-200)+'" y1="'+(H-14)+'" x2="'+(W-170)+'" y2="'+(H-14)+'" '
                    + 'stroke="'+CCHEM+'" stroke-width="2"/>';
                svg += '<text x="'+(W-165)+'" y="'+(H-11)+'" fill="'+CCHEM+'" font-size="8">Chemical synapse</text>';

                svg += '</svg>';
                document.getElementById('wiringDiagram').innerHTML = svg;
            })();

            // ── Wire up controls ──
            const stimGroupMap = {
                'VS_L':  ['VS1_L','VS2_L','VS3_L','VS4_L'],
                'VS_R':  ['VS1_R','VS2_R','VS3_R','VS4_R'],
                'HS_L':  ['HSN_L','HSE_L','HSS_L'],
                'HS_R':  ['HSN_R','HSE_R','HSS_R'],
                'ALL_L': ['VS1_L','VS2_L','VS3_L','VS4_L','HSN_L','HSE_L','HSS_L'],
                'MN_L':  ['MOS_L','MOT_L'],
            };

            function readParams() {
                stimTargets = stimGroupMap[document.getElementById('circStimGroup').value] || stimTargets;
                stimAmp   = parseFloat(document.getElementById('circStimAmp').value)   || 10;
                simTime   = parseFloat(document.getElementById('circSimTime').value)    || 1500;
                stimStart = parseFloat(document.getElementById('circStimStart').value)  || 90;
                stimEnd   = parseFloat(document.getElementById('circStimEnd').value)    || 590;
                noiseLevel= parseFloat(document.getElementById('circNoise').value)      || 3;
                pGlptc  = parseFloat(document.getElementById('pGlptc').value)  || 0.05;
                pClptc  = parseFloat(document.getElementById('pClptc').value)  || 0.05;
                pGmn    = parseFloat(document.getElementById('pGmn').value)    || 0.1;
                pCmn    = parseFloat(document.getElementById('pCmn').value)    || 0.8;
                pGgrad  = parseFloat(document.getElementById('pGgrad').value)  || 0.005;
                pGspike = parseFloat(document.getElementById('pGspike').value) || 0.02;
                pTauSyn = parseFloat(document.getElementById('pTauSyn').value) || 5;
            }

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
                let step = 1;
                if (tMs.length > 5000) step = Math.ceil(tMs.length / 5000);
                const tPlot = [], indices = [];
                for (let i = 0; i < tMs.length; i += step) { tPlot.push(tMs[i]); indices.push(i); }

                // LPTC traces
                const lptcOrder = [];
                ['L','R'].forEach(s => {
                    for (let k = 1; k <= 4; k++) lptcOrder.push('VS'+k+'_'+s);
                    ['HSN','HSE','HSS'].forEach(h => lptcOrder.push(h+'_'+s));
                });
                const colsVS = ['#9c27b0','#7b1fa2','#ce93d8','#e1bee7','#6a1b9a','#4a148c','#ab47bc','#ba68c8'];
                const colsHS = ['#4fc3f7','#0288d1','#01579b','#80d8ff','#40c4ff','#0091ea'];
                const lptcColors = {};
                let vi=0, hi=0;
                lptcOrder.forEach(n => {
                    if (n.startsWith('VS')) lptcColors[n] = colsVS[vi++ % colsVS.length];
                    else lptcColors[n] = colsHS[hi++ % colsHS.length];
                });
                const lptcTraces = lptcOrder.map(n => ({
                    x: tPlot, y: indices.map(i => res.records[n][i]),
                    name: n, type: 'scatter', mode: 'lines',
                    line: { color: lptcColors[n], width: 1 },
                }));
                lptcTraces.push({
                    x: [stimStart, stimEnd, stimEnd, stimStart],
                    y: [-80, -80, 20, 20],
                    fill: 'toself', fillcolor: 'rgba(255,255,0,0.08)',
                    line: { width: 0 }, showlegend: false, hoverinfo: 'skip',
                    type: 'scatter', mode: 'lines',
                });
                Plotly.react('circPlotLPTC', lptcTraces, {
                    title: { text: 'LPTCs (VS + HS) \u2014 soma Vm', font: { size: 11, color: '#ccc' } },
                    xaxis: { title: 'ms', color: '#888', gridcolor: '#333' },
                    yaxis: { title: 'mV', color: '#888', gridcolor: '#333', range: [-80, 20] },
                    paper_bgcolor: '#1a1a1a', plot_bgcolor: '#1a1a1a',
                    legend: { font: { size: 7, color: '#ccc' }, bgcolor: 'rgba(0,0,0,0.5)' },
                    margin: { l: 40, r: 10, t: 26, b: 30 },
                }, { responsive: true });

                const mnNames = ['MOS_L','MOS_R','MOT_L','MOT_R'];
                const mnColors = { MOS_L:'#ef5350', MOS_R:'#e53935', MOT_L:'#ff7043', MOT_R:'#ff5722' };
                const mnTraces = mnNames.map(n => ({
                    x: tPlot, y: indices.map(i => res.records[n][i]),
                    name: n, type: 'scatter', mode: 'lines',
                    line: { color: mnColors[n], width: 1.2 },
                }));
                mnTraces.push({
                    x: [stimStart, stimEnd, stimEnd, stimStart],
                    y: [-80, -80, 60, 60],
                    fill: 'toself', fillcolor: 'rgba(255,255,0,0.08)',
                    line: { width: 0 }, showlegend: false, hoverinfo: 'skip',
                    type: 'scatter', mode: 'lines',
                });
                Plotly.react('circPlotMN', mnTraces, {
                    title: { text: 'Motor Neurons (MOS + MOT) \u2014 soma Vm', font: { size: 11, color: '#ccc' } },
                    xaxis: { title: 'ms', color: '#888', gridcolor: '#333' },
                    yaxis: { title: 'mV', color: '#888', gridcolor: '#333', range: [-80, 60] },
                    paper_bgcolor: '#1a1a1a', plot_bgcolor: '#1a1a1a',
                    legend: { font: { size: 9, color: '#ccc' }, bgcolor: 'rgba(0,0,0,0.5)' },
                    margin: { l: 40, r: 10, t: 26, b: 30 },
                }, { responsive: true });
            }

            // Auto-run on init
            readParams();
            const res = buildAndRun();
            plotResults(res);
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

    # Read N_TOP_PATCHES from neurons.json
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neurons.json')
    with open(cfg_path, 'r') as f:
        _cfg = json.load(f)
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

    for (na, nb), grp in pair_groups.items():
        mx = grp['mid_x'].values
        my = grp['mid_y'].values
        mz = grp['mid_z'].values

        meta_items = em_meta_list.get((na, nb), [])

        if len(meta_items) <= 1:
            # Single cluster (or no metadata) — same as before
            meta = meta_items[0] if meta_items else {}
            em_idx = meta.get('idx', -1)
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
                for neuron, other in [(na, nb), (nb, na)]:
                    per_pair.setdefault(neuron, []).append({
                        'other': other,
                        'x': mx[cmask].tolist(),
                        'y': my[cmask].tolist(),
                        'z': mz[cmask].tolist(),
                        'idx': meta.get('idx', -1),
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
    print(f"[overlaps] {len(df)} overlap midpoints across "
          f"{len(per_neuron)} neurons ({total} trace points)")
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
    df = pd.read_csv(csv_file)

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
        """Convert a sub-DataFrame of faces into Mesh3d x/y/z/i/j/k arrays."""
        xs, ys, zs = [], [], []
        ii, jj, kk = [], [], []
        vert_map = {}  # (x,y,z) -> index for dedup
        labels = []

        for _, row in sub_df.iterrows():
            verts = [
                (row['vertex1_x'], row['vertex1_y'], row['vertex1_z']),
                (row['vertex2_x'], row['vertex2_y'], row['vertex2_z']),
                (row['vertex3_x'], row['vertex3_y'], row['vertex3_z']),
            ]
            idxs = []
            for v in verts:
                if v not in vert_map:
                    vert_map[v] = len(xs)
                    xs.append(v[0])
                    ys.append(v[1])
                    zs.append(v[2])
                idxs.append(vert_map[v])
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
    print(f"[overlap faces] {len(df)} valid faces across "
          f"{len(per_neuron_faces)} neurons "
          f"({total_verts} vertices, {total_faces} triangles in traces)")
    return per_neuron_faces, per_pair_faces


def load_overlap_table(results_dir):
    """Build overlap summary table from all_results_combined.csv.

    Counts Top1-N patches per pair for patch count column.
    """
    csv_file = os.path.join(results_dir, 'all_results_combined.csv')
    df = pd.read_csv(csv_file)
    contacts = df[df['Has_Contact'] == True]

    # Read N_TOP_PATCHES
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neurons.json')
    with open(cfg_path, 'r') as f:
        _cfg = json.load(f)
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
    """Load mesh and return subsampled point cloud."""
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
    """Build 3D figure with mesh, contact, synapse and overlap traces."""
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
            trace_info[f"{neuron_name}_synapses"] = len(traces)
            traces.append(go.Scatter3d(
                x=neuron_synapses['x'],
                y=neuron_synapses['y'],
                z=neuron_synapses['z'],
                mode='markers',
                name=f'{neuron_name}_synapses',
                visible=False,
                marker=dict(size=4, color='yellow', opacity=0.8),
                customdata=np.column_stack([
                    neuron_synapses['x'],
                    neuron_synapses['y'],
                    neuron_synapses['z'],
                    np.full(M, 'synapse'),
                    neuron_synapses['source'],
                    neuron_synapses['target'],
                    synapse_indices]),
                hovertemplate=(
                    'Synapse<br>%{customdata[4]} \u2192 %{customdata[5]}'
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
    snapshot_map = {'contact': {}, 'synapse': {}, 'overlap': {}}

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
            fname = f"contact_{idx_int}_segmented.png"
            if os.path.exists(os.path.join(em_snap_dir, fname)):
                snapshot_map['contact'][idx_int] = f"em_snaps/{fname}"
    print(f"[snapshots] Indexed {len(snapshot_map['contact'])} contacts "
          f"({len(indexed_clusters)} unique clusters) [file paths]")

    for idx in synapses['index'].unique():
        fname = f"synapse_{int(idx)}_segmented.png"
        if os.path.exists(os.path.join(em_snap_dir, fname)):
            snapshot_map['synapse'][int(idx)] = f"em_snaps/{fname}"
    print(f"[snapshots] Indexed {len(snapshot_map['synapse'])} synapses [file paths]")

    meta_file = os.path.join(results_dir, 'overlap_em_meta.json')
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as mf:
            overlap_meta = json.load(mf)
        for item in overlap_meta:
            idx = int(item['idx'])
            fname = f"overlap_{idx}_segmented.png"
            if os.path.exists(os.path.join(em_snap_dir, fname)):
                snapshot_map['overlap'][idx] = f"em_snaps/{fname}"
    print(f"[snapshots] Indexed {len(snapshot_map['overlap'])} overlaps [file paths]")

    return snapshot_map


# ── HTML generation ───────────────────────────────────────────────────

def generate_html(fig, contacts, synapses, trace_info,
                   overlap_pairs, overlap_pair_faces, em_snap_dir):
    """Generate HTML with all data embedded."""
    snapshot_mapping = index_em_snapshots(
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
            f' All overlap faces</label>'
            f'<label><input type="checkbox" id="curroverlapfaces_{neuron}">'
            f' Curr overlap faces</label>'
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
        '</div></div>'
    )

    checkboxes_html = global_controls + '\n'.join(neuron_checkboxes)

    # Build contact list with patch_area
    contact_list = contacts[[
        'idx', 'x', 'y', 'z', 'source', 'target', 'patch_area'
    ]].to_dict('records')
    synapse_list = synapses[['index', 'x', 'y', 'z', 'source', 'target']] \
        .rename(columns={'index': 'idx'}).to_dict('records')

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
    _ov_file_re = _re.compile(r'^overlap_(\d+)_z([+-])(\d+)\.png$')
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
            elif fname.startswith('overlap_') and fname.endswith('_segmented.png'):
                try:
                    ov_idx = int(fname.split('_')[1])
                    _ov_files_by_idx.setdefault(ov_idx, set()).add(0)
                except (ValueError, IndexError):
                    pass
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

    return html


# ── Main ──────────────────────────────────────────────────────────────

def main():
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


