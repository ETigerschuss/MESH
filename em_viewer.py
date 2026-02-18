"""
EM Viewer - Standalone Version
===============================

Complete self-contained EM viewer generator with integrated HTML template.
No external dependencies beyond standard Python libraries and Plotly.

Features:
- Left sidebar: Neuron controls
- Center: 3D mesh viewer with highlighting
- Right: EM snapshot viewer with resizable panels
- Highlights selected contact (solid red) or synapse (larger yellow) in 3D view

Usage:
    python em_viewer.py

Output:
    comprehensive_overlap_results/em_viewer.html
"""

import os
import json
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Tuple
from PIL import Image

# Neuron IDs - aligned to analysis script (restricted set + BIPS)
NEURON_IDS = {
    720575940618519710: 'MOT_L', 720575940630139386: 'MOT_R',
    720575940622361270: 'MOS_L', 720575940622168052: 'MOS_R',
    720575940626477498: 'VS1_L', 720575940619878961: 'VS1_R',
    720575940640722851: 'VS2_L', 720575940613126835: 'VS2_R',
    720575940622831740: 'VS3_L', 720575940641812699: 'VS3_R',
    720575940624273919: 'VS4_L', 720575940659799937: 'VS4_R',
    720575940628031249: 'HSN_L', 720575940615933919: 'HSN_R',
    720575940629153020: 'HSE_L', 720575940629148007: 'HSE_R',
    720575940622312965: 'HSS_L', 720575940628743496: 'HSS_R',
    720575940623618708: 'BIPS_L', 720575940622581173: 'BIPS_R',
}

# Group colors (single hue per group)
GROUP_COLOR = {
    'MOT': '#5E3C99',   # MOT
    'MOS': '#4D9221',   # MOS
    'VS':  '#D14900',   # VS
    'HS':  '#C51B7D',   # HS (HSN/HSE/HSS)
    'BIPS': '#000000',  # BIPS
}

def _group(name: str) -> str:
    if name.startswith('MOT'): return 'MOT'
    if name.startswith('MOS'): return 'MOS'
    if name.startswith('VS'):  return 'VS'
    if name.startswith('HS'):  return 'HS'
    if name.startswith('BIPS'): return 'BIPS'
    return 'OTHER'

def _build_color_map(neuron_ids: dict[int, str]) -> dict[str, str]:
    colors = {}
    for nm in neuron_ids.values():
        grp = _group(nm)
        colors[nm] = GROUP_COLOR.get(grp, '#888888')
    return colors

NEURON_COLORS = _build_color_map(NEURON_IDS)

# Default results dir: use latest comprehensive_overlap_results_* if present
def _default_results_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d))
                  and d.startswith('comprehensive_overlap_results_')]
    if candidates:
        return sorted(candidates)[-1]
    return 'comprehensive_overlap_results'

RESULTS_DIR = os.environ.get('MESH_RESULTS_DIR', _default_results_dir())


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def ensure_em_snapshots(results_dir: str, contacts: pd.DataFrame, synapses: pd.DataFrame,
                        size_px: int = 512, z_range: int = 20):
    """Ensure EM snapshots exist; if not, download center + Z-stack (±z_range slices) with segmentation overlay."""
    em_snap_dir = os.path.join(results_dir, 'em_snaps')
    os.makedirs(em_snap_dir, exist_ok=True)

    # Check if we already have z-stack images (not just center slices)
    existing = [f for f in os.listdir(em_snap_dir) if f.endswith('.png')]
    n_contacts = len(contacts)
    n_synapses = len(synapses)
    n_items = n_contacts + n_synapses
    expected_with_zstack = n_items * (2 * z_range + 1)  # center + ±z_range
    if len(existing) >= expected_with_zstack * 0.9:  # 90% threshold
        print(f"[snapshots] Found {len(existing)} images (expected ~{expected_with_zstack}), skipping download")
        return
    elif len(existing) > 0:
        print(f"[snapshots] Found {len(existing)} images but need ~{expected_with_zstack} for Z-stacks, downloading missing...")

    try:
        from cloudvolume import CloudVolume
    except ImportError:
        print("[snapshots] CloudVolume not installed; cannot download EM snapshots")
        return

    print(f"[snapshots] Downloading center + Z-stack (±{z_range}) to {em_snap_dir}")
    print(f"            {n_contacts} contacts + {n_synapses} synapses = {(n_contacts + n_synapses) * (2 * z_range + 1)} images")

    try:
        em_vol = CloudVolume('https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14',
                             mip=1, use_https=True, progress=False)
        seg_vol = CloudVolume('precomputed://gs://flywire_v141_m783',
                              mip=0, use_https=True, progress=False)
    except Exception as e:
        print(f"[snapshots] Failed to open CloudVolume: {e}")
        return

    name_to_id = {v: k for k, v in NEURON_IDS.items()}

    em_res_xy = float(em_vol.resolution[0])
    em_res_z = float(em_vol.resolution[2])
    seg_res_xy = float(seg_vol.resolution[0])
    seg_res_z = float(seg_vol.resolution[2])
    resolution_ratio = seg_res_xy / em_res_xy  # typically 2.0

    def grab(center_nm, source_name, target_name, kind: str, idx: int, z_offset: int = 0):
        """Download one EM slice with segmentation overlay at z_offset from center."""
        # Build output filename
        if z_offset == 0:
            out_path = os.path.join(em_snap_dir, f"{kind}_{idx}.png")
        else:
            sign = '+' if z_offset >= 0 else '-'
            out_path = os.path.join(em_snap_dir,
                                    f"{kind}_{idx}_z{sign}{abs(z_offset):03d}.png")
        if os.path.exists(out_path):
            return True  # already downloaded

        try:
            center_nm = np.array(center_nm, dtype=float)

            half_nm = (size_px * em_res_xy) / 2.0

            # EM voxel coordinates
            em_center_vox = center_nm / np.array(em_vol.resolution)
            x0 = int(em_center_vox[0] - size_px // 2)
            x1 = int(em_center_vox[0] + size_px // 2)
            y0 = int(em_center_vox[1] - size_px // 2)
            y1 = int(em_center_vox[1] + size_px // 2)
            z = int(em_center_vox[2]) + z_offset

            # Segmentation voxel coordinates
            seg_center_vox = center_nm / np.array(seg_vol.resolution)
            seg_half = int(size_px / resolution_ratio / 2)
            sx0 = int(seg_center_vox[0] - seg_half)
            sx1 = int(seg_center_vox[0] + seg_half)
            sy0 = int(seg_center_vox[1] - seg_half)
            sy1 = int(seg_center_vox[1] + seg_half)
            sz = int(seg_center_vox[2]) + z_offset

            em_raw = np.asarray(em_vol[x0:x1, y0:y1, z:z+1]).squeeze()
            seg_raw = np.asarray(seg_vol[sx0:sx1, sy0:sy1, sz:sz+1]).squeeze()

            if em_raw.ndim != 2 or seg_raw.ndim != 2:
                return False

            em_slice = np.clip(em_raw, 0, 255).astype(np.uint8)

            # Upsample segmentation to match EM resolution
            fx = max(1, int(np.ceil(em_slice.shape[0] / seg_raw.shape[0])))
            fy = max(1, int(np.ceil(em_slice.shape[1] / seg_raw.shape[1])))
            seg_up = np.repeat(np.repeat(seg_raw.astype(np.int64), fx, axis=0), fy, axis=1)
            # Pad if still smaller
            if seg_up.shape[0] < em_slice.shape[0] or seg_up.shape[1] < em_slice.shape[1]:
                seg_up = np.pad(seg_up,
                                ((0, max(0, em_slice.shape[0] - seg_up.shape[0])),
                                 (0, max(0, em_slice.shape[1] - seg_up.shape[1]))),
                                mode='edge')
            seg_up = seg_up[:em_slice.shape[0], :em_slice.shape[1]]

            # Build RGB image
            img = np.stack([em_slice, em_slice, em_slice], axis=-1).astype(np.uint8)

            # Color overlay
            sid = name_to_id.get(source_name)
            tid = name_to_id.get(target_name)
            if sid is not None:
                mask = seg_up == sid
                c = np.array(_hex_to_rgb(NEURON_COLORS.get(source_name, '#ff0000')), dtype=np.uint8)
                img[mask] = (img[mask].astype(float) * 0.65 + c.astype(float) * 0.35).astype(np.uint8)
            if tid is not None:
                mask = seg_up == tid
                c = np.array(_hex_to_rgb(NEURON_COLORS.get(target_name, '#00ff00')), dtype=np.uint8)
                img[mask] = (img[mask].astype(float) * 0.65 + c.astype(float) * 0.35).astype(np.uint8)

            Image.fromarray(img).save(out_path)
            return True

        except Exception as e:
            if z_offset == 0:
                print(f"[snapshots] Failed {kind} {idx}: {e}")
            return False

    # Download contacts
    total_ok = 0
    for i, (_, row) in enumerate(contacts.iterrows()):
        center = (float(row['x']), float(row['y']), float(row['z']))
        src, tgt = row['source'], row['target']
        cidx = int(row['idx'])
        for zo in range(-z_range, z_range + 1):
            if grab(center, src, tgt, 'contact', cidx, zo):
                total_ok += 1
        if (i + 1) % 10 == 0:
            print(f"  contacts: {i+1}/{len(contacts)}")
    print(f"[snapshots] Downloaded {total_ok} contact images")

    # Download synapses
    syn_col = 'idx' if 'idx' in synapses.columns else 'index'
    total_ok = 0
    for i, (_, row) in enumerate(synapses.iterrows()):
        center = (float(row['x']), float(row['y']), float(row['z']))
        src, tgt = row['source'], row['target']
        sidx = int(row[syn_col])
        for zo in range(-z_range, z_range + 1):
            if grab(center, src, tgt, 'synapse', sidx, zo):
                total_ok += 1
        if (i + 1) % 10 == 0:
            print(f"  synapses: {i+1}/{len(synapses)}")
    print(f"[snapshots] Downloaded {total_ok} synapse images")

# HTML Template (integrated)
HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>EM Overlay Viewer</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: #1a1a1a;
            color: white;
            overflow: hidden;
        }
        .container {
            display: flex;
            height: 100vh;
            width: 100vw;
        }
        
        /* Resizable panels */
        .sidebar {
            width: 150px;
            min-width: 100px;
            max-width: 300px;
            background: #2a2a2a;
            padding: 10px;
            overflow-y: auto;
            border-right: 1px solid #444;
        }
        
        .resizer {
            width: 5px;
            background: #444;
            cursor: col-resize;
            position: relative;
        }
        
        .resizer:hover {
            background: #666;
        }
        
        .mesh-container {
            flex: 1;
            min-width: 300px;
            display: flex;
            flex-direction: column;
            background: #1a1a1a;
        }
        
        .em-panel {
            width: 600px;
            min-width: 400px;
            max-width: 1000px;
            background: #2a2a2a;
            display: flex;
            flex-direction: column;
            border-left: 1px solid #444;
        }
        .sidebar h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #FFD400;
            border-bottom: 1px solid #444;
            padding-bottom: 5px;
        }
        .neuron-group {
            margin-bottom: 8px;
            padding: 6px;
            background: #1f1f1f;
            border-radius: 4px;
        }
        .neuron-name {
            font-weight: bold;
            color: #bbb;
            font-size: 12px;
            display: block;
            margin-bottom: 4px;
        }
        .neuron-controls {
            display: flex;
            flex-direction: column;
            gap: 2px;
            font-size: 11px;
        }
        .neuron-controls label {
            display: flex;
            align-items: center;
            gap: 4px;
            cursor: pointer;
            padding: 2px 0;
        }
        .neuron-controls input[type="checkbox"] {
            cursor: pointer;
        }
        
        /* Center: 3D Mesh */
        .mesh-container {
            background: #0a0a0a;
            display: flex;
            flex-direction: column;
        }
        .controls {
            background: #222;
            padding: 8px 15px;
            border-bottom: 1px solid #444;
        }
        .plot-container {
            flex: 1;
            position: relative;
        }
        
        /* Right: EM Panel */
        .em-panel {
            background: #2a2a2a;
            display: flex;
            flex-direction: column;
            border-left: 1px solid #444;
        }
        .em-header {
            background: #1f1f1f;
            padding: 10px 15px;
            border-bottom: 1px solid #444;
        }
        #emTitle {
            font-weight: bold;
            color: #FFD400;
            font-size: 13px;
            display: block;
            margin-bottom: 4px;
        }
        #emLocation {
            color: #888;
            font-size: 11px;
        }
        .em-display {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #0a0a0a;
            padding: 10px;
            min-height: 0;
            overflow: hidden;
        }
        #emImage {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        #emPlaceholder {
            color: #555;
            font-size: 13px;
            text-align: center;
        }
        .em-controls {
            background: #1f1f1f;
            border-top: 1px solid #444;
        }
        .control-row {
            padding: 8px 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid #333;
        }
        .control-row:last-child {
            border-bottom: none;
        }
        .control-row button {
            padding: 6px 12px;
            cursor: pointer;
            background: #444;
            color: white;
            border: 1px solid #666;
            border-radius: 3px;
            font-size: 11px;
            white-space: nowrap;
        }
        .control-row button:hover {
            background: #555;
        }
        .control-row button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .control-row input[type="range"] {
            flex: 1;
            min-width: 0;
        }
        .control-label {
            font-size: 11px;
            color: #999;
            min-width: 50px;
        }
        .control-value {
            font-size: 11px;
            color: #FFD400;
            font-weight: bold;
            text-align: center;
        }
        .info-text {
            color: #999;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Left: Neuron Controls -->
        <div class="sidebar">
            <h3>Neurons</h3>
            {CHECKBOXES_HTML}
        </div>
        
        <!-- Resizer 1: Between sidebar and mesh -->
        <div class="resizer" id="resizer1"></div>

        <!-- Center: 3D Mesh Viewer -->
        <div class="mesh-container">
            <div class="controls">
                <span id="infoText" class="info-text">Select neurons, then click contact (red circle) or synapse (yellow) points</span>
            </div>
            <div class="plot-container">
                {PLOT_DIV}
            </div>
        </div>
        
        <!-- Resizer 2: Between mesh and EM panel -->
        <div class="resizer" id="resizer2"></div>

        <!-- Right: EM Snapshot Viewer -->
        <div class="em-panel">
            <div class="em-header">
                <span id="emTitle">EM Snapshot</span>
                <span id="emLocation"></span>
            </div>
            <div class="em-display">
                <img id="emImage" style="display:none;" alt="EM Snapshot">
                <span id="emPlaceholder">Click a contact or synapse to view EM</span>
            </div>
            <div class="em-controls">
                <!-- Item Navigation -->
                <div class="control-row">
                    <button id="btnPrevItem">&lt; Prev</button>
                    <span id="itemInfo" class="control-value" style="flex: 1;"></span>
                    <button id="btnNextItem">Next &gt;</button>
                </div>
                <!-- Z-Stack Navigation -->
                <div class="control-row">
                    <button id="btnPrevZ">&lt;</button>
                    <span class="control-label">Z-Stack:</span>
                    <input type="range" id="zSlider" min="-20" max="20" value="0" step="1">
                    <span id="zValue" class="control-value" style="min-width: 80px;">0</span>
                    <button id="btnNextZ">&gt;</button>
                </div>
                <div class="control-row" style="justify-content: center; padding: 4px;">
                    <span id="zNote" style="color: #888; font-size: 10px;">±800nm depth range</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data embedded from Python
        const snapshotMap = {SNAPSHOT_JSON};
        const neuronNames = {NEURON_NAMES_JSON};
        const traceInfo = {TRACE_INFO_JSON};
        const contactList = {CONTACT_LIST_JSON};
        const synapseList = {SYNAPSE_LIST_JSON};
        const brainTraceIndices = {BRAIN_TRACE_INDICES_JSON};
        
        // DOM elements
        const plotDiv = document.getElementById('plotly3d');
        const emImage = document.getElementById('emImage');
        const emPlaceholder = document.getElementById('emPlaceholder');
        const emTitle = document.getElementById('emTitle');
        const emLocation = document.getElementById('emLocation');
        const infoText = document.getElementById('infoText');
        const itemInfo = document.getElementById('itemInfo');
        const zSlider = document.getElementById('zSlider');
        const zValue = document.getElementById('zValue');
        const zNote = document.getElementById('zNote');
        const btnPrevZ = document.getElementById('btnPrevZ');
        const btnNextZ = document.getElementById('btnNextZ');
        const btnPrevItem = document.getElementById('btnPrevItem');
        const btnNextItem = document.getElementById('btnNextItem');
        
        // Resizable panels
        const sidebar = document.querySelector('.sidebar');
        const meshContainer = document.querySelector('.mesh-container');
        const emPanel = document.querySelector('.em-panel');
        const resizer1 = document.getElementById('resizer1');
        const resizer2 = document.getElementById('resizer2');
        
        // State
        let currentKind = null;
        let currentIdx = null;
        let currentZ = 0;
        let currentList = [];
        let currentListIndex = -1;
        let isUpdatingHighlight = false;
        
        // Handle broken image loads (e.g. missing Z-stack files)
        emImage.addEventListener('error', function() {
            emImage.style.display = 'none';
            emPlaceholder.style.display = 'block';
            emPlaceholder.textContent = `Image file not found for ${currentKind} ${currentIdx} at z=${currentZ}`;
        });
        
        // Track mouse down for drag detection
        let mouseDownX = 0;
        let mouseDownY = 0;
        let mouseDownTime = 0;
        
        plotDiv.addEventListener('mousedown', function(e) {
            mouseDownX = e.clientX;
            mouseDownY = e.clientY;
            mouseDownTime = Date.now();
        });
        
        // Get visible neurons
        function getVisibleNeurons(type) {
            const visible = [];
            neuronNames.forEach(neuron => {
                const cb = document.getElementById(`${type}_${neuron}`);
                if (cb && cb.checked) {
                    visible.push(neuron);
                }
            });
            return visible;
        }
        
        // Filter items by visible neurons
        function getVisibleItems(kind) {
            const visibleNeurons = getVisibleNeurons(kind + 's');
            const itemList = kind === 'contact' ? contactList : synapseList;
            
            return itemList
                .filter(item => visibleNeurons.includes(item.source) || visibleNeurons.includes(item.target))
                .map(item => item.idx)
                .sort((a, b) => a - b);
        }
        
        // Checkbox handlers
        neuronNames.forEach(neuron => {
            ['mesh', 'contacts', 'synapses'].forEach(type => {
                const cb = document.getElementById(`${type}_${neuron}`);
                if (cb) {
                    cb.addEventListener('change', function() {
                        const traceIdx = traceInfo[`${neuron}_${type}`];
                        if (traceIdx !== undefined) {
                            Plotly.restyle(plotDiv, {'visible': this.checked}, [traceIdx]);
                        }
                    });
                }
            });
        });
        
        // Brain areas toggle and opacity
        (function() {
            const brainCb = document.getElementById('brainToggle');
            const brainSlider = document.getElementById('brainOpacity');
            const brainVal = document.getElementById('brainOpacityVal');
            if (brainCb && brainTraceIndices.length > 0) {
                brainCb.addEventListener('change', function() {
                    const vis = this.checked;
                    brainSlider.disabled = !vis;
                    Plotly.restyle(plotDiv, {'visible': vis}, brainTraceIndices);
                });
                if (brainSlider) {
                    brainSlider.addEventListener('input', function() {
                        const opacity = parseInt(this.value) / 100;
                        if (brainVal) brainVal.textContent = opacity.toFixed(2);
                        Plotly.restyle(plotDiv, {'opacity': opacity}, brainTraceIndices);
                    });
                }
            }
        })();
        
        // Click handler for 3D plot
        plotDiv.on('plotly_click', function(data) {
            if (isUpdatingHighlight) return;
            if (!data.points || data.points.length === 0) return;
            
            // Detect drag vs click
            if (data.event && mouseDownX !== undefined && mouseDownY !== undefined) {
                const mouseUpTime = Date.now();
                const distance = Math.sqrt(
                    Math.pow(data.event.clientX - mouseDownX, 2) + 
                    Math.pow(data.event.clientY - mouseDownY, 2)
                );
                
                if (distance > 5) return;  // Was a drag
            }
            
            const point = data.points[0];
            const customdata = point.customdata;
            if (!customdata) return;
            
            const [x, y, z, kind, source, target, idx] = customdata;
            
            if (kind === 'mesh') {
                if (currentKind === null) {
                    emTitle.textContent = `Mesh: ${source}`;
                    emLocation.textContent = `(${Math.round(x)}, ${Math.round(y)}, ${Math.round(z)})`;
                    emPlaceholder.textContent = 'Click a contact (red) or synapse (yellow) point';
                }
                return;
            }
            
            selectItem(kind, idx, x, y, z, source, target);
        });
        
        function selectItem(kind, idx, x, y, z, source, target) {
            currentKind = kind;
            currentIdx = idx;
            currentZ = 0;
            zSlider.value = 0;
            
            currentList = getVisibleItems(kind);
            currentListIndex = currentList.indexOf(idx);
            
            emTitle.textContent = `${source} → ${target}`;
            emLocation.textContent = `${kind.charAt(0).toUpperCase() + kind.slice(1)} #${idx} at (${Math.round(x)}, ${Math.round(y)}, ${Math.round(z)})`;
            itemInfo.textContent = `${currentListIndex + 1}/${currentList.length}`;
            
            loadImage(kind, idx, 0);
            highlightPoint(kind, idx);
        }
        
        function loadImage(kind, idx, zOffset) {
            if (zOffset === 0) {
                const imgData = snapshotMap[kind]?.[idx];
                if (!imgData) {
                    emImage.style.display = 'none';
                    emPlaceholder.style.display = 'block';
                    emPlaceholder.textContent = `No snapshot for ${kind} ${idx}`;
                    return;
                }
                emImage.src = imgData;
                emImage.style.display = 'block';
                emPlaceholder.style.display = 'none';
                updateZValue(0);
            } else {
                loadZStackImage(kind, idx, zOffset);
            }
        }
        
        function loadZStackImage(kind, idx, zOffset) {
            const sign = zOffset >= 0 ? '+' : '-';
            const pad = String(Math.abs(zOffset)).padStart(3, '0');
            const key = `${idx}_z${sign}${pad}`;
            const imgData = snapshotMap[kind]?.[key];
            
            if (imgData) {
                emImage.src = imgData;
                emImage.style.display = 'block';
                emPlaceholder.style.display = 'none';
                updateZValue(zOffset);
            } else {
                // No z-stack image available - show placeholder
                emImage.style.display = 'none';
                emPlaceholder.style.display = 'block';
                emPlaceholder.textContent = `No Z-stack image for ${kind} ${idx} at z=${zOffset}`;
            }
        }
        
        function updateZValue(zOffset) {
            currentZ = zOffset;
            const nmOffset = zOffset * 40;
            if (zOffset === 0) {
                zValue.textContent = '0 (center)';
                zNote.textContent = 'Center slice (segmented)';
                zNote.style.color = '#0a0';
            } else {
                const sign = zOffset > 0 ? '+' : '';
                zValue.textContent = `${sign}${zOffset} (${sign}${nmOffset}nm)`;
                zNote.textContent = `Depth offset: ${sign}${nmOffset} nm`;
                zNote.style.color = '#888';
            }
        }
        
        function highlightPoint(kind, idx) {
            if (isUpdatingHighlight) return;
            
            isUpdatingHighlight = true;
            
            const itemList = kind === 'contact' ? contactList : synapseList;
            const item = itemList.find(i => i.idx === idx);
            
            if (!item) {
                isUpdatingHighlight = false;
                return;
            }
            
            const neuronName = item.source;
            const highlightTraceName = `${neuronName}_${kind}s_highlight`;
            const highlightTraceIdx = traceInfo[highlightTraceName];
            
            if (highlightTraceIdx === undefined) {
                isUpdatingHighlight = false;
                return;
            }
            
            const updates = {x: [], y: [], z: [], visible: []};
            const indices = [];
            
            for (let traceName in traceInfo) {
                if (traceName.includes('_highlight')) {
                    const traceIdx = traceInfo[traceName];
                    indices.push(traceIdx);
                    
                    if (traceIdx === highlightTraceIdx) {
                        updates.x.push([item.x]);
                        updates.y.push([item.y]);
                        updates.z.push([item.z]);
                        updates.visible.push(true);
                    } else {
                        updates.x.push([]);
                        updates.y.push([]);
                        updates.z.push([]);
                        updates.visible.push(false);
                    }
                }
            }
            
            Plotly.restyle(plotDiv, updates, indices).then(() => {
                isUpdatingHighlight = false;
            }).catch(err => {
                isUpdatingHighlight = false;
            });
        }
        
        // Z-stack navigation
        zSlider.addEventListener('input', function() {
            if (currentKind && currentIdx !== null) {
                loadImage(currentKind, currentIdx, parseInt(this.value));
            }
        });
        
        btnPrevZ.addEventListener('click', function() {
            if (currentKind && currentIdx !== null) {
                const newZ = Math.max(-20, currentZ - 1);
                zSlider.value = newZ;
                loadImage(currentKind, currentIdx, newZ);
            }
        });
        
        btnNextZ.addEventListener('click', function() {
            if (currentKind && currentIdx !== null) {
                const newZ = Math.min(20, currentZ + 1);
                zSlider.value = newZ;
                loadImage(currentKind, currentIdx, newZ);
            }
        });
        
        // Item navigation helper
        function navigateToItem(newListIndex) {
            const newIdx = currentList[newListIndex];
            const itemList = currentKind === 'contact' ? contactList : synapseList;
            const item = itemList.find(i => i.idx === newIdx);
            if (!item) return;

            currentListIndex = newListIndex;
            currentIdx = newIdx;
            currentZ = 0;
            zSlider.value = 0;

            const label = currentKind.charAt(0).toUpperCase() + currentKind.slice(1);
            emTitle.textContent = `${item.source} → ${item.target}`;
            emLocation.textContent = `${label} #${newIdx} at (${Math.round(item.x)}, ${Math.round(item.y)}, ${Math.round(item.z)})`;
            itemInfo.textContent = `${newListIndex + 1}/${currentList.length}`;

            loadImage(currentKind, newIdx, 0);
            highlightPoint(currentKind, newIdx);
        }

        btnPrevItem.addEventListener('click', function() {
            if (currentKind && currentListIndex > 0) {
                navigateToItem(currentListIndex - 1);
            }
        });

        btnNextItem.addEventListener('click', function() {
            if (currentKind && currentListIndex < currentList.length - 1) {
                navigateToItem(currentListIndex + 1);
            }
        });
        
        // Resizable panels
        let isResizing = false;
        let currentResizer = null;
        
        function initResize(resizer, leftPanel, rightPanel) {
            resizer.addEventListener('mousedown', function(e) {
                isResizing = true;
                currentResizer = { resizer, leftPanel, rightPanel };
                document.body.style.cursor = 'col-resize';
                e.preventDefault();
            });
        }
        
        document.addEventListener('mousemove', function(e) {
            if (!isResizing || !currentResizer) return;
            
            const container = document.querySelector('.container');
            const containerRect = container.getBoundingClientRect();
            const { leftPanel, rightPanel } = currentResizer;
            
            if (leftPanel === sidebar) {
                const newWidth = e.clientX - containerRect.left;
                const minWidth = parseInt(getComputedStyle(sidebar).minWidth);
                const maxWidth = parseInt(getComputedStyle(sidebar).maxWidth);
                
                if (newWidth >= minWidth && newWidth <= maxWidth) {
                    sidebar.style.width = newWidth + 'px';
                }
            } else if (rightPanel === emPanel) {
                const newWidth = containerRect.right - e.clientX;
                const minWidth = parseInt(getComputedStyle(emPanel).minWidth);
                const maxWidth = parseInt(getComputedStyle(emPanel).maxWidth);
                
                if (newWidth >= minWidth && newWidth <= maxWidth) {
                    emPanel.style.width = newWidth + 'px';
                }
            }
            
            if (plotDiv) {
                Plotly.Plots.resize(plotDiv);
            }
        });
        
        document.addEventListener('mouseup', function() {
            if (isResizing) {
                isResizing = false;
                currentResizer = null;
                document.body.style.cursor = 'default';
            }
        });
        
        initResize(resizer1, sidebar, meshContainer);
        initResize(resizer2, meshContainer, emPanel);
        
        itemInfo.textContent = '';
        updateZValue(0);
    </script>
</body>
</html>"""


def load_contacts(results_dir):
    """Load all contact patches from CSV"""
    csv_file = os.path.join(results_dir, 'all_results_combined.csv')
    df = pd.read_csv(csv_file)
    df = df[df['Has_Contact'] == True]
    
    all_patches = []
    patch_idx = 0
    
    for _, row in df.iterrows():
        source = row['Source_Neuron']
        target = row['Target_Neuron']
        
        for patch_num in range(1, 7):
            x_col = f'Top{patch_num}_Patch_Centroid_X'
            y_col = f'Top{patch_num}_Patch_Centroid_Y'
            z_col = f'Top{patch_num}_Patch_Centroid_Z'
            
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                if not pd.isna(row[x_col]):
                    all_patches.append({
                        'idx': patch_idx,
                        'x': row[x_col],
                        'y': row[y_col],
                        'z': row[z_col],
                        'source': source,
                        'target': target,
                        'patch_num': patch_num
                    })
                    patch_idx += 1
    
    result_df = pd.DataFrame(all_patches)
    print(f"[contacts] Loaded {len(result_df)} contact patches from {len(df)} neuron pairs")
    print(f"           (avg {len(result_df)/len(df):.1f} patches per pair)")
    return result_df


def load_synapses(results_dir):
    """Load MOT/MOS synapses"""
    csv_file = os.path.join(results_dir, 'synapses.csv')
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=['x', 'y', 'z'])
    
    mot_mos_neurons = ['MOT_L', 'MOT_R', 'MOS_L', 'MOS_R']
    
    if 'pre_type' in df.columns:
        df['source'] = df['pre_type']
        df['target'] = df['post_type']
    else:
        df['source'] = df['pre']
        df['target'] = df['post']
    
    df = df[df['source'].isin(mot_mos_neurons) | df['target'].isin(mot_mos_neurons)]
    
    print(f"[synapses] Loaded {len(df)} MOT/MOS synapses")
    return df[['x', 'y', 'z', 'source', 'target']].reset_index()


def load_mesh_pointcloud(neuron_name, mesh_dir, max_points=10000):
    """Load mesh as point cloud"""
    neuron_id = {v: k for k, v in NEURON_IDS.items()}[neuron_name]
    mesh_file = os.path.join(mesh_dir, f"{neuron_id}.obj")
    
    if not os.path.exists(mesh_file):
        return None, None, None
    
    vertices = []
    with open(mesh_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
    
    vertices = np.array(vertices)
    if len(vertices) > max_points:
        stride = len(vertices) // max_points
        vertices = vertices[::stride][:max_points]
    
    return vertices[:, 0], vertices[:, 1], vertices[:, 2]


def _load_brain_neuropil_traces():
    """Load FlyWire brain neuropil meshes and return Plotly Mesh3d traces (hidden by default).
    Uses fafbseg.flywire.get_neuropil_volumes for JFRC2 neuropils mapped to FAFB14.1 space.
    Returns list of (trace, neuropil_name) tuples, or empty list on failure."""
    try:
        import fafbseg.flywire as fw
    except ImportError:
        print("[brain] fafbseg not installed, skipping brain neuropils")
        return []

    # Get list of available neuropils
    try:
        available = fw.get_neuropil_volumes(None)
        if not available:
            print("[brain] No neuropil volumes available")
            return []
        neuropil_names = sorted(available)
    except Exception as e:
        print(f"[brain] Could not list neuropils: {e}")
        return []

    print(f"[brain] Loading {len(neuropil_names)} neuropil volumes...")
    traces = []
    for name in neuropil_names:
        try:
            vol = fw.get_neuropil_volumes(name)
            if vol is None or not hasattr(vol, 'vertices') or len(vol.vertices) == 0:
                continue
            verts = np.array(vol.vertices)
            faces = np.array(vol.faces)
            traces.append((go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='#888888',
                opacity=0.08,
                name=f'brain_{name}',
                visible=False,
                hoverinfo='name',
                showlegend=False,
                flatshading=True,
                lighting=dict(ambient=0.8, diffuse=0.2),
            ), name))
        except Exception as e:
            print(f"[brain] Failed to load {name}: {e}")
            continue
    print(f"[brain] Loaded {len(traces)} neuropil regions")
    return traces


def build_figure(mesh_dir):
    """Build the 3D Plotly figure with highlighting capability"""
    contacts = load_contacts(RESULTS_DIR)
    synapses = load_synapses(RESULTS_DIR)
    ensure_em_snapshots(RESULTS_DIR, contacts, synapses)
    
    traces = []
    trace_info = {}
    
    print("[meshes] Loading neuron data...")
    for neuron_name in sorted(NEURON_IDS.values()):
        color = NEURON_COLORS.get(neuron_name, '#888888')
        
        # 1. Mesh trace
        x, y, z = load_mesh_pointcloud(neuron_name, mesh_dir)
        if x is not None:
            trace_idx = len(traces)
            trace_info[f"{neuron_name}_mesh"] = trace_idx
            traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                name=f"{neuron_name}_mesh",
                marker=dict(size=2, color=color, opacity=0.3),
                visible=False,
                hovertemplate=f'{neuron_name}<br>(%{{x}}, %{{y}}, %{{z}})<extra></extra>',
                customdata=np.column_stack([x, y, z, 
                    np.full(len(x), 'mesh'), 
                    np.full(len(x), neuron_name),
                    np.full(len(x), neuron_name),
                    np.full(len(x), -1)]),
                legendgroup=neuron_name
            ))
            print(f"  {neuron_name}: mesh ({len(x)} points)")
        
        # 2. Contact traces (normal + highlighted)
        neuron_contacts = contacts[
            (contacts['source'] == neuron_name) | (contacts['target'] == neuron_name)
        ]
        if not neuron_contacts.empty:
            N = len(neuron_contacts)
            
            # Normal contact trace
            trace_idx = len(traces)
            trace_info[f"{neuron_name}_contacts"] = trace_idx
            traces.append(go.Scatter3d(
                x=neuron_contacts['x'], y=neuron_contacts['y'], z=neuron_contacts['z'],
                mode='markers',
                name=f'{neuron_name}_contacts',
                visible=False,
                marker=dict(size=7, color='white', opacity=0.98, 
                           symbol='circle-open', line=dict(color='red', width=3)),
                customdata=np.column_stack([
                    neuron_contacts['x'], neuron_contacts['y'], neuron_contacts['z'],
                    np.full(N, 'contact'),
                    neuron_contacts['source'],
                    neuron_contacts['target'],
                    neuron_contacts['idx'],
                    neuron_contacts['patch_num']]),
                hovertemplate='Contact #%{customdata[6]}<br>%{customdata[4]} → %{customdata[5]}<extra></extra>',
                legendgroup=neuron_name
            ))
            
            # Highlighted contact trace
            trace_idx = len(traces)
            trace_info[f"{neuron_name}_contacts_highlight"] = trace_idx
            traces.append(go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                name=f'{neuron_name}_contacts_highlight',
                visible=False,
                marker=dict(size=12, color='red', opacity=1.0, symbol='circle'),
                hovertemplate='SELECTED Contact<extra></extra>',
                legendgroup=neuron_name,
                showlegend=False
            ))
            print(f"  {neuron_name}: {N} contacts")
        
        # 3. Synapse traces (normal + highlighted)
        neuron_synapses = synapses[
            (synapses['source'] == neuron_name) | (synapses['target'] == neuron_name)
        ]
        if not neuron_synapses.empty:
            M = len(neuron_synapses)
            synapse_indices = neuron_synapses['index'].values
            
            # Normal synapse trace
            trace_idx = len(traces)
            trace_info[f"{neuron_name}_synapses"] = trace_idx
            traces.append(go.Scatter3d(
                x=neuron_synapses['x'], y=neuron_synapses['y'], z=neuron_synapses['z'],
                mode='markers',
                name=f'{neuron_name}_synapses',
                visible=False,
                marker=dict(size=4, color='yellow', opacity=0.8),
                customdata=np.column_stack([
                    neuron_synapses['x'], neuron_synapses['y'], neuron_synapses['z'],
                    np.full(M, 'synapse'),
                    neuron_synapses['source'],
                    neuron_synapses['target'],
                    synapse_indices]),
                hovertemplate='Synapse<br>%{customdata[4]} → %{customdata[5]}<extra></extra>',
                legendgroup=neuron_name
            ))
            
            # Highlighted synapse trace
            trace_idx = len(traces)
            trace_info[f"{neuron_name}_synapses_highlight"] = trace_idx
            traces.append(go.Scatter3d(
                x=[], y=[], z=[],
                mode='markers',
                name=f'{neuron_name}_synapses_highlight',
                visible=False,
                marker=dict(size=12, color='yellow', opacity=1.0, symbol='circle'),
                hovertemplate='SELECTED Synapse<extra></extra>',
                legendgroup=neuron_name,
                showlegend=False
            ))
            print(f"  {neuron_name}: {M} synapses")
    
    # Load brain neuropil meshes
    brain_traces_info = []
    brain_neuropils = _load_brain_neuropil_traces()
    for brain_trace, np_name in brain_neuropils:
        idx = len(traces)
        trace_info[f"brain_{np_name}"] = idx
        brain_traces_info.append(idx)
        traces.append(brain_trace)
    
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
    
    return fig, contacts, synapses, trace_info, brain_traces_info


def index_em_snapshots(em_snap_dir, contacts, synapses, z_range: int = 20):
    """Index EM snapshots: embed CENTER slices as base64, reference Z-stack by filename.

    Center images are embedded so the viewer works as a standalone HTML file.
    Z-stack images are referenced as relative file paths to keep HTML size manageable
    (~42 MB for centers vs ~1.7 GB if all Z-stack were embedded).

    Returns dict: snapshot_map[kind][idx] = base64 data URI for center,
                  snapshot_map[kind][f"{idx}_z+003"] = relative file path for z-offsets.
    """
    snapshot_map = {'contact': {}, 'synapse': {}}

    if not os.path.isdir(em_snap_dir):
        print(f"[snapshots] Directory not found: {em_snap_dir} (no EM images will display)")
        return snapshot_map

    # Relative path from the HTML file to the em_snaps directory
    rel_snap_dir = os.path.basename(em_snap_dir)

    def _embed(filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
        return None

    def _file_ref(filepath, filename):
        """Return relative file path if file exists, else None."""
        if os.path.exists(filepath):
            return f"{rel_snap_dir}/{filename}"
        return None

    # Index contacts
    for idx in contacts['idx'].unique():
        idx = int(idx)
        # Center: embed as base64
        for candidate in [f"contact_{idx}_segmented.png", f"contact_{idx}.png"]:
            b64 = _embed(os.path.join(em_snap_dir, candidate))
            if b64:
                snapshot_map['contact'][idx] = b64
                break
        # Z-stack: reference by file path (not embedded)
        for zo in range(-z_range, z_range + 1):
            if zo == 0:
                continue
            sign = '+' if zo >= 0 else '-'
            key = f"{idx}_z{sign}{abs(zo):03d}"
            for candidate in [f"contact_{idx}_z{sign}{abs(zo):03d}_segmented.png",
                              f"contact_{idx}_z{sign}{abs(zo):03d}.png"]:
                ref = _file_ref(os.path.join(em_snap_dir, candidate), candidate)
                if ref:
                    snapshot_map['contact'][key] = ref
                    break

    n_center = sum(1 for k in snapshot_map['contact'] if isinstance(k, int))
    n_zstack = len(snapshot_map['contact']) - n_center
    print(f"[snapshots] Indexed {n_center} contact centers (embedded) + {n_zstack} Z-stack refs")

    # Index synapses
    syn_col = 'idx' if 'idx' in synapses.columns else 'index'
    for idx in synapses[syn_col].unique():
        idx = int(idx)
        for candidate in [f"synapse_{idx}_segmented.png", f"synapse_{idx}.png"]:
            b64 = _embed(os.path.join(em_snap_dir, candidate))
            if b64:
                snapshot_map['synapse'][idx] = b64
                break
        for zo in range(-z_range, z_range + 1):
            if zo == 0:
                continue
            sign = '+' if zo >= 0 else '-'
            key = f"{idx}_z{sign}{abs(zo):03d}"
            for candidate in [f"synapse_{idx}_z{sign}{abs(zo):03d}_segmented.png",
                              f"synapse_{idx}_z{sign}{abs(zo):03d}.png"]:
                ref = _file_ref(os.path.join(em_snap_dir, candidate), candidate)
                if ref:
                    snapshot_map['synapse'][key] = ref
                    break

    n_center = sum(1 for k in snapshot_map['synapse'] if isinstance(k, int))
    n_zstack = len(snapshot_map['synapse']) - n_center
    print(f"[snapshots] Indexed {n_center} synapse centers (embedded) + {n_zstack} Z-stack refs")

    return snapshot_map
def generate_html(fig, contacts, synapses, trace_info, em_snap_dir, brain_trace_indices=None):
    """Generate improved HTML with integrated template"""
    if brain_trace_indices is None:
        brain_trace_indices = []
    snapshot_mapping = index_em_snapshots(em_snap_dir, contacts, synapses)
    
    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        div_id='plotly3d',
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    # Create neuron checkboxes
    neuron_list = sorted(NEURON_IDS.values())
    neuron_checkboxes = []
    for neuron in neuron_list:
        neuron_checkboxes.append(
            f'<div class="neuron-group">'
            f'<span class="neuron-name">{neuron}</span>'
            f'<div class="neuron-controls">'
            f'<label><input type="checkbox" id="mesh_{neuron}"> Mesh</label>'
            f'<label><input type="checkbox" id="contacts_{neuron}"> Contacts</label>'
            f'<label><input type="checkbox" id="synapses_{neuron}"> Synapses</label>'
            f'</div></div>'
        )
    
    checkboxes_html = '\n'.join(neuron_checkboxes)
    
    # Add brain areas controls if brain traces are available
    if brain_trace_indices:
        checkboxes_html += '''
<div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid #444;">
  <h3 style="margin: 0 0 6px 0; font-size: 13px; color: #aaa;">Brain Areas</h3>
  <label style="display:flex; align-items:center; gap:4px; font-size:11px; cursor:pointer;">
    <input type="checkbox" id="brainToggle"> Show brain
  </label>
  <div style="margin-top:6px; font-size:11px;">
    <label style="color:#999;">Opacity</label>
    <input type="range" id="brainOpacity" min="0" max="40" value="8" step="1" 
           style="width:100%; margin-top:2px;" disabled>
    <span id="brainOpacityVal" style="color:#FFD400; font-size:10px;">0.08</span>
  </div>
</div>'''
    
    # Prepare data for JavaScript
    snapshot_json = json.dumps(snapshot_mapping)
    neuron_names_json = json.dumps(neuron_list)
    trace_info_json = json.dumps(trace_info)
    
    # Build contact and synapse lists for navigation
    contact_list = contacts[['idx', 'x', 'y', 'z', 'source', 'target']].to_dict('records')
    synapse_list = synapses[['index', 'x', 'y', 'z', 'source', 'target']].rename(
        columns={'index': 'idx'}).to_dict('records')
    
    contact_list_json = json.dumps(contact_list)
    synapse_list_json = json.dumps(synapse_list)
    brain_indices_json = json.dumps(brain_trace_indices)
    
    # Use integrated template
    html_content = HTML_TEMPLATE
    
    # Replace placeholders
    html_content = html_content.replace('{CHECKBOXES_HTML}', checkboxes_html)
    html_content = html_content.replace('{PLOT_DIV}', plot_div)
    html_content = html_content.replace('{SNAPSHOT_JSON}', snapshot_json)
    html_content = html_content.replace('{NEURON_NAMES_JSON}', neuron_names_json)
    html_content = html_content.replace('{TRACE_INFO_JSON}', trace_info_json)
    html_content = html_content.replace('{CONTACT_LIST_JSON}', contact_list_json)
    html_content = html_content.replace('{SYNAPSE_LIST_JSON}', synapse_list_json)
    html_content = html_content.replace('{BRAIN_TRACE_INDICES_JSON}', brain_indices_json)
    
    return html_content


def main():
    print("="*70)
    print("EM Viewer - Generating Interactive Viewer")
    print("="*70)
    
    mesh_dir = os.path.join(RESULTS_DIR, 'neuron_meshes')
    em_snap_dir = os.path.join(RESULTS_DIR, 'em_snaps')
    
    print("\n[1/3] Building 3D figure with highlighting...")
    fig, contacts, synapses, trace_info, brain_indices = build_figure(mesh_dir)
    
    print("\n[2/3] Generating HTML viewer...")
    html = generate_html(fig, contacts, synapses, trace_info, em_snap_dir, brain_trace_indices=brain_indices)
    
    print("\n[3/3] Writing output...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, 'em_viewer.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nComplete! Open: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()
