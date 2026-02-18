"""
Skeleton EM Viewer - Navis Skeleton Rendering
===============================

Complete self-contained EM viewer with navis skeleton rendering with integrated HTML template.
No external dependencies beyond standard Python libraries and Plotly.

Features:
- Left sidebar: Neuron controls
- Center: 3D mesh viewer with highlighting
- Right: EM snapshot viewer with resizable panels
- Highlights selected contact (solid red) or synapse (larger yellow) in 3D view

Usage:
    python skeleton_em_viewer.py

Output:
    comprehensive_overlap_results/skeleton_em_viewer.html
"""

import os
import json
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import trimesh
import navis

# Neuron IDs - restricted set + BIPS
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

# Neuroscience colors (MOS, VS, MOT, HS, BIPS)
NEURON_COLORS = {
    'MOS_L': '#4D9221', 'MOS_R': '#4D9221',
    'VS1_L': '#D14900', 'VS1_R': '#D14900',
    'VS2_L': '#D14900', 'VS2_R': '#D14900',
    'VS3_L': '#D14900', 'VS3_R': '#D14900',
    'VS4_L': '#D14900', 'VS4_R': '#D14900',
    'MOT_L': '#5E3C99', 'MOT_R': '#5E3C99',
    'HSN_L': '#C51B7D', 'HSN_R': '#C51B7D',
    'HSE_L': '#C51B7D', 'HSE_R': '#C51B7D',
    'HSS_L': '#C51B7D', 'HSS_R': '#C51B7D',
    'BIPS_L': '#000000', 'BIPS_R': '#000000',
}

def _default_results_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d))
                  and d.startswith('comprehensive_overlap_results_')]
    if candidates:
        return sorted(candidates)[-1]
    return 'comprehensive_overlap_results'

RESULTS_DIR = os.environ.get('MESH_RESULTS_DIR', _default_results_dir())

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
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
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
                .filter(item => visibleNeurons.includes(item.source))
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
            
            emTitle.textContent = `${kind.charAt(0).toUpperCase() + kind.slice(1)} #${idx}`;
            emLocation.textContent = `${source} → ${target} at (${Math.round(x)}, ${Math.round(y)}, ${Math.round(z)})`;
            itemInfo.textContent = `${kind.charAt(0).toUpperCase() + kind.slice(1)} ${currentListIndex + 1}/${currentList.length} (idx: ${idx})`;
            
            loadImage(kind, idx, 0);
            highlightPoint(kind, idx);
        }
        
        function loadImage(kind, idx, zOffset) {
            const imgData = snapshotMap[kind]?.[idx];
            if (!imgData) {
                emImage.style.display = 'none';
                emPlaceholder.style.display = 'block';
                emPlaceholder.textContent = `No snapshot for ${kind} ${idx}`;
                return;
            }
            
            if (zOffset === 0) {
                emImage.src = imgData;
                emImage.style.display = 'block';
                emPlaceholder.style.display = 'none';
                updateZValue(0);
            } else {
                loadZStackImage(kind, idx, zOffset);
            }
        }
        
        function loadZStackImage(kind, idx, zOffset) {
            const zStr = zOffset >= 0 ? `z+${String(Math.abs(zOffset)).padStart(3, '0')}` : `z-${String(Math.abs(zOffset)).padStart(3, '0')}`;
            const filename = `${kind}_${idx}_${zStr}.png`;
            const path = `em_snaps/${filename}`;
            
            emImage.onerror = function() {
                const centerData = snapshotMap[kind]?.[idx];
                if (centerData && zOffset !== 0) {
                    emImage.src = centerData;
                }
            };
            
            emImage.onload = function() {
                emImage.style.display = 'block';
                emPlaceholder.style.display = 'none';
                updateZValue(zOffset);
            };
            
            emImage.src = path;
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
        
        // Item navigation
        btnPrevItem.addEventListener('click', function() {
            if (currentKind && currentListIndex > 0) {
                currentListIndex--;
                const newIdx = currentList[currentListIndex];
                currentIdx = newIdx;
                loadImage(currentKind, newIdx, currentZ);
                itemInfo.textContent = `${currentKind.charAt(0).toUpperCase() + currentKind.slice(1)} ${currentListIndex + 1}/${currentList.length} (idx: ${newIdx})`;
                highlightPoint(currentKind, newIdx);
            }
        });
        
        btnNextItem.addEventListener('click', function() {
            if (currentKind && currentListIndex < currentList.length - 1) {
                currentListIndex++;
                const newIdx = currentList[currentListIndex];
                currentIdx = newIdx;
                loadImage(currentKind, newIdx, currentZ);
                itemInfo.textContent = `${currentKind.charAt(0).toUpperCase() + currentKind.slice(1)} ${currentListIndex + 1}/${currentList.length} (idx: ${newIdx})`;
                highlightPoint(currentKind, newIdx);
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


def load_mesh_skeleton(neuron_name, mesh_dir, downsample_factor=50):
    """Load mesh and convert to skeleton lines using navis"""
    import trimesh
    import navis

    neuron_id = {v: k for k, v in NEURON_IDS.items()}[neuron_name]
    mesh_file = os.path.join(mesh_dir, f"{neuron_id}.obj")

    if not os.path.exists(mesh_file):
        return None, None, None

    try:
        # Load mesh
        mesh = trimesh.load(mesh_file)

        # Convert to navis MeshNeuron
        mesh_neuron = navis.MeshNeuron(mesh, id=neuron_id, name=neuron_name)

        # Downsample to reduce complexity
        mesh_neuron_ds = navis.downsample_neuron(mesh_neuron, downsampling_factor=downsample_factor)        # Get vertices and faces
        vertices = mesh_neuron_ds.vertices
        faces = mesh_neuron_ds.faces
        
        # Create line segments from faces (edges)
        edges = set()
        for face in faces:
            # Each triangular face has 3 edges
            edges.add(tuple(sorted([face[0], face[1]])))
            edges.add(tuple(sorted([face[1], face[2]])))
            edges.add(tuple(sorted([face[2], face[0]])))
        
        # Convert edges to line coordinates
        x_lines, y_lines, z_lines = [], [], []
        for v1_idx, v2_idx in edges:
            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]
            x_lines.extend([v1[0], v2[0], None])  # None creates a break
            y_lines.extend([v1[1], v2[1], None])
            z_lines.extend([v1[2], v2[2], None])
        
        return np.array(x_lines), np.array(y_lines), np.array(z_lines)
    except Exception as e:
        print(f"Error loading skeleton for {neuron_name}: {e}")
        return None, None, None


def build_figure(mesh_dir):
    """Build the 3D Plotly figure with highlighting capability"""
    contacts = load_contacts(RESULTS_DIR)
    synapses = load_synapses(RESULTS_DIR)
    
    traces = []
    trace_info = {}
    
    print("[meshes] Loading neuron data...")
    for neuron_name in sorted(NEURON_IDS.values()):
        color = NEURON_COLORS.get(neuron_name, '#888888')
        
        # 1. Mesh trace
        x, y, z = load_mesh_skeleton(neuron_name, mesh_dir)
        if x is not None:
            trace_idx = len(traces)
            trace_info[f"{neuron_name}_mesh"] = trace_idx
            traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                name=f"{neuron_name}_mesh",
                line=dict(width=1, color=color),
                opacity=0.5,
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
                marker=dict(size=15, color='yellow', opacity=1.0, symbol='diamond'),
                hovertemplate='SELECTED Synapse<extra></extra>',
                legendgroup=neuron_name,
                showlegend=False
            ))
            print(f"  {neuron_name}: {M} synapses")
    
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
    
    return fig, contacts, synapses, trace_info


def index_em_snapshots(em_snap_dir, contacts, synapses):
    """Index and embed EM snapshots as base64"""
    snapshot_map = {'contact': {}, 'synapse': {}}
    
    # Index contacts
    for idx in contacts['idx'].unique():
        segmented_file = os.path.join(em_snap_dir, f"contact_{idx}_segmented.png")
        if os.path.exists(segmented_file):
            with open(segmented_file, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
                snapshot_map['contact'][int(idx)] = f"data:image/png;base64,{b64}"
    
    print(f"[snapshots] Indexed {len(snapshot_map['contact'])} contacts")
    
    # Index synapses
    for idx in synapses['index'].unique():
        segmented_file = os.path.join(em_snap_dir, f"synapse_{idx}_segmented.png")
        if os.path.exists(segmented_file):
            with open(segmented_file, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
                snapshot_map['synapse'][int(idx)] = f"data:image/png;base64,{b64}"
    
    print(f"[snapshots] Indexed {len(snapshot_map['synapse'])} synapses")
    
    return snapshot_map


def generate_html(fig, contacts, synapses, trace_info, em_snap_dir):
    """Generate improved HTML with integrated template"""
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
    
    return html_content


def main():
    print("="*70)
    print("EM Viewer - Generating Interactive Viewer")
    print("="*70)
    
    mesh_dir = os.path.join(RESULTS_DIR, 'neuron_meshes')
    em_snap_dir = os.path.join(RESULTS_DIR, 'em_snaps')
    
    print("\n[1/3] Building 3D figure with highlighting...")
    fig, contacts, synapses, trace_info = build_figure(mesh_dir)
    
    print("\n[2/3] Generating HTML viewer...")
    html = generate_html(fig, contacts, synapses, trace_info, em_snap_dir)
    
    print("\n[3/3] Writing output...")
    output_file = os.path.join(RESULTS_DIR, 'skeleton_em_viewer.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\nComplete! Open: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()


