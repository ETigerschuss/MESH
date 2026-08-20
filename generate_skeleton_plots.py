"""
generate_skeleton_plots.py — Step 2 of the MESH pipeline
=========================================================
Generates 2D PNG projection plots of neuron meshes with overlapping faces
highlighted, for a quick visual overview of all contact regions.

WHAT IT DOES
------------
For each pair of neurons that have overlap faces (from ``contact_faces.csv``):
  1. Load both neuron meshes (OBJ files from ``neuron_meshes/``).
  2. Project mesh triangles into a top-down (XY), front (XZ), or side (YZ) view.
  3. Draw the projected mesh triangles as filled polygons (using matplotlib
     ``PolyCollection``) in each neuron's configured color.
  4. Highlight overlapping faces in red (``#FF0030``) on top.
  5. Save to ``skeleton_plots/<source>_<target>.png``.

In addition, 9 summary scenarios are produced:
  - ALL, MOT-only, MOS-only (for each: all neurons, left hemisphere, right hemisphere)

REQUIREMENTS
------------
trimesh, navis, matplotlib, pandas, numpy, tqdm
Run after overlap_analysis.py (needs contact_faces.csv and OBJ meshes).
"""

"""
Quick script to generate skeleton/volume overlap plots from existing analysis results.
Run this after overlap_analysis.py has completed to generate the 2D PNG plots.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import trimesh
import navis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from tqdm import tqdm

from mesh_config import load_config

# Paths - auto-detect latest results directory
def _default_results_dir():
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d))
                  and d.startswith('comprehensive_overlap_results_')]
    if candidates:
        return sorted(candidates)[-1]
    return 'comprehensive_overlap_results'

RESULTS_DIR = os.environ.get('MESH_RESULTS_DIR', _default_results_dir())
GEO_DATA_DIR = os.path.join(RESULTS_DIR, "geometric_data")
MESH_DIR = os.path.join(RESULTS_DIR, "neuron_meshes")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "overlap_plots_skeleton")

# Load neuron IDs from the active config profile.
_cfg, _CONFIG_PATH = load_config()
neuron_ids = {int(info['id']): name for name, info in _cfg['neurons'].items()}
neuron_colors = {
    name: info.get('color_hex', '#999999')
    for name, info in _cfg['neurons'].items()
}

# Color map (MOS, VS, MOT, HS, BIPS)
def get_color(name):
    return neuron_colors.get(name, '#999999')

def get_group(name):
    if not isinstance(name, str):
        return 'OTHER'
    if 'MOT' in name:
        return 'MOT'
    if 'MOS' in name:
        return 'MOS'
    if 'VS' in name:
        return 'VS'
    if 'HS' in name:
        return 'HS'
    if 'BIPS' in name:
        return 'BIPS'
    return 'OTHER'

print("="*60)
print("Generating Skeleton/Volume Overlap Plots")
print("="*60)

# Load contact faces
contact_faces_file = os.path.join(GEO_DATA_DIR, "contact_faces.csv")
if not os.path.exists(contact_faces_file):
    print(f"ERROR: Contact faces file not found: {contact_faces_file}")
    sys.exit(1)

print(f"Loading contact faces from: {contact_faces_file}")
contact_faces_df = pd.read_csv(contact_faces_file)
print(f"  Loaded {len(contact_faces_df)} contact faces")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load meshes
print("\nLoading neuron meshes...")
neurons = {}
for neuron_id, name in tqdm(neuron_ids.items(), desc="Loading meshes"):
    mesh_file = os.path.join(MESH_DIR, f"{neuron_id}.obj")
    if not os.path.exists(mesh_file):
        print(f"  Warning: Mesh file not found for {name}: {mesh_file}")
        continue
    try:
        mesh = trimesh.load(mesh_file)
        neurons[name] = mesh
    except Exception as e:
        print(f"  Error loading {name}: {e}")

print(f"Loaded {len(neurons)}/{len(neuron_ids)} meshes")

# Define scenarios
scenarios = [
    ("ALL", {"MOT", "MOS"}, {"HS", "VS"}, None),
    ("MOT_only", {"MOT"}, {"HS", "VS"}, None),
    ("MOS_only", {"MOS"}, {"HS", "VS"}, None),
    ("ALL_L", {"MOT", "MOS"}, {"HS", "VS"}, "L"),
    ("ALL_R", {"MOT", "MOS"}, {"HS", "VS"}, "R"),
    ("MOT_only_L", {"MOT"}, {"HS", "VS"}, "L"),
    ("MOT_only_R", {"MOT"}, {"HS", "VS"}, "R"),
    ("MOS_only_L", {"MOS"}, {"HS", "VS"}, "L"),
    ("MOS_only_R", {"MOS"}, {"HS", "VS"}, "R"),
]

# Process each scenario
for scenario_name, source_groups, target_groups, hemi_filter in scenarios:
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")
    
    # Filter contact faces for this scenario
    scenario_faces = contact_faces_df.copy()
    
    # Apply group filters
    scenario_faces = scenario_faces[
        scenario_faces['neuron_a'].apply(lambda x: get_group(x) in source_groups if source_groups else True) &
        scenario_faces['neuron_b'].apply(lambda x: get_group(x) in target_groups if target_groups else True)
    ]
    
    # Apply hemisphere filter
    if hemi_filter:
        scenario_faces = scenario_faces[
            scenario_faces['neuron_a'].astype(str).str.endswith(f'_{hemi_filter}', na=False) &
            scenario_faces['neuron_b'].astype(str).str.endswith(f'_{hemi_filter}', na=False)
        ]
    
    if len(scenario_faces) == 0:
        print(f"  No overlap faces found for this scenario, skipping...")
        continue
    
    print(f"  Found {len(scenario_faces)} overlap faces")
    
    # Group by source neuron
    source_neurons = scenario_faces['neuron_a'].unique()
    print(f"  Source neurons: {list(source_neurons)}")
    
    # Generate plots for each source neuron
    for source_name in source_neurons:
        if source_name not in neurons:
            print(f"    Skipping {source_name} - mesh not loaded")
            continue
        
        print(f"\n  Processing {source_name}...")
        
        # Get overlaps for this neuron
        neuron_faces = scenario_faces[scenario_faces['neuron_a'] == source_name]
        print(f"    Overlap faces: {len(neuron_faces)}")
        
        # Extract overlap triangles, grouped by PARTNER neuron (neuron_b) so each
        # overlap patch is drawn in its partner's color. Degenerate faces (all 3
        # vertices identical) fall back to a centroid scatter point.
        overlap_tris_by_target = {}    # target -> [[v1,v2,v3], ...]
        overlap_cents_by_target = {}   # target -> [[cx,cy,cz], ...]
        for _, row in neuron_faces.iterrows():
            tgt = row['neuron_b']
            v1 = [row['vertex1_x'], row['vertex1_y'], row['vertex1_z']]
            v2 = [row['vertex2_x'], row['vertex2_y'], row['vertex2_z']]
            v3 = [row['vertex3_x'], row['vertex3_y'], row['vertex3_z']]
            # Check if triangle is degenerate (all vertices same or near-same)
            if (np.allclose(v1, v2, atol=1.0) and np.allclose(v2, v3, atol=1.0)):
                # Degenerate face from recycled data - use centroid
                cx = row.get('centroid_x', v1[0])
                cy = row.get('centroid_y', v1[1])
                cz = row.get('centroid_z', v1[2])
                overlap_cents_by_target.setdefault(tgt, []).append([cx, cy, cz])
            else:
                overlap_tris_by_target.setdefault(tgt, []).append([v1, v2, v3])
        n_tris = sum(len(v) for v in overlap_tris_by_target.values())
        n_cents = sum(len(v) for v in overlap_cents_by_target.values())
        
        # Get neuron mesh
        neuron_mesh = neurons[source_name]
        
        # Get neuron ID for navis
        neuron_id = next((nid for nid, nname in neuron_ids.items() if nname == source_name), source_name)
        
        try:
            # Convert to navis and downsample
            print(f"    Converting to navis MeshNeuron...")
            mesh_neuron = navis.MeshNeuron(neuron_mesh, id=neuron_id, name=source_name)
            mesh_neuron_ds = navis.downsample_neuron(mesh_neuron, downsampling_factor=10)
        
            # Generate plots (frontal and horizontal views)
            views = [
                ('frontal', 0, 1, ('x', '-y')),
                ('horizontal', 0, 2, ('x', '-z'))
            ]
        
            for view_name, x_idx, y_idx, navis_view in views:
                print(f"    Generating {view_name} view...")
            
                # Plot neuron with navis
                fig, ax = navis.plot2d(
                    mesh_neuron_ds,
                    color=get_color(source_name),
                    alpha=0.55,
                    view=navis_view,
                    method='2d',
                    radius=True,
                    figsize=(12, 10)
                )
            
                # Handle axis if returned as array
                if isinstance(ax, np.ndarray):
                    ax = ax[0]
            
                # Draw each PARTNER's overlap in its color, EXAGGERATED so it
                # pops: a big translucent glow halo + a saturated body underneath,
                # with the crisp filled patch on top. Dense overlaps bloom into
                # bold colored blobs.
                legend_targets = set()
                for tgt, tris in overlap_tris_by_target.items():
                    col = get_color(tgt)
                    tris_2d = [[[v[x_idx], v[y_idx]] for v in tri] for tri in tris]
                    gx = [(tri[0][x_idx] + tri[1][x_idx] + tri[2][x_idx]) / 3.0 for tri in tris]
                    gy = [(tri[0][y_idx] + tri[1][y_idx] + tri[2][y_idx]) / 3.0 for tri in tris]
                    ax.scatter(gx, gy, s=340, c=col, alpha=0.20, edgecolors='none', zorder=98)
                    ax.scatter(gx, gy, s=120, c=col, alpha=0.50, edgecolors='none', zorder=99)
                    ax.add_collection(PolyCollection(
                        tris_2d, alpha=1.0, facecolors=col, edgecolors=col,
                        linewidths=0.6, zorder=100))
                    legend_targets.add(tgt)
                for tgt, cents in overlap_cents_by_target.items():
                    col = get_color(tgt)
                    cx = [c[x_idx] for c in cents]; cy = [c[y_idx] for c in cents]
                    ax.scatter(cx, cy, s=340, c=col, alpha=0.20, edgecolors='none', zorder=98)
                    ax.scatter(cx, cy, s=120, c=col, alpha=0.50, edgecolors='none', zorder=99)
                    ax.scatter(cx, cy, s=55, c=col, alpha=1.0, edgecolors='#333333',
                               linewidths=0.5, zorder=101, marker='o')
                    legend_targets.add(tgt)
                # Legend: one swatch per partner color present (HS / VS1-4 / VS5-8)
                _grp_label = {"VS": "VS1-4", "VS5_8": "VS5-8", "HS": "HS",
                              "MOT": "MOT", "MOS": "MOS"}
                _seen = {}  # color -> label
                for tgt in legend_targets:
                    grp = _cfg['neurons'].get(tgt, {}).get('group', '')
                    _seen.setdefault(get_color(tgt), _grp_label.get(grp, grp or tgt))
                if _seen:
                    from matplotlib.patches import Patch as _Patch
                    ax.legend(handles=[_Patch(facecolor=c, label=l) for c, l in _seen.items()],
                              loc='upper right', fontsize=8, frameon=False,
                              title='Overlap partner')

                overlap_label = f'{n_tris} faces + {n_cents} centroids'            # Calculate axis limits
                neuron_coords_x = neuron_mesh.vertices[:, x_idx]
                neuron_coords_y = neuron_mesh.vertices[:, y_idx]
            
                x_min, x_max = neuron_coords_x.min(), neuron_coords_x.max()
                y_min, y_max = neuron_coords_y.min(), neuron_coords_y.max()
            
                x_padding = (x_max - x_min) * 0.05
                y_padding = (y_max - y_min) * 0.05
            
                ax.set_xlim(x_min - x_padding, x_max + x_padding)
                ax.set_ylim(y_min - y_padding, y_max + y_padding)
                ax.set_aspect('equal')
                ax.set_xlabel('X (nm)' if x_idx == 0 else 'Y (nm)', fontsize=10)
                ax.set_ylabel('Y (nm)' if y_idx == 1 else 'Z (nm)', fontsize=10)
                ax.grid(False)
                ax.set_facecolor('white')
            
                plt.tight_layout()
            
                # Save plot
                output_file = os.path.join(OUTPUT_DIR, f"{source_name}_{view_name}_{scenario_name}_0.1um.png")
                plt.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
                plt.close()
            
                print(f"      Saved: {output_file}")
        except Exception as e:
            print(f"    Error processing {source_name}: {e}")
            continue

print(f"\n{'='*60}")
print(f"All plots saved to: {OUTPUT_DIR}")
print(f"{'='*60}")
