"""
Quick script to generate skeleton/volume overlap plots from existing analysis results.
Run this after overlap_analysis.py has completed to generate the 2D PNG plots.
"""

import os
import sys
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

# Neuron IDs (restricted set + BIPS)
neuron_ids = {
    720575940618519710: 'MOT_L',
    720575940630139386: 'MOT_R',
    720575940622361270: 'MOS_L',
    720575940622168052: 'MOS_R',
    720575940626477498: 'VS1_L',
    720575940619878961: 'VS1_R',
    720575940640722851: 'VS2_L',
    720575940613126835: 'VS2_R',
    720575940622831740: 'VS3_L',
    720575940641812699: 'VS3_R',
    720575940624273919: 'VS4_L',
    720575940659799937: 'VS4_R',
    720575940628031249: 'HSN_L',
    720575940615933919: 'HSN_R',
    720575940629153020: 'HSE_L',
    720575940629148007: 'HSE_R',
    720575940622312965: 'HSS_L',
    720575940628743496: 'HSS_R',
    720575940623618708: 'BIPS_L',
    720575940622581173: 'BIPS_R',
}

# Color map (MOS, VS, MOT, HS, BIPS)
def get_color(name):
    if 'MOS' in name:
        return '#4D9221'
    if 'VS' in name:
        return '#D14900'
    if 'MOT' in name:
        return '#5E3C99'
    if 'HS' in name:
        return '#C51B7D'
    if 'BIPS' in name:
        return '#000000'
    return '#999999'

def get_group(name):
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
    ("ALL", {"MOT", "MOS", "BIPS"}, {"HS", "VS", "BIPS"}, None),
    ("MOT_only", {"MOT"}, {"HS", "VS", "BIPS"}, None),
    ("MOS_only", {"MOS"}, {"HS", "VS", "BIPS"}, None),
    ("ALL_L", {"MOT", "MOS", "BIPS"}, {"HS", "VS", "BIPS"}, "L"),
    ("ALL_R", {"MOT", "MOS", "BIPS"}, {"HS", "VS", "BIPS"}, "R"),
    ("MOT_only_L", {"MOT"}, {"HS", "VS", "BIPS"}, "L"),
    ("MOT_only_R", {"MOT"}, {"HS", "VS", "BIPS"}, "R"),
    ("MOS_only_L", {"MOS"}, {"HS", "VS", "BIPS"}, "L"),
    ("MOS_only_R", {"MOS"}, {"HS", "VS", "BIPS"}, "R"),
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
            scenario_faces['neuron_a'].str.endswith(f'_{hemi_filter}') &
            scenario_faces['neuron_b'].str.endswith(f'_{hemi_filter}')
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
        
        # Extract overlap triangles
        overlap_triangles = []
        for _, row in neuron_faces.iterrows():
            v1 = [row['vertex1_x'], row['vertex1_y'], row['vertex1_z']]
            v2 = [row['vertex2_x'], row['vertex2_y'], row['vertex2_z']]
            v3 = [row['vertex3_x'], row['vertex3_y'], row['vertex3_z']]
            overlap_triangles.append([v1, v2, v3])
        
        # Get neuron mesh
        neuron_mesh = neurons[source_name]
        
        # Get neuron ID for navis
        neuron_id = next((nid for nid, nname in neuron_ids.items() if nname == source_name), source_name)
        
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
                alpha=1.0,
                view=navis_view,
                method='2d',
                radius=True,
                figsize=(12, 10)
            )
            
            # Handle axis if returned as array
            if isinstance(ax, np.ndarray):
                ax = ax[0]
            
            # Project overlap triangles to 2D
            overlap_triangles_2d = []
            for triangle in overlap_triangles:
                projected = [[v[x_idx], v[y_idx]] for v in triangle]
                overlap_triangles_2d.append(projected)
            
            # Add overlap surfaces
            if overlap_triangles_2d:
                overlap_collection = PolyCollection(
                    overlap_triangles_2d,
                    alpha=0.7,
                    facecolors='red',
                    edgecolors='darkred',
                    linewidths=0.3,
                    zorder=100
                )
                ax.add_collection(overlap_collection)
            
            # Calculate axis limits
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
            
            # Add legend
            target_count = len(neuron_faces['neuron_b'].unique())
            legend_elements = [
                Patch(facecolor=get_color(source_name), alpha=1.0, label=source_name),
                Patch(facecolor='red', alpha=0.7, label=f'Overlaps ({target_count} targets)')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
            
            plt.tight_layout()
            
            # Save plot
            output_file = os.path.join(OUTPUT_DIR, f"{source_name}_{view_name}_{scenario_name}_0.1um.png")
            plt.savefig(output_file, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
            
            print(f"      Saved: {output_file}")

print(f"\n{'='*60}")
print(f"All plots saved to: {OUTPUT_DIR}")
print(f"{'='*60}")
