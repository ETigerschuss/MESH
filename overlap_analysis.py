# Comprehensive Pairwise Neuron Analysis Script
# This script analyzes all neuron pairs at multiple distance thresholds

import navis
import fafbseg
from fafbseg import flywire
from fafbseg.flywire.synapses import get_synapses
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import colors as mpl_colors
from colorsys import rgb_to_hls, hls_to_rgb
import matplotlib.pyplot as plt
import trimesh
from tqdm import tqdm
from scipy.spatial import cKDTree
import time
import os
import shutil
import pickle
import hashlib
import traceback

# Set Flywire Token (non-interactive-safe)
# Try env var first, then fall back to saved cave-secret.json
token = os.environ.get('FLYWIRE_TOKEN')
if not token:
    import json
    _secret_path = os.path.expanduser("~/.cloudvolume/secrets/cave-secret.json")
    if os.path.exists(_secret_path):
        with open(_secret_path) as _f:
            token = json.load(_f).get('token', '')
if not token:
    raise ValueError("FlyWire token missing. Set env FLYWIRE_TOKEN or save in ~/.cloudvolume/secrets/cave-secret.json")

fafbseg.flywire.set_chunkedgraph_secret(token, overwrite=True)

# Load neuron config from central neurons.json
import json as _json
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neurons.json'), 'r') as _nf:
    _cfg = _json.load(_nf)
neuron_ids = {int(info['id']): name for name, info in _cfg['neurons'].items()}
N_TOP_PATCHES = _cfg.get('top_patches', 10)

# Configuration
THRESHOLDS_MICRONS = [0.1]
LOD = 50
CACHE_DIR = "overlap_cache"

def _default_results_dir():
    """Always create a new results folder with today's date."""
    from datetime import date
    return f"comprehensive_overlap_results_{date.today().isoformat()}"

def _find_previous_results():
    """Find all previous results directories (sorted newest first).
    Returns list of absolute paths to previous result dirs."""
    base = os.path.dirname(os.path.abspath(__file__))
    today_dir = _default_results_dir()
    candidates = [os.path.join(base, d) for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d))
                  and d.startswith('comprehensive_overlap_results_')
                  and d != today_dir]
    return sorted(candidates, reverse=True)  # newest first

def _find_previous_meshes():
    """Scan previous results for already-downloaded neuron mesh OBJ files.
    Returns dict: {neuron_id: path_to_obj}"""
    found = {}
    for prev_dir in _find_previous_results():
        mesh_dir = os.path.join(prev_dir, 'neuron_meshes')
        if not os.path.isdir(mesh_dir):
            continue
        for fname in os.listdir(mesh_dir):
            if fname.endswith('.obj'):
                nid = int(fname.replace('.obj', ''))
                if nid not in found:
                    found[nid] = os.path.join(mesh_dir, fname)
    return found

def _find_previous_pair_results(threshold_um):
    """Scan previous results for already-computed pair overlap data.
    Returns dict mapping pair_key -> (area, geo_data) where geo_data has
    DISABLED: Always recompute to get real mesh face geometry."""
    return {}  # Force fresh computation for real triangle vertices
    found = {}
    for prev_dir in _find_previous_results():
        csv_file = os.path.join(prev_dir, f'results_{threshold_um}um.csv')
        if not os.path.exists(csv_file):
            continue
        try:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                src = row.get('Source_Neuron', row.get('Source', ''))
                tgt = row.get('Target_Neuron', row.get('Target', ''))
                pair_key = f"{src}→{tgt}"
                if pair_key not in found:
                    area = float(row.get('Contact_Area_um2', 0))
                    # Reconstruct face_data from Top1..TopN columns
                    face_data = []
                    for pn in range(1, N_TOP_PATCHES + 1):
                        cx = row.get(f'Top{pn}_Patch_Centroid_X', np.nan)
                        cy = row.get(f'Top{pn}_Patch_Centroid_Y', np.nan)
                        cz = row.get(f'Top{pn}_Patch_Centroid_Z', np.nan)
                        pa = row.get(f'Top{pn}_Patch_Area_um2', np.nan)
                        if pd.notna(cx) and pd.notna(cy) and pd.notna(cz):
                            face_data.append({
                                'face_idx': pn - 1,
                                'vertices': np.array([[cx,cy,cz],[cx,cy,cz],[cx,cy,cz]]),
                                'area': float(pa) * 1e6 if pd.notna(pa) else 0.0,
                                'centroid': np.array([float(cx), float(cy), float(cz)])
                            })
                    geo = create_empty_geometric_data()
                    geo['face_data'] = face_data
                    geo['contact_area'] = area
                    # Also store other fields if available
                    for field, key in [('total_area_meshA','Total_Area_Source_um2'),
                                       ('total_area_meshB','Total_Area_Target_um2')]:
                        v = row.get(key, np.nan)
                        if pd.notna(v):
                            geo[field] = float(v)
                    found[pair_key] = (area, geo)
        except Exception as e:
            print(f"  Warning: could not read {csv_file}: {e}")
        if found:
            break
    return found

def _find_previous_patch_data(threshold_um):
    """Scan previous results for saved individual patch CSVs.
    Returns dict: {pair_key: path_to_csv}"""
    found = {}
    for prev_dir in _find_previous_results():
        patch_dir = os.path.join(prev_dir, f'individual_patches_threshold_{threshold_um}um')
        if not os.path.isdir(patch_dir):
            continue
        for fname in os.listdir(patch_dir):
            if fname.endswith('_patch_data.csv'):
                # e.g. MOT_L_to_HSN_L_patch_data.csv
                stem = fname.replace('_patch_data.csv', '')
                parts = stem.split('_to_')
                if len(parts) == 2:
                    pair_key = f"{parts[0]}→{parts[1]}"
                    if pair_key not in found:
                        found[pair_key] = os.path.join(patch_dir, fname)
        if found:
            break
    return found

def _find_previous_synapses():
    """Scan previous results for saved synapses.csv."""
    for prev_dir in _find_previous_results():
        syn_file = os.path.join(prev_dir, 'synapses.csv')
        if os.path.exists(syn_file):
            return syn_file
    return None

def _find_previous_em_snaps():
    """Scan previous results for em_snaps directory."""
    for prev_dir in _find_previous_results():
        snap_dir = os.path.join(prev_dir, 'em_snaps')
        if os.path.isdir(snap_dir) and os.listdir(snap_dir):
            return snap_dir
    return None

RESULTS_DIR = os.environ.get('MESH_RESULTS_DIR', _default_results_dir())

def get_cache_key(neuron_ids, thresholds, lod):
    """Generate cache key for analysis parameters"""
    key_string = f"{sorted(neuron_ids.items())}_{thresholds}_{lod}"
    return hashlib.md5(key_string.encode()).hexdigest()

def save_to_cache(data, cache_key):
    """Save analysis results to cache with memory management"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    try:
        # Try to save normally first
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Results cached to: {cache_file}")
    except MemoryError:
        print("Memory error during caching - saving simplified version")
        # Create a simplified version without geometric data
        simplified_data = {}
        if 'all_results' in data:
            simplified_data['all_results'] = {}
            for threshold, results in data['all_results'].items():
                simplified_data['all_results'][threshold] = {}
                for pair_key, result in results.items():
                    # Keep only the area, not the geometric data
                    if isinstance(result, tuple):
                        area = result[0]
                    else:
                        area = result
                    simplified_data['all_results'][threshold][pair_key] = area
        
        if 'valid_names' in data:
            simplified_data['valid_names'] = data['valid_names']
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(simplified_data, f)
            print(f"Simplified results cached to: {cache_file}")
        except Exception as e:
            print(f"Failed to cache results: {e}")

def save_incremental_results(results, threshold_um, output_dir):
    """Save results incrementally as they are computed"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual threshold results
    results_list = []
    for pair_key, result in results.items():
        source, target = pair_key.split('→')
        # Extract area and geometric data from tuple (area, geo_data) or use result directly
        if isinstance(result, tuple):
            area, geo_data = result
        else:
            area = result
            geo_data = None
        # Calculate contact patch centroid and other derived values
        contact_centroid = np.array([0.0, 0.0, 0.0])
        is_larger_patch = area > 10.0
        if geo_data and area > 0:
            face_data = geo_data.get('face_data', [])
            if face_data:
                # Weighted average by area
                total_patch_area = 0
                for face_info in face_data:
                    patch_area = face_info['area'] / 1e6  # Convert to um²
                    contact_centroid += face_info['centroid'] * patch_area
                    total_patch_area += patch_area
                if total_patch_area > 0:
                    contact_centroid /= total_patch_area
        
        # Compute top-N largest contact patches (by area) if available
        top_patches = []
        if geo_data and area > 0:
            face_data = geo_data.get('face_data', []) or []
            if face_data:
                top_patches = sorted(face_data, key=lambda fd: fd.get('area', 0), reverse=True)[:N_TOP_PATCHES]

        result_row = {
            'Source_Neuron': source,
            'Target_Neuron': target,
            'Contact_Area_um2': area,
            'Threshold_um': threshold_um,
            'Has_Contact': area > 0,
            'Contact_Patch_Centroid_X': contact_centroid[0] if area > 0 else np.nan,
            'Contact_Patch_Centroid_Y': contact_centroid[1] if area > 0 else np.nan,
            'Contact_Patch_Centroid_Z': contact_centroid[2] if area > 0 else np.nan,
            'Contact_Patch_Centroid_X_Norm': contact_centroid[0] / 4.0 if area > 0 else np.nan,
            'Contact_Patch_Centroid_Y_Norm': contact_centroid[1] / 4.0 if area > 0 else np.nan,
            'Contact_Patch_Centroid_Z_Norm': contact_centroid[2] / 40.0 if area > 0 else np.nan,
            'Is_Larger_Patch': is_larger_patch if area > 0 else False
        }
        # Add Top1..TopN patch areas (um²) and centroids
        for idx in range(N_TOP_PATCHES):
            key_area = f'Top{idx+1}_Patch_Area_um2'
            key_cx = f'Top{idx+1}_Patch_Centroid_X'
            key_cy = f'Top{idx+1}_Patch_Centroid_Y'
            key_cz = f'Top{idx+1}_Patch_Centroid_Z'
            if idx < len(top_patches):
                fd = top_patches[idx]
                result_row[key_area] = (fd.get('area', np.nan) / 1e6)
                c = fd.get('centroid', [np.nan, np.nan, np.nan])
                result_row[key_cx] = c[0] if isinstance(c, (list, tuple, np.ndarray)) and len(c) >= 1 else np.nan
                result_row[key_cy] = c[1] if isinstance(c, (list, tuple, np.ndarray)) and len(c) >= 2 else np.nan
                result_row[key_cz] = c[2] if isinstance(c, (list, tuple, np.ndarray)) and len(c) >= 3 else np.nan
            else:
                result_row[key_area] = np.nan
                result_row[key_cx] = np.nan
                result_row[key_cy] = np.nan
                result_row[key_cz] = np.nan
        
        # Add additional geometric data if available
        if geo_data and area > 0:
            result_row.update({
                'Total_Area_Source_um2': geo_data.get('total_area_meshA', np.nan),
                'Total_Area_Target_um2': geo_data.get('total_area_meshB', np.nan),
                'Num_Contact_Vertices': len(geo_data.get('close_vertices_meshA', [])),
                'Num_Contact_Faces': len(geo_data.get('face_data', []))
            })
        else:
            result_row.update({
                'Total_Area_Source_um2': np.nan,
                'Total_Area_Target_um2': np.nan,
                'Num_Contact_Vertices': 0,
                'Num_Contact_Faces': 0
            })
        
        results_list.append(result_row)
    
    df = pd.DataFrame(results_list)
    output_file = os.path.join(output_dir, f"results_{threshold_um}um.csv")
    df.to_csv(output_file, index=False)
    print(f"Incremental results saved: {output_file}")
    
    # Also save a summary for this threshold
    areas = [row['Contact_Area_um2'] for row in results_list if row['Contact_Area_um2'] > 0]
    total_pairs = len(results_list)
    connected_pairs = len(areas)
    
    summary = {
        'Threshold_um': threshold_um,
        'Total_Pairs': total_pairs,
        'Connected_Pairs': connected_pairs,
        'Connection_Percentage': (connected_pairs/total_pairs)*100 if total_pairs > 0 else 0,
        'Mean_Contact_Area': np.mean(areas) if areas else 0,
        'Std_Contact_Area': np.std(areas) if areas else 0,
        'Min_Contact_Area': np.min(areas) if areas else 0,
        'Max_Contact_Area': np.max(areas) if areas else 0,
        'Total_Contact_Area': np.sum(areas) if areas else 0
    }
    
    summary_file = os.path.join(output_dir, f"summary_{threshold_um}um.csv")
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved: {summary_file}")
    
    return df, summary

def load_from_cache(cache_key):
    """Load analysis results from cache"""
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded cached results from: {cache_file}")
            
            # Check if this is simplified cache data (areas only)
            if 'all_results' in data:
                sample_result = None
                for threshold_results in data['all_results'].values():
                    for result in threshold_results.values():
                        sample_result = result
                        break
                    if sample_result is not None:
                        break
                
                # If cached data only has areas (not tuples), convert to expected format
                if sample_result is not None and not isinstance(sample_result, tuple):
                    print("Converting simplified cache data to full format...")
                    for threshold, results in data['all_results'].items():
                        for pair_key, area in results.items():
                            data['all_results'][threshold][pair_key] = (area, create_empty_geometric_data())
            
            return data
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    return None

def save_detailed_geometric_data(all_results, output_dir):
    """Save detailed geometric data to CSV files"""
    print("\n============================================================")
    print("STEP 6: Saving detailed geometric data")
    print("============================================================")
    
    # Create subdirectory for geometric data
    geo_dir = os.path.join(output_dir, 'geometric_data')
    os.makedirs(geo_dir, exist_ok=True)
    
    # Initialize data collections
    all_vertices_data = []
    all_faces_data = []
    all_centroids_data = []
    all_areas_data = []
    
    # Convert the format to include neuron pairs as keys
    restructured_results = {}
    total_pairs = 0
    pairs_with_data = 0
    
    for threshold, results in all_results.items():
        print(f"Processing threshold {threshold}: {len(results)} pairs")
        for pair_key, result_data in results.items():
            total_pairs += 1
            neuron_a, neuron_b = pair_key.split('→')
            pair_tuple = (neuron_a, neuron_b)
            if pair_tuple not in restructured_results:
                restructured_results[pair_tuple] = {}
            
            # Handle tuple format (area, geo_data)
            if isinstance(result_data, tuple):
                area, geo_data = result_data
                pairs_with_data += 1
                print(f"  Pair {pair_key}: area={area:.4f}, has_geo_data={geo_data is not None}")
            else:
                area = result_data
                geo_data = create_empty_geometric_data()
                print(f"  Pair {pair_key}: area={area:.4f}, no geometric data")
            
            restructured_results[pair_tuple][threshold] = (area, geo_data)
    
    print(f"Total processing summary:")
    print(f"  Total pairs processed: {total_pairs}")
    print(f"  Pairs with tuple data: {pairs_with_data}")
    print(f"  Restructured pairs: {len(restructured_results)}")
    
    for (neuron_a, neuron_b), threshold_data in restructured_results.items():
        for threshold, (area, geo_data) in threshold_data.items():
            if area > 0 and geo_data:  # Only process pairs with actual overlap
                base_record = {
                    'neuron_a': neuron_a,
                    'neuron_b': neuron_b,
                    'threshold_um': threshold,
                    'contact_area_um2': area
                }
                
                # Vertices data
                if len(geo_data['close_vertices_meshA']) > 0:
                    for i, (vert_a, vert_b, dist) in enumerate(zip(
                        geo_data['close_vertices_meshA'],
                        geo_data['close_vertices_meshB'],
                        geo_data['distances']
                    )):
                        mid = (np.array(vert_a) + np.array(vert_b)) / 2.0
                        all_vertices_data.append({
                            **base_record,
                            'vertex_pair_id': i,
                            'vertex_a_x': vert_a[0],
                            'vertex_a_y': vert_a[1],
                            'vertex_a_z': vert_a[2],
                            'vertex_b_x': vert_b[0],
                            'vertex_b_y': vert_b[1],
                            'vertex_b_z': vert_b[2],
                            'distance': dist,
                            'mid_x': float(mid[0]),
                            'mid_y': float(mid[1]),
                            'mid_z': float(mid[2])
                        })
                
                # Face data
                for i, face_info in enumerate(geo_data['face_data']):
                    # Face centroid
                    all_faces_data.append({
                        **base_record,
                        'face_id': i,
                        'face_idx_original': face_info['face_idx'],
                        'face_area_um2': face_info['area'] / 1e6,  # Convert to um²
                        'centroid_x': face_info['centroid'][0],
                        'centroid_y': face_info['centroid'][1],
                        'centroid_z': face_info['centroid'][2],
                        'vertex1_x': face_info['vertices'][0][0],
                        'vertex1_y': face_info['vertices'][0][1],
                        'vertex1_z': face_info['vertices'][0][2],
                        'vertex2_x': face_info['vertices'][1][0],
                        'vertex2_y': face_info['vertices'][1][1],
                        'vertex2_z': face_info['vertices'][1][2],
                        'vertex3_x': face_info['vertices'][2][0],
                        'vertex3_y': face_info['vertices'][2][1],
                        'vertex3_z': face_info['vertices'][2][2]
                    })
                
                # Centroids and areas summary
                all_centroids_data.append({
                    **base_record,
                    'centroid_a_x': geo_data['centroid_meshA'][0],
                    'centroid_a_y': geo_data['centroid_meshA'][1],
                    'centroid_a_z': geo_data['centroid_meshA'][2],
                    'centroid_b_x': geo_data['centroid_meshB'][0],
                    'centroid_b_y': geo_data['centroid_meshB'][1],
                    'centroid_b_z': geo_data['centroid_meshB'][2],
                    'total_area_a_um2': geo_data['total_area_meshA'],
                    'total_area_b_um2': geo_data['total_area_meshB'],
                    'num_vertices_a': geo_data['num_vertices_meshA'],
                    'num_vertices_b': geo_data['num_vertices_meshB'],
                    'num_faces_a': geo_data['num_faces_meshA'],
                    'num_faces_b': geo_data['num_faces_meshB'],
                    'num_contact_vertices': len(geo_data['close_vertices_meshA']),
                    'num_contact_faces': len(geo_data['face_data'])
                })
                
                # Individual area data for each face
                for i, face_info in enumerate(geo_data['face_data']):
                    all_areas_data.append({
                        **base_record,
                        'patch_id': i,
                        'patch_area_um2': face_info['area'] / 1e6,
                        'patch_centroid_x': face_info['centroid'][0],
                        'patch_centroid_y': face_info['centroid'][1],
                        'patch_centroid_z': face_info['centroid'][2]
                    })
    
    # Save to CSV files
    if all_vertices_data:
        vertices_df = pd.DataFrame(all_vertices_data)
        vertices_file = os.path.join(geo_dir, 'contact_vertices.csv')
        vertices_df.to_csv(vertices_file, index=False)
        print(f"Saved contact vertices: {vertices_file}")
        print(f"  Total vertex pairs: {len(all_vertices_data)}")
    
    if all_faces_data:
        faces_df = pd.DataFrame(all_faces_data)
        faces_file = os.path.join(geo_dir, 'contact_faces.csv')
        faces_df.to_csv(faces_file, index=False)
        print(f"Saved contact faces: {faces_file}")
        print(f"  Total contact faces: {len(all_faces_data)}")
    
    if all_centroids_data:
        centroids_df = pd.DataFrame(all_centroids_data)
        centroids_file = os.path.join(geo_dir, 'neuron_centroids_and_areas.csv')
        centroids_df.to_csv(centroids_file, index=False)
        print(f"Saved neuron centroids and areas: {centroids_file}")
        print(f"  Total neuron pairs: {len(all_centroids_data)}")
    
    if all_areas_data:
        areas_df = pd.DataFrame(all_areas_data)
        areas_file = os.path.join(geo_dir, 'contact_patches.csv')
        areas_df.to_csv(areas_file, index=False)
        print(f"Saved contact patches: {areas_file}")
        print(f"  Total contact patches: {len(all_areas_data)}")
    
    print(f"\nAll geometric data saved in: {geo_dir}/")
    return geo_dir

def _log_viz_debug(msg: str):
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, 'viz_debug.log'), 'a', encoding='utf-8') as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass

def _write_placeholder_html(output_dir, name="visualization_placeholder.html"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        fig = go.Figure(data=[go.Scatter3d(x=[0], y=[0], z=[0], mode='markers',
                                           marker=dict(size=6, color='red'),
                                           name='Placeholder')])
        fig.update_layout(title='Visualization placeholder',
                          scene=dict(aspectmode='cube'))
        path = os.path.join(output_dir, name)
        fig.write_html(path)
        print(f"Wrote placeholder HTML: {path}")
        _log_viz_debug(f"Placeholder HTML written to {path}")
    except Exception as e:
        _log_viz_debug(f"Failed to write placeholder HTML: {e}")

def build_meshes_only_html(neurons, valid_names, output_dir, title_suffix="All neuron meshes"):
    """Save an interactive HTML rendering of all loaded neuron meshes with group colors.
    This is generated immediately after loading to guarantee at least one figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    color_map = _build_color_map(valid_names)

    traces = []
    for name in valid_names:
        neuron = neurons.get(name)
        if neuron is None:
            continue
        traces.append(_mesh_trace_from_neuron(neuron, name, color_map.get(name, '#888888'), opacity=0.22, max_faces_per_mesh=30000))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title_suffix,
        scene=dict(
            xaxis=dict(title='X', showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(title='Y', showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(title='Z', showbackground=False, showgrid=False, zeroline=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.1))
        ),
        legend=dict(itemsizing='constant', bgcolor='rgba(255,255,255,0.6)'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=1400,
        height=1100,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    html_path = os.path.join(output_dir, "meshes_only.html")
    fig.write_html(html_path)
    print(f"Saved meshes-only visualization: {html_path}")

def _get_group(name: str) -> str:
    if name.startswith("MOT"):
        return "MOT"
    if name.startswith("MOS"):
        return "MOS"
    if name.startswith("VS"):
        return "VS"
    if name.startswith("HS"):
        return "HS"
    if name.startswith("BIPS"):
        return "BIPS"
    return "OTHER"

def _hemi_of(name: str) -> str | None:
    if name.endswith('_L'):
        return 'L'
    if name.endswith('_R'):
        return 'R'
    return None

def _build_overlay_toggle_js(syn_idx: list[int], mid_idx: list[int], cen_idx: list[int], ovl_idx: list[int] = None) -> str:
    """Return JavaScript that toggles Plotly trace visibility via checkboxes."""
    if ovl_idx is None:
        ovl_idx = []
    return f"""
    function toggleTraces() {{
        var gd = document.querySelectorAll('.plotly-graph-div')[0];
        var synIdx = {syn_idx};
        var midIdx = {mid_idx};
        var cenIdx = {cen_idx};
        var ovlIdx = {ovl_idx};
        var synOn = document.getElementById('chk_syn') ? document.getElementById('chk_syn').checked : true;
        var midOn = document.getElementById('chk_mid') ? document.getElementById('chk_mid').checked : true;
        var cenOn = document.getElementById('chk_cen') ? document.getElementById('chk_cen').checked : true;
        var ovlOn = document.getElementById('chk_ovl') ? document.getElementById('chk_ovl').checked : true;
        var update = {{}};
        var indices = [];
        var vis = [];
        synIdx.forEach(function(i) {{ indices.push(i); vis.push(synOn); }});
        midIdx.forEach(function(i) {{ indices.push(i); vis.push(midOn); }});
        cenIdx.forEach(function(i) {{ indices.push(i); vis.push(cenOn); }});
        ovlIdx.forEach(function(i) {{ indices.push(i); vis.push(ovlOn); }});
        if (indices.length > 0) {{
            Plotly.restyle(gd, {{'visible': vis}}, indices);
        }}
    }}
    """


def _build_color_map(valid_names):
    """Assign group-based colors (L/R share the same hue):
    MOS: #4D9221, VS: #D14900, MOT: #5E3C99, HS: #C51B7D, BIPS: black.
    HS shades split across HSN/HSE/HSS; VS shades across VS1-4; others share their base.
    """

    BASES = {
        'MOT': '#5E3C99',
        'MOS': '#4D9221',
        'HS':  '#C51B7D',
        'VS':  '#D14900',
        'BIPS': '#000000',
    }

    def base_id(name: str) -> str:
        return name.split('_')[0]

    ids_by_group: dict[str, set[str]] = {'MOT': set(), 'MOS': set(), 'HS': set(), 'VS': set(), 'BIPS': set()}
    for n in (base_id(x) for x in valid_names):
        grp = _get_group(n)
        if grp in ids_by_group:
            ids_by_group[grp].add(n)

    def make_shades(base_hex: str, labels: list[str]) -> dict[str, str]:
        if not labels:
            return {}
        r, g, b = mpl_colors.to_rgb(base_hex)
        h, l, s = rgb_to_hls(r, g, b)
        n = max(1, len(labels))
        l_min = max(0.22, l * 0.65)
        l_max = min(0.92, l * 1.15)
        shades = {}
        for idx, lab in enumerate(sorted(labels)):
            t = 0.0 if n == 1 else idx / (n - 1)
            li = l_min + t * (l_max - l_min)
            ri, gi, bi = hls_to_rgb(h, li, s)
            shades[lab] = mpl_colors.to_hex((ri, gi, bi))
        return shades

    mot_shades = make_shades(BASES['MOT'], sorted(ids_by_group['MOT']))
    mos_shades = make_shades(BASES['MOS'], sorted(ids_by_group['MOS']))
    hs_shades = make_shades(BASES['HS'], sorted(ids_by_group['HS']))
    vs_shades = make_shades(BASES['VS'], sorted(ids_by_group['VS']))
    bips_shades = {lab: BASES['BIPS'] for lab in ids_by_group['BIPS']}

    color_map = {}
    for name in valid_names:
        base = base_id(name)
        if base in mot_shades:
            color_map[name] = mot_shades[base]
        elif base in mos_shades:
            color_map[name] = mos_shades[base]
        elif base in hs_shades:
            color_map[name] = hs_shades[base]
        elif base in vs_shades:
            color_map[name] = vs_shades[base]
        elif base in bips_shades:
            color_map[name] = bips_shades[base]
        else:
            color_map[name] = '#888888'
    return color_map

def _decimate_mesh_for_plot(verts: np.ndarray, faces: np.ndarray, max_faces: int | None = None, seed: int = 42):
    """Return decimated x,y,z and i,j,k lists for Plotly Mesh3d.
    Randomly sample faces if over limit and rebuild a unique vertex list.
    """
    if max_faces is None or len(faces) <= max_faces:
        # Use original arrays directly
        return verts[:, 0], verts[:, 1], verts[:, 2], faces[:, 0], faces[:, 1], faces[:, 2]

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(faces), size=max_faces, replace=False)
    faces_sel = faces[sel]

    # Build unique vertex mapping
    unique_idx = np.unique(faces_sel.reshape(-1))
    idx_map = {int(old): i for i, old in enumerate(unique_idx)}
    verts_sel = verts[unique_idx]
    # Remap faces
    remap = np.vectorize(lambda a: idx_map[int(a)])
    faces_remap = remap(faces_sel)
    return (
        verts_sel[:, 0], verts_sel[:, 1], verts_sel[:, 2],
        faces_remap[:, 0], faces_remap[:, 1], faces_remap[:, 2]
    )

def _mesh_trace_from_neuron(neuron, name, color, opacity=0.18, max_faces_per_mesh: int | None = None):
    verts = neuron.vertices
    faces = neuron.faces
    grp = _get_group(name)

    x, y, z, i, j, k = _decimate_mesh_for_plot(verts, faces, max_faces=max_faces_per_mesh)
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        name=name,
        color=color,
        opacity=opacity,
        legendgroup=grp,
        hoverinfo='name',
        lighting=dict(ambient=0.55, diffuse=0.9, specular=0.25, roughness=0.45, fresnel=0.15),
        lightposition=dict(x=2000, y=2000, z=3000),
        flatshading=False
    )

def _overlap_traces_for_mot_mos(all_results, threshold_um, color_map,
                                max_faces_per_pair=None, max_total_faces=None,
                                dedup_vertices=False, round_decimals=3,
                                overlap_opacity=0.22, add_red_contours=False,
                                hemi_filter: str | None = None,
                                source_group_filter: set[str] | None = None,
                                target_group_filter: set[str] | None = None,
                                include_names: set[str] | None = None):
    """Create Mesh3d traces for overlap patches where source is MOT/MOS and target is HS/VS.
    Color by the target neuron's color.
    """
    traces = []
    results = all_results.get(threshold_um, {})
    total_faces_accum = 0

    for pair_key, result in results.items():
        try:
            source, target = pair_key.split('→')
        except ValueError:
            continue

        if _get_group(source) not in ("MOT", "MOS"):
            continue
        if _get_group(target) not in ("VS", "HS"):
            continue
        # Optional filter: only include overlaps whose source group is in the set
        if source_group_filter is not None:
            if _get_group(source) not in source_group_filter:
                continue
        # Optional filter: only include specific target groups
        if target_group_filter is not None:
            if _get_group(target) not in target_group_filter:
                continue
        # If filtering by hemisphere, require BOTH source and target to be in this hemi
        if hemi_filter is not None:
            if _hemi_of(source) != hemi_filter or _hemi_of(target) != hemi_filter:
                continue
        # Optional: include only if both names are in include_names set
        if include_names is not None and (source not in include_names or target not in include_names):
            continue

        area, geo = (result if isinstance(result, tuple) else (result, None))
        if not geo or area <= 0:
            continue
        face_data = geo.get('face_data', [])
        if not face_data:
            continue

        # Assemble triangles
        xs, ys, zs, I, J, K = [], [], [], [], [], []
        tri_count = 0
        idx_map = {}

        for face_idx, fd in enumerate(face_data):
            if max_faces_per_pair is not None and tri_count >= max_faces_per_pair:
                break
            if max_total_faces is not None and total_faces_accum >= max_total_faces:
                break
            verts = fd.get('vertices')
            if verts is None or len(verts) != 3:
                continue
            if dedup_vertices:
                tri_indices = []
                for v in verts:
                    key = (round(float(v[0]), round_decimals), round(float(v[1]), round_decimals), round(float(v[2]), round_decimals))
                    if key in idx_map:
                        tri_indices.append(idx_map[key])
                    else:
                        idx = len(xs)
                        xs.append(key[0]); ys.append(key[1]); zs.append(key[2])
                        idx_map[key] = idx
                        tri_indices.append(idx)
                I.append(tri_indices[0]); J.append(tri_indices[1]); K.append(tri_indices[2])
            else:
                base = len(xs)
                xs.extend([verts[0][0], verts[1][0], verts[2][0]])
                ys.extend([verts[0][1], verts[1][1], verts[2][1]])
                zs.extend([verts[0][2], verts[1][2], verts[2][2]])
                I.append(base + 0); J.append(base + 1); K.append(base + 2)
            tri_count += 1
            total_faces_accum += 1

        if tri_count == 0:
            continue

        color = color_map.get(target, '#ff0000')
        traces.append(go.Mesh3d(
            x=xs, y=ys, z=zs, i=I, j=J, k=K,
            name=f"Overlap {source} ↔ {target}",
            color=color,
            opacity=overlap_opacity,
            legendgroup=f"overlap_{_get_group(target)}",
            hoverinfo='name',
            showscale=False,
            lighting=dict(ambient=0.35, diffuse=0.95, specular=0.4, roughness=0.35, fresnel=0.1),
            lightposition=dict(x=-1500, y=-1200, z=2500),
            flatshading=False
        ))

        if add_red_contours:
            # Build red contour edges for this patch
            ex, ey, ez = [], [], []
            for fd in face_data:
                verts = fd.get('vertices')
                if verts is None or len(verts) != 3:
                    continue
                # edge v0->v1
                ex += [verts[0][0], verts[1][0], None]
                ey += [verts[0][1], verts[1][1], None]
                ez += [verts[0][2], verts[1][2], None]
                # edge v1->v2
                ex += [verts[1][0], verts[2][0], None]
                ey += [verts[1][1], verts[2][1], None]
                ez += [verts[1][2], verts[2][2], None]
                # edge v2->v0
                ex += [verts[2][0], verts[0][0], None]
                ey += [verts[2][1], verts[0][1], None]
                ez += [verts[2][2], verts[0][2], None]
            traces.append(go.Scatter3d(
                x=ex, y=ey, z=ez, mode='lines', name=f"Overlap edges {source}↔{target}",
                line=dict(color='red', width=2.5), opacity=0.9,
                hoverinfo='skip', legendgroup=f"overlap_{_get_group(target)}"
            ))

    return traces

def build_mesh_and_overlap_html(neurons, all_results, valid_names, thresholds_microns, output_dir,
                                max_faces_per_mesh: int | None = 150000,
                                neuron_mesh_opacity: float = 0.22,
                                filename_suffix: str = "",
                                source_group_filter: set[str] | None = None,
                                target_group_filter: set[str] | None = None,
                                include_names: set[str] | None = None,
                                synapses_df: pd.DataFrame | None = None,
                                synapse_marker_size: float = 4.0):
    """Visualize all neuron meshes with group colors and overlay only MOT/MOS overlaps
    colored by the VS/HS target, and save to HTML.
    """
    os.makedirs(output_dir, exist_ok=True)
    color_map = _build_color_map(valid_names)

    # Mesh traces for all neurons
    mesh_traces = []
    for name in valid_names:
        neuron = neurons.get(name)
        if neuron is None:
            continue
        if include_names is not None and name not in include_names:
            continue
        mesh_traces.append(_mesh_trace_from_neuron(
            neuron, name, color_map.get(name, '#888888'),
            opacity=neuron_mesh_opacity, max_faces_per_mesh=max_faces_per_mesh
        ))

    # Overlap traces (take the first/only threshold)
    threshold_um = thresholds_microns[0]
    overlap_traces = _overlap_traces_for_mot_mos(all_results, threshold_um, color_map,
                                                 max_faces_per_pair=None, max_total_faces=None, dedup_vertices=False,
                                                 overlap_opacity=neuron_mesh_opacity, add_red_contours=True,
                                                 hemi_filter=None, source_group_filter=source_group_filter,
                                                 target_group_filter=target_group_filter,
                                                 include_names=include_names)

    # Plot midpoints of all close vertex pairs (as small red hollow markers)
    results = all_results.get(threshold_um, {})
    mid_x, mid_y, mid_z = [], [], []
    max_pts = 80000  # safety cap
    for pair_key, result in results.items():
        try:
            source, target = pair_key.split('→')
        except ValueError:
            continue
        if _get_group(source) not in ("MOT", "MOS") or _get_group(target) not in ("VS", "HS"):
            continue
        if source_group_filter is not None and _get_group(source) not in source_group_filter:
            continue
        area, geo = (result if isinstance(result, tuple) else (result, None))
        if not geo or area <= 0:
            continue
        va = geo.get('close_vertices_meshA')
        vb = geo.get('close_vertices_meshB')
        if va is None or vb is None or len(va) == 0 or len(vb) == 0:
            continue
        if include_names is not None and (source not in include_names or target not in include_names):
            continue
        n = min(len(va), len(vb))
        # sample if too many
        step = max(1, int(np.ceil(n / max_pts)))
        for i in range(0, n, step):
            m = (va[i] + vb[i]) / 2.0
            mid_x.append(float(m[0])); mid_y.append(float(m[1])); mid_z.append(float(m[2]))
            if len(mid_x) >= max_pts:
                break
        if len(mid_x) >= max_pts:
            break

    if mid_x:
        mesh_traces.append(go.Scatter3d(
            x=mid_x, y=mid_y, z=mid_z, mode='markers',
            name='Contact vertex midpoints',
            marker=dict(size=2.2, color='white', opacity=0.98, symbol='circle-open',
                        line=dict(color='red', width=2)),
            hoverinfo='skip'
        ))

    syn_traces = []
    if synapses_df is not None:
        syn_traces = build_synapse_traces(
            synapses_df,
            include_names=include_names,  # only neurons drawn as meshes
            source_group_filter={"MOT","MOS"},
            target_group_filter={"HS","VS"},
            hemi_filter=None
        )
    fig = go.Figure(data=mesh_traces + overlap_traces + syn_traces)
    # Add Top-N patch centroids as red hollow circles (bigger)
    try:
        threshold_um = thresholds_microns[0]
        results = all_results.get(threshold_um, {})
        for pair_key, result in results.items():
            source, target = pair_key.split('→')
            if include_names is not None and (source not in include_names or target not in include_names):
                continue
            if _get_group(source) not in ("MOT","MOS") or _get_group(target) not in ("HS","VS"):
                continue
            area, geo = (result if isinstance(result, tuple) else (result, None))
            if not geo or area <= 0:
                continue
            fds = geo.get('face_data', []) or []
            if not fds:
                continue
            topN = sorted(fds, key=lambda fd: fd.get('area', 0), reverse=True)[:N_TOP_PATCHES]
            cx = [fd.get('centroid',[None,None,None])[0] for fd in topN]
            cy = [fd.get('centroid',[None,None,None])[1] for fd in topN]
            cz = [fd.get('centroid',[None,None,None])[2] for fd in topN]
            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz, mode='markers', name=f"Top-N centroids → {target}",
                marker=dict(size=7.5, color='white', opacity=0.98, symbol='circle-open',
                            line=dict(color='red', width=3)),
                hoverinfo='name', legendgroup=f"centroids_{_get_group(target)}"
            ))
    except Exception:
        pass
    fig.update_layout(
        title=f"Meshes and Overlap Patches — threshold {threshold_um} um{(' ' + filename_suffix) if filename_suffix else ''}",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.1))
        ),
        legend=dict(itemsizing='constant', bgcolor='rgba(255,255,255,0.6)'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        width=1400,
        height=1100,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    suffix = ("_" + filename_suffix) if filename_suffix else ""
    html_path = os.path.join(output_dir, f"meshes_and_overlaps{suffix}_{threshold_um}um.html")

    # --- Toggle checkboxes for synapses, contact midpoints, and centroids ---
    syn_idx = [i for i, t in enumerate(fig.data) if getattr(t, 'name', '') == 'Synapses']
    mid_idx = [i for i, t in enumerate(fig.data) if getattr(t, 'name', '') == 'Contact vertex midpoints']
    cen_idx = [i for i, t in enumerate(fig.data) if 'Top-N centroids' in (getattr(t, 'name', '') or '')]
    ovl_idx = [i for i, t in enumerate(fig.data) if (getattr(t, 'name', '') or '').startswith('Overlap ')]
    overlay_idx = sorted(syn_idx + mid_idx + cen_idx + ovl_idx)
    if overlay_idx:
        raw_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        checkbox_js = _build_overlay_toggle_js(syn_idx, mid_idx, cen_idx, ovl_idx)
        full_html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>
<div style="position:fixed;top:10px;left:10px;z-index:999;background:rgba(255,255,255,0.92);
padding:10px 14px;border-radius:8px;border:1px solid #aaa;font-family:Arial,sans-serif;font-size:13px;">
  <b>Overlay toggles</b><br>
  {'<label><input type="checkbox" id="chk_syn" checked onchange="toggleTraces()"> Synapses</label><br>' if syn_idx else ''}
  {'<label><input type="checkbox" id="chk_mid" checked onchange="toggleTraces()"> Contact midpoints</label><br>' if mid_idx else ''}
  {'<label><input type="checkbox" id="chk_cen" checked onchange="toggleTraces()"> Top-N centroids</label><br>' if cen_idx else ''}
  {'<label><input type="checkbox" id="chk_ovl" checked onchange="toggleTraces()"> Overlap areas</label>' if ovl_idx else ''}
</div>
{raw_html}
<script>{checkbox_js}</script>
</body></html>"""
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
    else:
        fig.write_html(html_path)
    print(f"Saved 3D mesh + overlap visualization: {html_path}")

def build_mesh_and_overlap_html_lite(neurons, all_results, valid_names, thresholds_microns, output_dir,
                                     max_faces_per_pair=10000, max_total_faces=150000,
                                     dedup_vertices=True, round_decimals=3,
                                     max_faces_per_mesh=20000,
                                     neuron_mesh_opacity: float = 0.22,
                                     hemi_filter: str | None = None,
                                     mot_mos_faces: int | None = 40000,
                                     hs_vs_faces: int | None = 15000,
                                     points_per_neuron: int = 40000,
                                     filename_suffix: str = "",
                                     source_group_filter: set[str] | None = None,
                                     target_group_filter: set[str] | None = None,
                                     include_names: set[str] | None = None,
                                     synapses_df: pd.DataFrame | None = None,
                                     synapse_marker_size: float = 3.2):
    """Build a lighter-weight HTML by limiting faces per pair and overall, with optional vertex dedup."""
    os.makedirs(output_dir, exist_ok=True)
    color_map = _build_color_map(valid_names)
    mesh_traces = []
    if hemi_filter is None:
        # Default LITE view: decimated meshes
        for name in valid_names:
            neuron = neurons.get(name)
            if neuron is None:
                continue
            if include_names is not None and name not in include_names:
                continue
            grp = _get_group(name)
            per_mesh_cap = max_faces_per_mesh
            if grp in ("MOT", "MOS"):
                per_mesh_cap = mot_mos_faces
            elif grp in ("HS", "VS"):
                per_mesh_cap = hs_vs_faces
            mesh_traces.append(_mesh_trace_from_neuron(neuron, name, color_map.get(name, '#888888'), opacity=neuron_mesh_opacity, max_faces_per_mesh=per_mesh_cap))
    else:
        # Hemisphere LITE view: replace meshes with point-clouds for neurons in the selected hemisphere
        rng = np.random.default_rng(42)
        for name in valid_names:
            if _hemi_of(name) != hemi_filter:
                continue
            neuron = neurons.get(name)
            if neuron is None or len(neuron.vertices) == 0:
                continue
            if include_names is not None and name not in include_names:
                continue
            verts = neuron.vertices
            n = len(verts)
            k = min(points_per_neuron, n)
            idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
            v = verts[idx]
            mesh_traces.append(go.Scatter3d(
                x=v[:,0], y=v[:,1], z=v[:,2],
                mode='markers',
                name=name,
                marker=dict(size=1.4, color=color_map.get(name, '#888888'), opacity=0.35),
                legendgroup=_get_group(name),
                hoverinfo='name'
            ))

    threshold_um = thresholds_microns[0]
    overlap_traces = _overlap_traces_for_mot_mos(
        all_results, threshold_um, color_map,
        max_faces_per_pair=max_faces_per_pair,
        max_total_faces=max_total_faces,
        dedup_vertices=dedup_vertices,
        round_decimals=round_decimals,
        overlap_opacity=neuron_mesh_opacity,
        add_red_contours=True,
    hemi_filter=hemi_filter,
    source_group_filter=source_group_filter,
    target_group_filter=target_group_filter,
    include_names=include_names
    )

    # Add midpoints of close vertex pairs (small red hollow markers)
    results = all_results.get(threshold_um, {})
    mid_x, mid_y, mid_z = [], [], []
    max_pts = 60000  # a bit lower for lite
    for pair_key, result in results.items():
        try:
            source, target = pair_key.split('→')
        except ValueError:
            continue
        if _get_group(source) not in ("MOT", "MOS") or _get_group(target) not in ("VS", "HS"):
            continue
        if source_group_filter is not None and _get_group(source) not in source_group_filter:
            continue
        if hemi_filter is not None:
            # Only keep midpoints for overlaps where BOTH neurons are in the selected hemisphere
            if _hemi_of(source) != hemi_filter or _hemi_of(target) != hemi_filter:
                continue
        area, geo = (result if isinstance(result, tuple) else (result, None))
        if not geo or area <= 0:
            continue
        va = geo.get('close_vertices_meshA')
        vb = geo.get('close_vertices_meshB')
        if va is None or vb is None or len(va) == 0 or len(vb) == 0:
            continue
        if include_names is not None and (source not in include_names or target not in include_names):
            continue
        n = min(len(va), len(vb))
        step = max(1, int(np.ceil(n / max_pts)))
        for i in range(0, n, step):
            m = (va[i] + vb[i]) / 2.0
            mid_x.append(float(m[0])); mid_y.append(float(m[1])); mid_z.append(float(m[2]))
            if len(mid_x) >= max_pts:
                break
        if len(mid_x) >= max_pts:
            break

    if mid_x:
        mesh_traces.append(go.Scatter3d(
            x=mid_x, y=mid_y, z=mid_z, mode='markers',
            name='Contact vertex midpoints',
            marker=dict(size=2.0, color='white', opacity=0.98, symbol='circle-open',
                        line=dict(color='red', width=2)),
            hoverinfo='skip'
        ))

    syn_traces = []
    if synapses_df is not None:
        syn_traces = build_synapse_traces(
            synapses_df,
            include_names=include_names,  # only neurons drawn as meshes (hemi filtered separately)
            source_group_filter={"MOT","MOS"},
            target_group_filter={"HS","VS"},
            hemi_filter=hemi_filter
        )
    fig = go.Figure(data=mesh_traces + overlap_traces + syn_traces)
    # Add Top-N patch centroids per pair as red hollow circles (bigger)
    try:
        threshold_um = thresholds_microns[0]
        results = all_results.get(threshold_um, {})
        for pair_key, result in results.items():
            source, target = pair_key.split('→')
            if include_names is not None and (source not in include_names or target not in include_names):
                continue
            if hemi_filter is not None and (_hemi_of(source) != hemi_filter or _hemi_of(target) != hemi_filter):
                continue
            if _get_group(source) not in ("MOT","MOS") or _get_group(target) not in ("HS","VS"):
                continue
            area, geo = (result if isinstance(result, tuple) else (result, None))
            if not geo or area <= 0:
                continue
            fds = geo.get('face_data', []) or []
            if not fds:
                continue
            topN = sorted(fds, key=lambda fd: fd.get('area', 0), reverse=True)[:N_TOP_PATCHES]
            cx = [fd.get('centroid',[None,None,None])[0] for fd in topN]
            cy = [fd.get('centroid',[None,None,None])[1] for fd in topN]
            cz = [fd.get('centroid',[None,None,None])[2] for fd in topN]
            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz, mode='markers', name=f"Top-N centroids → {target}",
                marker=dict(size=7.5, color='white', opacity=0.98, symbol='circle-open',
                            line=dict(color='red', width=3)),
                hoverinfo='name', legendgroup=f"centroids_{_get_group(target)}"
            ))
    except Exception:
        pass
    fig.update_layout(
        title=f"Meshes and Overlap (LITE) — threshold {threshold_um} um",
        scene=dict(
            xaxis=dict(title='X', showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(title='Y', showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(title='Z', showbackground=False, showgrid=False, zeroline=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.1))
        ),
        legend=dict(itemsizing='constant', bgcolor='rgba(255,255,255,0.6)'),
        paper_bgcolor='white', plot_bgcolor='white', width=1400, height=1100,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    # --- Toggle checkboxes for synapses, contact midpoints, and centroids ---
    syn_idx = [i for i, t in enumerate(fig.data) if getattr(t, 'name', '') == 'Synapses']
    mid_idx = [i for i, t in enumerate(fig.data) if getattr(t, 'name', '') == 'Contact vertex midpoints']
    cen_idx = [i for i, t in enumerate(fig.data) if 'Top-N centroids' in (getattr(t, 'name', '') or '')]
    ovl_idx = [i for i, t in enumerate(fig.data) if (getattr(t, 'name', '') or '').startswith('Overlap ')]

    hemi_suffix = f"_hemi_{hemi_filter}" if hemi_filter else ""
    extra_suffix = ("_" + filename_suffix) if filename_suffix else ""
    html_path = os.path.join(output_dir, f"meshes_and_overlaps_LITE{hemi_suffix}{extra_suffix}_{threshold_um}um.html")

    overlay_idx = sorted(syn_idx + mid_idx + cen_idx + ovl_idx)
    if overlay_idx:
        # Write HTML with custom checkbox controls
        raw_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        checkbox_js = _build_overlay_toggle_js(syn_idx, mid_idx, cen_idx, ovl_idx)
        full_html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>
<div style="position:fixed;top:10px;left:10px;z-index:999;background:rgba(255,255,255,0.92);
padding:10px 14px;border-radius:8px;border:1px solid #aaa;font-family:Arial,sans-serif;font-size:13px;">
  <b>Overlay toggles</b><br>
  {'<label><input type="checkbox" id="chk_syn" checked onchange="toggleTraces()"> Synapses</label><br>' if syn_idx else ''}
  {'<label><input type="checkbox" id="chk_mid" checked onchange="toggleTraces()"> Contact midpoints</label><br>' if mid_idx else ''}
  {'<label><input type="checkbox" id="chk_cen" checked onchange="toggleTraces()"> Top-N centroids</label><br>' if cen_idx else ''}
  {'<label><input type="checkbox" id="chk_ovl" checked onchange="toggleTraces()"> Overlap areas</label>' if ovl_idx else ''}
</div>
{raw_html}
<script>{checkbox_js}</script>
</body></html>"""
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
    else:
        fig.write_html(html_path)
    print(f"Saved LITE 3D mesh + overlap visualization: {html_path}")

def build_pointcloud_and_overlap_html(neurons, all_results, valid_names, thresholds_microns, output_dir,
                                      points_per_neuron=20000, patch_mode='points',
                                      max_patch_faces=60000,
                                      filename_suffix: str = "",
                                      source_group_filter: set[str] | None = None,
                                      target_group_filter: set[str] | None = None,
                                      include_names: set[str] | None = None,
                                      synapses_df: pd.DataFrame | None = None,
                                      synapse_marker_size: float = 3.0):
    """Build a point-cloud based HTML: sample vertices per neuron, render overlaps as points or limited meshes."""
    os.makedirs(output_dir, exist_ok=True)
    color_map = _build_color_map(valid_names)
    rng = np.random.default_rng(42)

    # Point clouds for all neurons
    traces = []
    for name in valid_names:
        neuron = neurons.get(name)
        if neuron is None or len(neuron.vertices) == 0:
            continue
        if include_names is not None and name not in include_names:
            continue
        verts = neuron.vertices
        n = len(verts)
        k = min(points_per_neuron, n)
        idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
        v = verts[idx]
        traces.append(go.Scatter3d(
            x=v[:,0], y=v[:,1], z=v[:,2],
            mode='markers',
            name=name,
            marker=dict(size=1.2, color=color_map.get(name, '#888888'), opacity=0.35),
            legendgroup=_get_group(name),
            hoverinfo='name'
        ))

    # Overlaps as points (triangle vertices) or capped meshes
    threshold_um = thresholds_microns[0]
    results = all_results.get(threshold_um, {})
    faces_added = 0
    for pair_key, result in results.items():
        try:
            source, target = pair_key.split('→')
        except ValueError:
            continue
        if _get_group(source) not in ("MOT", "MOS") or _get_group(target) not in ("VS", "HS"):
            continue
        if source_group_filter is not None and _get_group(source) not in source_group_filter:
            continue
        if target_group_filter is not None and _get_group(target) not in target_group_filter:
            continue
        if include_names is not None and (source not in include_names or target not in include_names):
            continue
        if source_group_filter is not None and _get_group(source) not in source_group_filter:
            continue
        area, geo = (result if isinstance(result, tuple) else (result, None))
        if not geo or area <= 0:
            continue
        face_data = geo.get('face_data', [])
        if not face_data:
            continue
        color = color_map.get(target, '#ff0000')

        if patch_mode == 'mesh':
            xs, ys, zs, I, J, K = [], [], [], [], [], []
            for fd in face_data:
                if max_patch_faces is not None and faces_added >= max_patch_faces:
                    break
                verts = fd.get('vertices')
                if verts is None or len(verts) != 3:
                    continue
                base = len(xs)
                xs.extend([verts[0][0], verts[1][0], verts[2][0]])
                ys.extend([verts[0][1], verts[1][1], verts[2][1]])
                zs.extend([verts[0][2], verts[1][2], verts[2][2]])
                I.append(base+0); J.append(base+1); K.append(base+2)
                faces_added += 1
            if len(I) > 0:
                traces.append(go.Mesh3d(
                    x=xs, y=ys, z=zs, i=I, j=J, k=K,
                    name=f"Overlap {source} ↔ {target}",
                    color='red', opacity=0.9,
                    legendgroup=f"overlap_{_get_group(target)}",
                    hoverinfo='name', showscale=False
                ))
        else:
            # points: plot all triangle vertices (or their centroids) for speed
            px, py, pz = [], [], []
            for fd in face_data:
                if max_patch_faces is not None and faces_added >= max_patch_faces:
                    break
                verts = fd.get('vertices')
                if verts is None or len(verts) != 3:
                    continue
                # use vertices for crisper patch outline
                px.extend([verts[0][0], verts[1][0], verts[2][0]])
                py.extend([verts[0][1], verts[1][1], verts[2][1]])
                pz.extend([verts[0][2], verts[1][2], verts[2][2]])
                faces_added += 1
            if px:
                # Layered markers to approximate dual-color ring:
                # 1) Outer red hollow ring (smaller than before)
                traces.append(go.Scatter3d(
                    x=px, y=py, z=pz, mode='markers',
                    name=f"Overlap {source} ↔ {target} (outer)",
                    marker=dict(size=3.2, color='white', opacity=0.98, symbol='circle-open',
                                line=dict(color='red', width=3)),
                    legendgroup=f"overlap_{_get_group(target)}",
                    hoverinfo='skip', showlegend=False
                ))
                # 2) Inner ring with MOT/MOS (source) color
                traces.append(go.Scatter3d(
                    x=px, y=py, z=pz, mode='markers',
                    name=f"Overlap {source} ↔ {target} (inner)",
                    marker=dict(size=2.4, color='white', opacity=0.98, symbol='circle-open',
                                line=dict(color=color_map.get(source, '#1f77b4'), width=2)),
                    legendgroup=f"overlap_{_get_group(target)}",
                    hoverinfo='skip', showlegend=False
                ))
                # 3) Tiny filled dot with target color in the center
                traces.append(go.Scatter3d(
                    x=px, y=py, z=pz, mode='markers',
                    name=f"Overlap {source} ↔ {target}",
                    marker=dict(size=1.2, color=color_map.get(target, '#ff0000'), opacity=0.98, symbol='circle'),
                    legendgroup=f"overlap_{_get_group(target)}",
                    hoverinfo='name'
                ))

    # Remove overall centroid markers per request

    # synapses
    if synapses_df is not None:
        syn_traces = build_synapse_traces(
            synapses_df, include_names=include_names,
            source_group_filter={"MOT","MOS"},
            target_group_filter={"HS","VS"},
            hemi_filter=None
        )
        traces.extend(syn_traces)

    # Add Top-N patch centroids per pair as red hollow circles (bigger) to point-cloud figure
    try:
        threshold_um = thresholds_microns[0]
        results = all_results.get(threshold_um, {})
        for pair_key, result in results.items():
            source, target = pair_key.split('→')
            if include_names is not None and (source not in include_names or target not in include_names):
                continue
            if _get_group(source) not in ("MOT","MOS") or _get_group(target) not in ("HS","VS"):
                continue
            area, geo = (result if isinstance(result, tuple) else (result, None))
            if not geo or area <= 0:
                continue
            fds = geo.get('face_data', []) or []
            if not fds:
                continue
            topN = sorted(fds, key=lambda fd: fd.get('area', 0), reverse=True)[:N_TOP_PATCHES]
            cx = [fd.get('centroid',[None,None,None])[0] for fd in topN]
            cy = [fd.get('centroid',[None,None,None])[1] for fd in topN]
            cz = [fd.get('centroid',[None,None,None])[2] for fd in topN]
            traces.append(go.Scatter3d(
                x=cx, y=cy, z=cz, mode='markers', name=f"Top-N centroids → {target}",
                marker=dict(size=7.5, color='white', opacity=0.98, symbol='circle-open',
                            line=dict(color='red', width=3)),
                hoverinfo='name', legendgroup=f"centroids_{_get_group(target)}"
            ))
    except Exception:
        pass
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Point-cloud meshes + overlaps ({patch_mode}) — threshold {threshold_um} um",
        scene=dict(
            xaxis=dict(title='X', showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(title='Y', showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(title='Z', showbackground=False, showgrid=False, zeroline=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.1))
        ),
        legend=dict(itemsizing='constant', bgcolor='rgba(255,255,255,0.6)'),
        paper_bgcolor='white', plot_bgcolor='white', width=1400, height=1100,
        margin=dict(l=0, r=0, t=60, b=0)
    )

    suffix = ("_" + filename_suffix) if filename_suffix else ""
    html_path = os.path.join(output_dir, f"meshes_and_overlaps_POINTCLOUD{suffix}_{threshold_um}um.html")

    # --- Toggle checkboxes for synapses, contact midpoints, and centroids ---
    syn_idx = [i for i, t in enumerate(fig.data) if getattr(t, 'name', '') == 'Synapses']
    mid_idx = [i for i, t in enumerate(fig.data) if getattr(t, 'name', '') == 'Contact vertex midpoints']
    cen_idx = [i for i, t in enumerate(fig.data) if 'Top-N centroids' in (getattr(t, 'name', '') or '')]
    ovl_idx = [i for i, t in enumerate(fig.data) if (getattr(t, 'name', '') or '').startswith('Overlap ')]
    overlay_idx = sorted(syn_idx + mid_idx + cen_idx + ovl_idx)
    if overlay_idx:
        raw_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        checkbox_js = _build_overlay_toggle_js(syn_idx, mid_idx, cen_idx, ovl_idx)
        full_html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>
<div style="position:fixed;top:10px;left:10px;z-index:999;background:rgba(255,255,255,0.92);
padding:10px 14px;border-radius:8px;border:1px solid #aaa;font-family:Arial,sans-serif;font-size:13px;">
  <b>Overlay toggles</b><br>
  {'<label><input type="checkbox" id="chk_syn" checked onchange="toggleTraces()"> Synapses</label><br>' if syn_idx else ''}
  {'<label><input type="checkbox" id="chk_mid" checked onchange="toggleTraces()"> Contact midpoints</label><br>' if mid_idx else ''}
  {'<label><input type="checkbox" id="chk_cen" checked onchange="toggleTraces()"> Top-N centroids</label><br>' if cen_idx else ''}
  {'<label><input type="checkbox" id="chk_ovl" checked onchange="toggleTraces()"> Overlap areas</label>' if ovl_idx else ''}
</div>
{raw_html}
<script>{checkbox_js}</script>
</body></html>"""
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
    else:
        fig.write_html(html_path)
    print(f"Saved point-cloud 3D visualization: {html_path}")

def _wireframe_edges_from_faces(verts: np.ndarray, faces: np.ndarray, max_edges: int = 80000, seed: int = 42):
    rng = np.random.default_rng(seed)
    nfaces = len(faces)
    if nfaces == 0:
        return [], [], []
    # sample faces and draw their three edges
    step = max(1, int(np.ceil((nfaces * 3) / max_edges)))
    xs, ys, zs = [], [], []
    for idx in range(0, nfaces, step):
        f = faces[idx]
        e = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
        for a, b in e:
            xa, ya, za = verts[a]
            xb, yb, zb = verts[b]
            xs += [xa, xb, None]
            ys += [ya, yb, None]
            zs += [za, zb, None]
        if len(xs) // 3 >= max_edges:
            break
    return xs, ys, zs

def build_overlap_wireframe_html_lite(neurons, all_results, valid_names, thresholds_microns, output_dir,
                                      hemi_filter: str | None = None,
                                      filename_suffix: str = "",
                                      include_names: set[str] | None = None,
                                      source_group_filter: set[str] | None = None,
                                      target_group_filter: set[str] | None = None,
                                      synapses_df: pd.DataFrame | None = None,
                                      synapse_marker_size: float = 3.0):
    os.makedirs(output_dir, exist_ok=True)
    color_map = _build_color_map(valid_names)
    traces = []
    # wireframe neurons
    for name in valid_names:
        if include_names is not None and name not in include_names:
            continue
        if hemi_filter is not None and _hemi_of(name) != hemi_filter:
            continue
        neuron = neurons.get(name)
        if neuron is None:
            continue
        xs, ys, zs = _wireframe_edges_from_faces(neuron.vertices, neuron.faces, max_edges=90000)
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines', name=name,
                                   line=dict(color=color_map.get(name, '#888'), width=1),
                                   opacity=0.35, hoverinfo='name'))
    # overlap patches (red with edges only)
    threshold_um = thresholds_microns[0]
    overlap_traces = _overlap_traces_for_mot_mos(all_results, threshold_um, color_map,
                                                 max_faces_per_pair=5000, max_total_faces=60000,
                                                 dedup_vertices=True, round_decimals=3,
                                                 overlap_opacity=0.28, add_red_contours=True,
                                                 hemi_filter=hemi_filter,
                                                 source_group_filter=source_group_filter,
                                                 target_group_filter=target_group_filter,
                                                 include_names=include_names)
    syn_traces = []
    if synapses_df is not None:
        syn_traces = build_synapse_traces(
            synapses_df, include_names=include_names,
            source_group_filter={"MOT","MOS"},
            target_group_filter={"HS","VS"},
            hemi_filter=hemi_filter
        )
    # Add Top-N patch centroids per pair as red hollow circles (bigger) to wireframe
    try:
        threshold_um = thresholds_microns[0]
        results = all_results.get(threshold_um, {})
        for pair_key, result in results.items():
            source, target = pair_key.split('→')
            if include_names is not None and (source not in include_names or target not in include_names):
                continue
            if hemi_filter is not None and (_hemi_of(source) != hemi_filter or _hemi_of(target) != hemi_filter):
                continue
            if _get_group(source) not in ("MOT","MOS") or _get_group(target) not in ("HS","VS"):
                continue
            area, geo = (result if isinstance(result, tuple) else (result, None))
            if not geo or area <= 0:
                continue
            fds = geo.get('face_data', []) or []
            if not fds:
                continue
            topN = sorted(fds, key=lambda fd: fd.get('area', 0), reverse=True)[:N_TOP_PATCHES]
            cx = [fd.get('centroid',[None,None,None])[0] for fd in topN]
            cy = [fd.get('centroid',[None,None,None])[1] for fd in topN]
            cz = [fd.get('centroid',[None,None,None])[2] for fd in topN]
            traces.append(go.Scatter3d(
                x=cx, y=cy, z=cz, mode='markers', name=f"Top-N centroids → {target}",
                marker=dict(size=7.5, color='white', opacity=0.98, symbol='circle-open',
                            line=dict(color='red', width=3)),
                hoverinfo='name', legendgroup=f"centroids_{_get_group(target)}"
            ))
    except Exception:
        pass
    fig = go.Figure(data=traces + overlap_traces + syn_traces)
    fig.update_layout(title=f"Wireframe + Overlaps — {threshold_um} um{(' ' + filename_suffix) if filename_suffix else ''}",
                      scene=dict(aspectmode='data',
                                 xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
                                 yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
                                 zaxis=dict(showbackground=False, showgrid=False, zeroline=False)),
                      width=1400, height=1100,
                      legend=dict(itemsizing='constant', bgcolor='rgba(255,255,255,0.6)'))
    hemi_suffix = f"_hemi_{hemi_filter}" if hemi_filter else ""
    extra_suffix = ("_" + filename_suffix) if filename_suffix else ""
    path = os.path.join(output_dir, f"overlaps_WIREFRAME{hemi_suffix}{extra_suffix}_{threshold_um}um.html")

    # --- Toggle checkboxes for synapses, contact midpoints, and centroids ---
    syn_idx = [i for i, t in enumerate(fig.data) if getattr(t, 'name', '') == 'Synapses']
    mid_idx = [i for i, t in enumerate(fig.data) if getattr(t, 'name', '') == 'Contact vertex midpoints']
    cen_idx = [i for i, t in enumerate(fig.data) if 'Top-N centroids' in (getattr(t, 'name', '') or '')]
    ovl_idx = [i for i, t in enumerate(fig.data) if (getattr(t, 'name', '') or '').startswith('Overlap ')]
    overlay_idx = sorted(syn_idx + mid_idx + cen_idx + ovl_idx)
    if overlay_idx:
        raw_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        checkbox_js = _build_overlay_toggle_js(syn_idx, mid_idx, cen_idx, ovl_idx)
        full_html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>
<div style="position:fixed;top:10px;left:10px;z-index:999;background:rgba(255,255,255,0.92);
padding:10px 14px;border-radius:8px;border:1px solid #aaa;font-family:Arial,sans-serif;font-size:13px;">
  <b>Overlay toggles</b><br>
  {'<label><input type="checkbox" id="chk_syn" checked onchange="toggleTraces()"> Synapses</label><br>' if syn_idx else ''}
  {'<label><input type="checkbox" id="chk_mid" checked onchange="toggleTraces()"> Contact midpoints</label><br>' if mid_idx else ''}
  {'<label><input type="checkbox" id="chk_cen" checked onchange="toggleTraces()"> Top-N centroids</label><br>' if cen_idx else ''}
  {'<label><input type="checkbox" id="chk_ovl" checked onchange="toggleTraces()"> Overlap areas</label>' if ovl_idx else ''}
</div>
{raw_html}
<script>{checkbox_js}</script>
</body></html>"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(full_html)
    else:
        fig.write_html(path)
    print(f"Saved wireframe overlap visualization: {path}")

def build_synapses_only_html(valid_names, output_dir,
                             synapses_df: pd.DataFrame | None,
                             hemi_filter: str | None = None,
                             title_suffix: str = ""):
    os.makedirs(output_dir, exist_ok=True)
    if synapses_df is None or synapses_df.empty:
        print("No synapses to plot for synapses-only page")
        return
    df = synapses_df
    if hemi_filter is not None:
        df = df[(df['pre_hemi'] == hemi_filter) & (df['post_hemi'] == hemi_filter)]
    if df.empty:
        print("No synapses after hemi filter for synapses-only page")
        return
    # Color by pre_group using the requested palette
    palette = {
        'MOT': '#7570B3',  # MNs base
        'MOS': '#7570B3',
        'HS': '#1B9E77',
        'VS': '#D95F02'
    }
    colors = df['pre_group'].map(lambda g: palette.get(g, '#7e1edf'))
    trace = go.Scatter3d(
        x=df['x'], y=df['y'], z=df['z'], mode='markers', name='Synapses',
        marker=dict(size=6.5, color=colors, opacity=0.98, symbol='circle'),
        hovertext=df[['pre_type','post_type','pre_group','post_group']].astype(str).agg(' → '.join, axis=1),
        hoverinfo='text'
    )
    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=f"Synapses only {('(' + title_suffix + ')') if title_suffix else ''}{(' — hemi ' + hemi_filter) if hemi_filter else ''}",
        scene=dict(
            xaxis=dict(title='X', showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(title='Y', showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(title='Z', showbackground=False, showgrid=False, zeroline=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.1))
        ),
        legend=dict(itemsizing='constant', bgcolor='rgba(255,255,255,0.6)'),
        paper_bgcolor='white', plot_bgcolor='white', width=1200, height=900,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    hemi_suffix = f"_hemi_{hemi_filter}" if hemi_filter else ""
    path = os.path.join(output_dir, f"synapses_ONLY{hemi_suffix}.html")
    fig.write_html(path)
    print(f"Saved synapses-only visualization: {path}")

def fetch_synapses_for_neurons(neuron_ids_dict: dict[int, str]) -> pd.DataFrame | None:
    try:
        ids = list(neuron_ids_dict.keys())
        syns = get_synapses(ids)
        if syns is None or len(syns) == 0:
            print("No synapses returned")
            return None
        # ensure coordinate columns (robust detection)
        def find_coord_cols(df: pd.DataFrame):
            # Preferred explicit sets
            candidates = [
                ('pt_x_nm','pt_y_nm','pt_z_nm'),
                ('x_nm','y_nm','z_nm'),
                ('pt_x','pt_y','pt_z'),
                ('x','y','z')
            ]
            for a,b,c in candidates:
                if a in df.columns and b in df.columns and c in df.columns:
                    return (a,b,c)
            # Generic pattern-based detection: try to find a triple with common prefix
            cols = set(df.columns)
            # Look for any *_x, *_y, *_z
            x_cols = [c for c in cols if c.endswith('_x') or c.endswith('x')]
            for xc in x_cols:
                prefix = xc[:-1] if xc.endswith('x') else xc[:-2]
                yc = prefix + ('y' if xc.endswith('x') else '_y')
                zc = prefix + ('z' if xc.endswith('x') else '_z')
                if yc in cols and zc in cols:
                    return (xc, yc, zc)
            return None

        coord = find_coord_cols(syns)
        if coord is None:
            print("Synapse coordinates not found; skipping overlay")
            return None
        cx, cy, cz = coord
        syns = syns.copy()
        # Keep coordinates in the same units as meshes (FlyWire meshes typically in nanometers)
        syns['x'] = syns[cx]
        syns['y'] = syns[cy]
        syns['z'] = syns[cz]

        # map types
        id_to_name = neuron_ids_dict
        syns['pre_type'] = syns['pre'].map(id_to_name)
        syns['post_type'] = syns['post'].map(id_to_name)
        # drop synapses not among our set
        syns = syns.dropna(subset=['pre_type','post_type'])
        # add groups and hemispheres
        syns['pre_group'] = syns['pre_type'].apply(_get_group)
        syns['post_group'] = syns['post_type'].apply(_get_group)
        syns['pre_hemi'] = syns['pre_type'].apply(_hemi_of)
        syns['post_hemi'] = syns['post_type'].apply(_hemi_of)
        print(f"Synapses loaded: {len(syns)} between provided neurons")
        # Save to CSV for verification
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            syn_csv = os.path.join(RESULTS_DIR, "synapses.csv")
            syns.to_csv(syn_csv, index=False)
            print(f"Saved synapse table: {syn_csv}")
        except Exception as _:
            pass
        return syns
    except Exception as e:
        print(f"Synapse fetch failed: {e}")
        return None

def build_synapse_traces(syn_df: pd.DataFrame,
                         include_names: set[str] | None = None,
                         source_group_filter: set[str] | None = None,
                         target_group_filter: set[str] | None = None,
                         hemi_filter: str | None = None,
                         marker_size: float = 2.2) -> list[go.Scatter3d]:
    df = syn_df
    # Restrict to synapses among provided neuron set
    df = df.dropna(subset=['pre_type','post_type'])
    if include_names is not None:
        df = df[df['pre_type'].isin(include_names) & df['post_type'].isin(include_names)]
    # Group filtering: allow MOT/MOS with HS/VS in EITHER direction if both filters provided
    if source_group_filter is not None and target_group_filter is not None:
        mask1 = df['pre_group'].isin(source_group_filter) & df['post_group'].isin(target_group_filter)
        mask2 = df['pre_group'].isin(target_group_filter) & df['post_group'].isin(source_group_filter)
        df = df[mask1 | mask2]
    else:
        if source_group_filter is not None:
            df = df[df['pre_group'].isin(source_group_filter)]
        if target_group_filter is not None:
            df = df[df['post_group'].isin(target_group_filter)]
    if hemi_filter is not None:
        df = df[(df['pre_hemi'] == hemi_filter) & (df['post_hemi'] == hemi_filter)]
    if df.empty:
        return []
    trace = go.Scatter3d(
        x=df['x'], y=df['y'], z=df['z'], mode='markers', name='Synapses',
        # Yellow, smaller, more opaque filled circles
        marker=dict(size=marker_size, color='#FFD400', opacity=0.98, symbol='circle'),
        hoverinfo='skip'
    )
    return [trace]

def compute_contact_participants(all_results, threshold_um: float,
                                 sources: set[str], targets: set[str]) -> set[str]:
    names: set[str] = set()
    results = all_results.get(threshold_um, {})
    for pair_key, res in results.items():
        try:
            src, tgt = pair_key.split('→')
        except ValueError:
            continue
        area = res[0] if isinstance(res, tuple) else res
        if area <= 0:
            continue
        if _get_group(src) in sources and _get_group(tgt) in targets:
            names.add(src)
            names.add(tgt)
    return names

def calculate_large_mesh_overlap(neuronA, neuronB, threshold=100.0):
    """Memory-efficient overlap calculation for very large meshes using sampling"""
    print(f"  Processing large meshes with sampling...")
    
    try:
        meshA = trimesh.Trimesh(vertices=neuronA.vertices, faces=neuronA.faces)
        meshB = trimesh.Trimesh(vertices=neuronB.vertices, faces=neuronB.faces)
        
        # Quick bounding box check
        bounds_A = meshA.bounds
        bounds_B = meshB.bounds
        padding = threshold * 2
        
        for dim in range(3):
            if bounds_A[1][dim] + padding < bounds_B[0][dim] or bounds_A[0][dim] > bounds_B[1][dim] + padding:
                print("  No overlap (bounding box)")
                return 0.0, create_empty_geometric_data()
        
        # Sample vertices to reduce memory usage
        max_vertices_A = min(50000, len(meshA.vertices))
        max_vertices_B = min(100000, len(meshB.vertices))
        
        # Sample vertices from mesh A
        if len(meshA.vertices) > max_vertices_A:
            indices_A = np.random.choice(len(meshA.vertices), max_vertices_A, replace=False)
            sampled_vertices_A = meshA.vertices[indices_A]
        else:
            indices_A = np.arange(len(meshA.vertices))
            sampled_vertices_A = meshA.vertices
        
        # Create KDTree for mesh B (potentially sampled)
        if len(meshB.vertices) > max_vertices_B:
            indices_B = np.random.choice(len(meshB.vertices), max_vertices_B, replace=False)
            tree_vertices_B = meshB.vertices[indices_B]
        else:
            indices_B = np.arange(len(meshB.vertices))
            tree_vertices_B = meshB.vertices
        
        tree = cKDTree(tree_vertices_B)
        
        # Find close vertices
        dists, tree_indices = tree.query(sampled_vertices_A)
        close_mask = dists < threshold
        close_vertices_A_local = np.where(close_mask)[0]
        
        if len(close_vertices_A_local) == 0:
            print("  No close vertices (sampled)")
            return 0.0, create_empty_geometric_data()
        
        # Map back to original indices
        close_vertices_A = indices_A[close_vertices_A_local]
        close_vertices_B = indices_B[tree_indices[close_mask]]
        
        # Find faces containing close vertices (limited sampling)
        close_faces = set()
        max_faces_to_check = min(50000, len(meshA.faces))
        face_indices_to_check = np.random.choice(len(meshA.faces), max_faces_to_check, replace=False)
        
        for face_idx in face_indices_to_check:
            face = meshA.faces[face_idx]
            if any(v in close_vertices_A for v in face):
                close_faces.add(face_idx)
        
        # Calculate area from sampled faces
        total_area = 0.0
        face_data = []
        max_faces_to_store = 1000  # Further limit for large meshes
        
        for i, face_idx in enumerate(close_faces):
            if i >= max_faces_to_store:
                # Estimate remaining area based on average
                if i > 0:
                    avg_area = total_area / i
                    estimated_remaining = avg_area * (len(close_faces) - i)
                    total_area += estimated_remaining
                break
                
            face = meshA.faces[face_idx]
            verts = meshA.vertices[face]
            v1 = verts[1] - verts[0]
            v2 = verts[2] - verts[0]
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            total_area += area
            
            centroid = np.mean(verts, axis=0)
            face_data.append({
                'face_idx': face_idx,
                'vertices': verts.copy(),
                'area': area,
                'centroid': centroid.copy()
            })
        
        # Scale up the area estimate based on sampling ratio
        if len(meshA.faces) > max_faces_to_check:
            scale_factor = len(meshA.faces) / max_faces_to_check
            total_area *= scale_factor
            print(f"  Scaled area estimate by factor {scale_factor:.2f}")
        
        area_um2 = total_area / 1e6
        
        # Create geometric data with sampled information
        geometric_data = {
            'contact_area': area_um2,
            'close_vertices_meshA': meshA.vertices[close_vertices_A[:1000]].copy(),  # Limit stored vertices
            'close_vertices_meshB': meshB.vertices[close_vertices_B[:1000]].copy(),
            'distances': dists[close_mask][:1000].copy(),
            'face_data': face_data,
            'total_area_meshA': meshA.area / 1e6,
            'total_area_meshB': meshB.area / 1e6,
            'centroid_meshA': np.mean(meshA.vertices, axis=0),
            'centroid_meshB': np.mean(meshB.vertices, axis=0),
            'num_vertices_meshA': len(meshA.vertices),
            'num_vertices_meshB': len(meshB.vertices),
            'num_faces_meshA': len(meshA.faces),
            'num_faces_meshB': len(meshB.faces),
            'threshold_used': threshold,
            'sampled': True,
            'sample_ratio_A': len(sampled_vertices_A) / len(meshA.vertices),
            'sample_ratio_B': len(tree_vertices_B) / len(meshB.vertices)
        }
        
        print(f"    # close vertices (sampled): {len(close_vertices_A)}")
        print(f"    # close faces (sampled): {len(close_faces)}")
        print(f"    meshA total area: {meshA.area / 1e6:.4f} um²")
        print(f"    meshB total area: {meshB.area / 1e6:.4f} um²")
        print(f"    Contact area (estimated): {area_um2:.4f} um²")
        
        return area_um2, geometric_data
        
    except MemoryError as e:
        print(f"  Memory error even with sampling: {e}")
        return 0.0, create_empty_geometric_data()
    except Exception as e:
        print(f"  Error in large mesh calculation: {e}")
        return 0.0, create_empty_geometric_data()

def calculate_neuron_overlap_simple(neuronA, neuronB, threshold=100.0):
    """Simplified overlap calculation with detailed geometric data extraction and memory management"""
    print(f"  Processing meshes...")
    
    try:
        # Check if neurons are valid
        if neuronA is None or neuronB is None:
            print("  Error: One or both neurons are None")
            return 0.0, create_empty_geometric_data()
        
        # Check for extremely large mesh sizes and handle them differently
        if len(neuronA.vertices) > 1000000 or len(neuronB.vertices) > 1000000:
            print(f"  Warning: Very large mesh detected (A: {len(neuronA.vertices)}, B: {len(neuronB.vertices)} vertices)")
            print("  Using simplified analysis for memory efficiency")
            # Use a more memory-efficient approach for very large meshes
            return calculate_large_mesh_overlap(neuronA, neuronB, threshold)
        elif len(neuronA.vertices) > 500000 or len(neuronB.vertices) > 500000:
            print(f"  Large mesh detected (A: {len(neuronA.vertices)}, B: {len(neuronB.vertices)} vertices)")
            print("  Using memory-optimized analysis")
        
        meshA = trimesh.Trimesh(vertices=neuronA.vertices, faces=neuronA.faces)
        meshB = trimesh.Trimesh(vertices=neuronB.vertices, faces=neuronB.faces)
        
        # Quick bounding box check
        bounds_A = meshA.bounds
        bounds_B = meshB.bounds
        padding = threshold * 2
        
        for dim in range(3):
            if bounds_A[1][dim] + padding < bounds_B[0][dim] or bounds_A[0][dim] > bounds_B[1][dim] + padding:
                print("  No overlap (bounding box)")
                return 0.0, create_empty_geometric_data()
        
        # Create KDTree for B with memory check
        try:
            tree = cKDTree(meshB.vertices)
        except MemoryError:
            print("  Memory error creating KDTree")
            return 0.0, create_empty_geometric_data()
        
        # Find close vertices in A with memory check
        try:
            dists, indices = tree.query(meshA.vertices)
        except MemoryError:
            print("  Memory error in KDTree query")
            return 0.0, create_empty_geometric_data()
        
        close_vertices_mask = dists < threshold
        close_vertices = np.where(close_vertices_mask)[0]
        
        if len(close_vertices) == 0:
            print("  No close vertices")
            return 0.0, create_empty_geometric_data()
        
        # Find faces containing close vertices
        close_faces = set()
        for face_idx, face in enumerate(meshA.faces):
            if any(v in close_vertices for v in face):
                close_faces.add(face_idx)
        
        # Calculate area and collect geometric data (limit face data to prevent memory issues)
        total_area = 0.0
        face_data = []
        max_faces_to_store = 10000  # Limit stored face data
        
        for i, face_idx in enumerate(close_faces):
            face = meshA.faces[face_idx]
            verts = meshA.vertices[face]
            v1 = verts[1] - verts[0]
            v2 = verts[2] - verts[0]
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            total_area += area
            
            # Only store detailed face data for first N faces to save memory
            if i < max_faces_to_store:
                centroid = np.mean(verts, axis=0)
                face_data.append({
                    'face_idx': face_idx,
                    'vertices': verts.copy(),  # Make a copy to avoid reference issues
                    'area': area,
                    'centroid': centroid.copy()
                })

        # Convert to um²
        area_um2 = total_area / 1e6
        
        # Create simplified geometric data to reduce memory usage
        max_vertices_to_store = 5000
        stored_vertices = min(len(close_vertices), max_vertices_to_store)
        
        geometric_data = {
            'contact_area': area_um2,
            'close_vertices_meshA': meshA.vertices[close_vertices[:stored_vertices]].copy(),
            'close_vertices_meshB': meshB.vertices[indices[close_vertices_mask][:stored_vertices]].copy(),
            'distances': dists[close_vertices_mask][:stored_vertices].copy(),
            'face_data': face_data,
            'total_area_meshA': meshA.area / 1e6,
            'total_area_meshB': meshB.area / 1e6,
            'centroid_meshA': np.mean(meshA.vertices, axis=0),
            'centroid_meshB': np.mean(meshB.vertices, axis=0),
            'num_vertices_meshA': len(meshA.vertices),
            'num_vertices_meshB': len(meshB.vertices),
            'num_faces_meshA': len(meshA.faces),
            'num_faces_meshB': len(meshB.faces),
            'threshold_used': threshold
        }
        
        # Diagnostic prints
        print(f"    # close vertices: {len(close_vertices)}")
        print(f"    # close faces: {len(close_faces)}")
        print(f"    meshA total area: {meshA.area / 1e6:.4f} um²")
        print(f"    meshB total area: {meshB.area / 1e6:.4f} um²")
        print(f"    Contact area: {area_um2:.4f} um²")
        return area_um2, geometric_data
        
    except MemoryError as e:
        print(f"  Memory error: {e}")
        return 0.0, create_empty_geometric_data()
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return 0.0, create_empty_geometric_data()

def create_empty_geometric_data():
    """Create empty geometric data structure for no-overlap cases"""
    return {
        'contact_area': 0.0,
        'close_vertices_meshA': np.array([]),
        'close_vertices_meshB': np.array([]),
        'distances': np.array([]),
        'face_data': [],
        'total_area_meshA': 0.0,
        'total_area_meshB': 0.0,
    'centroid_meshA': np.array([0.0, 0.0, 0.0]),
    'centroid_meshB': np.array([0.0, 0.0, 0.0]),
        'num_vertices_meshA': 0,
        'num_vertices_meshB': 0,
        'num_faces_meshA': 0,
        'num_faces_meshB': 0,
        'threshold_used': 0.0
    }

def load_all_neurons(neuron_ids, lod=50):
    """Load all neurons, recycling meshes from previous results when available."""
    import navis
    neurons = {}
    previous_meshes = _find_previous_meshes()
    recycled = 0
    downloaded = 0

    print(f"Loading {len(neuron_ids)} neurons...")
    if previous_meshes:
        print(f"  Found {len(previous_meshes)} recycled meshes from previous runs")

    for neuron_id, name in tqdm(neuron_ids.items(), desc="Loading neurons"):
        # Try recycling from previous OBJ files first
        if neuron_id in previous_meshes:
            try:
                import trimesh as _tm
                mesh = _tm.load(previous_meshes[neuron_id])
                neuron = navis.MeshNeuron(mesh, id=neuron_id, name=name,
                                         units='nm')
                neurons[name] = neuron
                recycled += 1
                print(f"  Recycled {name} from {os.path.basename(os.path.dirname(os.path.dirname(previous_meshes[neuron_id])))}")
                continue
            except Exception as e:
                print(f"  Recycle failed for {name}, downloading fresh: {e}")

        # Download from FlyWire
        try:
            neuron = flywire.get_mesh_neuron(neuron_id, dataset="public", lod=lod)
            neurons[name] = neuron
            downloaded += 1
            print(f"  Downloaded {name}: {len(neuron.vertices)} vertices, {len(neuron.faces)} faces")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            neurons[name] = None

    print(f"\nNeuron loading complete: {recycled} recycled, {downloaded} downloaded, "
          f"{sum(1 for v in neurons.values() if v is None)} failed")
    return neurons


def save_individual_patch_data(source, target, contact_area, geo_data, threshold_um):
    """Save detailed patch data for individual pairs with contact"""
    # Create directory for individual patch data
    patch_dir = os.path.join(RESULTS_DIR, f"individual_patches_threshold_{threshold_um}um")
    os.makedirs(patch_dir, exist_ok=True)
    
    # Create filename
    filename = f"{source}_to_{target}_patch_data.csv"
    filepath = os.path.join(patch_dir, filename)
    
    # Prepare patch data
    patch_data = []
    
    if geo_data and 'face_data' in geo_data:
        for i, face_info in enumerate(geo_data['face_data']):
            patch_row = {
                'Source_Neuron': source,
                'Target_Neuron': target,
                'Threshold_um': threshold_um,
                'Total_Contact_Area_um2': contact_area,
                'Patch_ID': i,
                'Patch_Area_um2': face_info['area'] / 1e6,  # Convert to um²
                'Patch_Centroid_X': face_info['centroid'][0],
                'Patch_Centroid_Y': face_info['centroid'][1],
                'Patch_Centroid_Z': face_info['centroid'][2],
                'Patch_Centroid_X_Norm': face_info['centroid'][0] / 4.0,
                'Patch_Centroid_Y_Norm': face_info['centroid'][1] / 4.0,
                'Patch_Centroid_Z_Norm': face_info['centroid'][2] / 40.0,
                'Mesh_A_Face_ID': face_info.get('face_a_id', -1),
                'Mesh_B_Face_ID': face_info.get('face_b_id', -1),
                'Distance_nm': face_info.get('distance', 0) * 1000,  # Convert to nm
                'Num_Close_Vertices_A': len(face_info.get('vertices_a', [])),
                'Num_Close_Vertices_B': len(face_info.get('vertices_b', []))
            }
            patch_data.append(patch_row)
    
    # Save to CSV
    if patch_data:
        df = pd.DataFrame(patch_data)
        df.to_csv(filepath, index=False)
        print(f"    Saved {len(patch_data)} patches to {filename}")

def generate_target_pairs():
    """Generate target pairs from neurons.json pairing_rules (no BIPS)"""
    import json as _json2
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neurons.json')
    with open(cfg_path, 'r') as _f2:
        _cfg2 = _json2.load(_f2)

    # Build name→group mapping
    name_to_group = {}
    for name, info in _cfg2['neurons'].items():
        name_to_group[name] = info.get('group', '')

    all_names = list(_cfg2['neurons'].keys())
    target_pairs = []

    for rule in _cfg2.get('pairing_rules', {}).get('rules', []):
        group_a_set = set(rule['source'])
        group_b_set = set(rule['target'])
        bidirectional = rule.get('bidirectional', True)

        names_a = [n for n in all_names if name_to_group.get(n, '') in group_a_set]
        names_b = [n for n in all_names if name_to_group.get(n, '') in group_b_set]

        for a in names_a:
            for b in names_b:
                if a == b:
                    continue
                target_pairs.append((a, b))
                if bidirectional:
                    target_pairs.append((b, a))

    # Deduplicate
    target_pairs = list(dict.fromkeys(target_pairs))
    print(f"Generated {len(target_pairs)} target pairs for analysis")
    return target_pairs


def analyze_all_pairs(neurons, thresholds_microns):
    """Analyze target pairs at all thresholds with caching and incremental saving"""
    valid_neurons = {name: neuron for name, neuron in neurons.items() if neuron is not None}
    valid_names = list(valid_neurons.keys())
    
    # Generate target pairs
    target_pairs = generate_target_pairs()
    
    # Filter pairs to only include those with valid neurons
    valid_pairs = []
    for source, target in target_pairs:
        if source in valid_neurons and target in valid_neurons:
            valid_pairs.append((source, target))
    
    print(f"Analyzing {len(valid_pairs)} pairs at {len(thresholds_microns)} thresholds")
    print(f"Valid neurons: {valid_names}")
    print(f"Total valid pairs: {len(valid_pairs)}")
    
    # Check for cached results
    cache_key = get_cache_key(neuron_ids, thresholds_microns, LOD)
    cached_data = load_from_cache(cache_key)

    if cached_data is not None:
        print("Using cached results!")
        return cached_data['all_results'], cached_data['valid_names']



    
    all_results = {}

    # Create output directory for incremental saves
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for threshold_um in thresholds_microns:
        threshold_nm = threshold_um * 1000
        print(f"\n=== Threshold {threshold_um} um ({threshold_nm} nm) ===")

        results = {}
        pair_count = 0
        total_pairs = len(valid_pairs)
        successful_pairs = 0
        error_pairs = 0
        recycled_pairs = 0

        # Recycle pair results from previous runs (areas + patch files)
        prev_pair_data = _find_previous_pair_results(threshold_um)
        prev_patch_dir = _find_previous_patch_data(threshold_um)

        for source, target in valid_pairs:
            pair_count += 1
            pair_key = f"{source}→{target}"

            # Try recycling from previous results (now includes reconstructed geo_data)
            if pair_key in prev_pair_data:
                prev_result = prev_pair_data[pair_key]
                if isinstance(prev_result, tuple):
                    prev_area, prev_geo = prev_result
                else:
                    prev_area, prev_geo = prev_result, create_empty_geometric_data()
                results[pair_key] = (prev_area, prev_geo)
                recycled_pairs += 1
                successful_pairs += 1

                # Copy patch file if it existed
                if prev_patch_dir and prev_area > 0 and pair_key in prev_patch_dir:
                    prev_patch_file = prev_patch_dir[pair_key]
                    if os.path.exists(prev_patch_file):
                        patch_filename = f"{source}_to_{target}_patch_data.csv"
                        new_patch_dir = os.path.join(RESULTS_DIR, f"individual_patches_threshold_{threshold_um}um")
                        os.makedirs(new_patch_dir, exist_ok=True)
                        shutil.copy2(prev_patch_file, os.path.join(new_patch_dir, patch_filename))

                if pair_count % 20 == 0:
                    print(f"  Progress: {pair_count}/{total_pairs} ({recycled_pairs} recycled)")
                continue

            print(f"Pair {pair_count}/{total_pairs}: {pair_key}")

            try:
                area, geo_data = calculate_neuron_overlap_simple(
                    valid_neurons[source], 
                    valid_neurons[target], 
                    threshold=threshold_nm
                )
                results[pair_key] = (area, geo_data)
                successful_pairs += 1
                
                # Save individual patch data if there's contact
                if area > 0 and geo_data:
                    save_individual_patch_data(source, target, area, geo_data, threshold_um)
                
                # Save progress every 20 pairs
                if pair_count % 20 == 0:
                    print(f"  Progress: {pair_count}/{total_pairs} pairs completed")
                    print(f"  Success rate: {successful_pairs}/{pair_count} ({100*successful_pairs/pair_count:.1f}%)")
                    
            except Exception as e:
                print(f"  Error: {e}")
                results[pair_key] = (0.0, create_empty_geometric_data())
                error_pairs += 1
                
                # Continue processing even with errors
                continue
        
        all_results[threshold_um] = results
        
        # Save incremental results after each threshold
        print(f"\nThreshold {threshold_um}um completed:")
        print(f"  Recycled pairs: {recycled_pairs}/{total_pairs}")
        print(f"  Computed pairs: {successful_pairs - recycled_pairs}/{total_pairs}")
        print(f"  Error pairs: {error_pairs}/{total_pairs}")
        
        try:
            save_incremental_results(results, threshold_um, RESULTS_DIR)
        except Exception as e:
            print(f"Error saving incremental results: {e}")
    
    # Try to cache the complete results (but continue even if caching fails)
    try:
        cache_data = {'all_results': all_results, 'valid_names': valid_names}
        save_to_cache(cache_data, cache_key)
    except Exception as e:
        print(f"Warning: Could not cache results: {e}")
        print("Continuing without caching...")
    
    return all_results, valid_names

def create_matrices(all_results, valid_names, thresholds_microns):
    """Create matrices for visualization"""
    matrices = {}
    
    for threshold_um in thresholds_microns:
        n = len(valid_names)
        matrix = np.zeros((n, n))
        results = all_results[threshold_um]
        
        for i, nameA in enumerate(valid_names):
            for j, nameB in enumerate(valid_names):
                if nameA != nameB:
                    pair_key = f"{nameA}→{nameB}"
                    # Extract area from tuple (area, geo_data)
                    result = results.get(pair_key, (0.0, None))
                    matrix[i, j] = result[0] if isinstance(result, tuple) else result
        
        matrices[threshold_um] = matrix
    
    # Create mean matrix
    mean_matrix = np.zeros((len(valid_names), len(valid_names)))
    for i, nameA in enumerate(valid_names):
        for j, nameB in enumerate(valid_names):
            if nameA != nameB:
                areas = []
                for threshold_um in thresholds_microns:
                    ab_val = all_results[threshold_um].get(f"{nameA}→{nameB}", 0.0)
                    ba_val = all_results[threshold_um].get(f"{nameB}→{nameA}", 0.0)
                    ab_area = ab_val[0] if isinstance(ab_val, tuple) else ab_val
                    ba_area = ba_val[0] if isinstance(ba_val, tuple) else ba_val
                    if ab_area > 0 or ba_area > 0:
                        areas.extend([ab_area, ba_area])
                if areas:
                    mean_matrix[i, j] = np.mean(areas)
    
    matrices['mean_overall'] = mean_matrix
    return matrices

def visualize_matrices(matrices, valid_names, thresholds_microns):
    """Create visualizations"""
    
def visualize_matrices(matrices, valid_names, thresholds_microns):
    """Create visualizations with improved readability and interactivity"""
    
    # Set smaller font for readability
    plt.rcParams.update({'font.size': 8})
    
    # Static matplotlib visualization
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()
    
    for idx, threshold_um in enumerate(thresholds_microns):
        if idx < 3:
            ax = axes[idx]
            matrix = matrices[threshold_um]
            
            # Mask small values for clarity
            masked_matrix = np.where(matrix > np.percentile(matrix, 90), matrix, 0)
            
            im = ax.imshow(masked_matrix, cmap='viridis', aspect='auto')
            ax.set_title(f'Overlap at {threshold_um} um threshold (Top 10%)', fontsize=12)
            ax.set_xlabel('Target Neuron', fontsize=10)
            ax.set_ylabel('Source Neuron', fontsize=10)
            
            ax.set_xticks(range(len(valid_names)))
            ax.set_yticks(range(len(valid_names)))
            ax.set_xticklabels(valid_names, rotation=45, ha='right', fontsize=6)
            ax.set_yticklabels(valid_names, fontsize=6)
            
            plt.colorbar(im, ax=ax, label='Contact Area (um²)')
            
            # Add annotations only for significant values
            for i in range(len(valid_names)):
                for j in range(len(valid_names)):
                    if masked_matrix[i, j] > 0:
                        color = 'white' if masked_matrix[i, j] > masked_matrix.max()/2 else 'black'
                        ax.text(j, i, f'{masked_matrix[i, j]:.2f}', 
                               ha='center', va='center', color=color, fontsize=5)
    
    # Mean matrix
    ax = axes[3]
    mean_matrix = matrices['mean_overall']
    masked_mean = np.where(mean_matrix > np.percentile(mean_matrix, 90), mean_matrix, 0)
    
    im = ax.imshow(masked_mean, cmap='plasma', aspect='auto')
    ax.set_title('Mean Overlap - Top 10% (All Thresholds)', fontsize=12)
    ax.set_xlabel('Target Neuron', fontsize=10)
    ax.set_ylabel('Source Neuron', fontsize=10)
    
    ax.set_xticks(range(len(valid_names)))
    ax.set_yticks(range(len(valid_names)))
    ax.set_xticklabels(valid_names, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(valid_names, fontsize=6)
    
    plt.colorbar(im, ax=ax, label='Mean Contact Area (um²)')
    
    # Add annotations
    for i in range(len(valid_names)):
        for j in range(len(valid_names)):
            if masked_mean[i, j] > 0:
                color = 'white' if masked_mean[i, j] > masked_mean.max()/2 else 'black'
                ax.text(j, i, f'{masked_mean[i, j]:.2f}', 
                       ha='center', va='center', color=color, fontsize=5)
    
    plt.tight_layout()
    plt.savefig('comprehensive_overlap_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('comprehensive_overlap_analysis.svg', format='svg', bbox_inches='tight')
    plt.show()
    
    # Create advanced interactive visualization
    fig_plotly = create_advanced_interactive_viz(matrices, valid_names, thresholds_microns)
    
    return fig, fig_plotly

def create_advanced_interactive_viz(matrices, valid_names, thresholds_microns):
    """Create advanced interactive visualization with threshold and neuron type selection"""
    
    # Group neurons by type for filtering
    neuron_types = {}
    for name in valid_names:
        if any(x in name for x in ['MOT', 'MOS']):
            neuron_types.setdefault('Motor', []).append(name)
        elif any(x in name for x in ['VS']):
            neuron_types.setdefault('Visual', []).append(name)
        elif any(x in name for x in ['HS']):
            neuron_types.setdefault('Horizontal', []).append(name)
        elif any(x in name for x in ['H2']):
            neuron_types.setdefault('H2', []).append(name)
        elif any(x in name for x in ['MEME']):
            neuron_types.setdefault('MEME', []).append(name)
        else:
            neuron_types.setdefault('Other', []).append(name)
    
    # Create the main interactive figure
    fig = go.Figure()
    
    # Add traces for each threshold
    for threshold_um in thresholds_microns:
        fig.add_trace(
            go.Heatmap(
                z=matrices[threshold_um],
                x=valid_names,
                y=valid_names,
                colorscale='Viridis',
                name=f'{threshold_um} um',
                hovertemplate=f'Threshold: {threshold_um} um<br>Source: %{{y}}<br>Target: %{{x}}<br>Area: %{{z:.4f}} um²<extra></extra>',
                visible=True if threshold_um == thresholds_microns[0] else False
            )
        )
    
    # Add mean matrix trace
    fig.add_trace(
        go.Heatmap(
            z=matrices['mean_overall'],
            x=valid_names,
            y=valid_names,
            colorscale='Plasma',
            name='Mean Overall',
            hovertemplate='Mean across all thresholds<br>Source: %{y}<br>Target: %{x}<br>Area: %{z:.4f} um²<extra></extra>',
            visible=False
        )
    )
    
    # Create dropdown for threshold selection
    threshold_buttons = []
    for i, threshold_um in enumerate(thresholds_microns):
        visibility = [False] * (len(thresholds_microns) + 1)
        visibility[i] = True
        threshold_buttons.append(
            dict(
                label=f'{threshold_um} um',
                method='update',
                args=[{'visible': visibility},
                      {'title': f'Neuron Overlap Matrix - {threshold_um} um Threshold'}]
            )
        )
    
    # Add mean matrix button
    mean_visibility = [False] * (len(thresholds_microns) + 1)
    mean_visibility[-1] = True
    threshold_buttons.append(
        dict(
            label='Mean Overall',
            method='update',
            args=[{'visible': mean_visibility},
                  {'title': 'Mean Neuron Overlap Matrix (All Thresholds)'}]
        )
    )
    
    # Update layout with dropdown
    fig.update_layout(
        title=f'Interactive Neuron Overlap Matrix - {thresholds_microns[0]} um Threshold',
        xaxis_title='Target Neuron',
        yaxis_title='Source Neuron',
        width=900,
        height=800,
        updatemenus=[
            dict(
                buttons=threshold_buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.02,
                yanchor="top"
            ),
        ],
        annotations=[
            dict(text="Threshold:", showarrow=False,
                 x=0.01, y=1.08, yref="paper", align="left")
        ]
    )
    
    # Save interactive HTML
    fig.write_html('interactive_neuron_overlap_matrix.html')
    fig.show()
    
    return fig

def save_results(all_results, matrices, valid_names, thresholds_microns):
    """Save all results to files with multiple formats (wrapper function for compatibility)"""
    return save_final_combined_results(all_results, valid_names, thresholds_microns)

def save_final_combined_results(all_results, valid_names, thresholds_microns):
    """Save final combined results with error handling"""
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Combine all results
        all_results_combined = []
        for threshold_um in thresholds_microns:
            if threshold_um in all_results:
                for pair_key, result in all_results[threshold_um].items():
                    source, target = pair_key.split('→')
                    # Extract area and geometric data from tuple (area, geo_data)
                    if isinstance(result, tuple):
                        area, geo_data = result
                    else:
                        area = result
                        geo_data = None
                    
                    # Calculate contact patch centroid and other derived values
                    contact_centroid = np.array([0.0, 0.0, 0.0])
                    is_larger_patch = area > 10.0
                    
                    if geo_data and area > 0:
                        face_data = geo_data.get('face_data', [])
                        if face_data:
                            # Weighted average by area
                            total_patch_area = 0
                            for face_info in face_data:
                                patch_area = face_info['area'] / 1e6  # Convert to um²
                                contact_centroid += face_info['centroid'] * patch_area
                                total_patch_area += patch_area
                            if total_patch_area > 0:
                                contact_centroid /= total_patch_area
                    
                    # Compute top-N patches by area for this pair (if geo_data present)
                    top_patches = []
                    if geo_data and area > 0:
                        fd_list = geo_data.get('face_data', []) or []
                        if fd_list:
                            top_patches = sorted(fd_list, key=lambda fd: fd.get('area', 0), reverse=True)[:N_TOP_PATCHES]

                    result_row = {
                        'Source_Neuron': source,
                        'Target_Neuron': target,
                        'Contact_Area_um2': area,
                        'Threshold_um': threshold_um,
                        'Has_Contact': area > 0,
                        'Contact_Patch_Centroid_X': contact_centroid[0] if area > 0 else np.nan,
                        'Contact_Patch_Centroid_Y': contact_centroid[1] if area > 0 else np.nan,
                        'Contact_Patch_Centroid_Z': contact_centroid[2] if area > 0 else np.nan,
                        'Contact_Patch_Centroid_X_Norm': contact_centroid[0] / 4.0 if area > 0 else np.nan,
                        'Contact_Patch_Centroid_Y_Norm': contact_centroid[1] / 4.0 if area > 0 else np.nan,
                        'Contact_Patch_Centroid_Z_Norm': contact_centroid[2] / 40.0 if area > 0 else np.nan,
                        'Is_Larger_Patch': is_larger_patch if area > 0 else False
                    }
                    
                    # Add additional geometric data if available
                    if geo_data and area > 0:
                        result_row.update({
                            'Total_Area_Source_um2': geo_data.get('total_area_meshA', np.nan),
                            'Total_Area_Target_um2': geo_data.get('total_area_meshB', np.nan),
                            'Num_Contact_Vertices': len(geo_data.get('close_vertices_meshA', [])),
                            'Num_Contact_Faces': len(geo_data.get('face_data', []))
                        })
                    # Add Top1..TopN patch areas (um²) and centroids
                    for idx in range(N_TOP_PATCHES):
                        key_area = f'Top{idx+1}_Patch_Area_um2'
                        key_cx = f'Top{idx+1}_Patch_Centroid_X'
                        key_cy = f'Top{idx+1}_Patch_Centroid_Y'
                        key_cz = f'Top{idx+1}_Patch_Centroid_Z'
                        if idx < len(top_patches):
                            fd = top_patches[idx]
                            result_row[key_area] = (fd.get('area', np.nan) / 1e6)
                            c = fd.get('centroid', [np.nan, np.nan, np.nan])
                            result_row[key_cx] = c[0] if isinstance(c, (list, tuple, np.ndarray)) and len(c) >= 1 else np.nan
                            result_row[key_cy] = c[1] if isinstance(c, (list, tuple, np.ndarray)) and len(c) >= 2 else np.nan
                            result_row[key_cz] = c[2] if isinstance(c, (list, tuple, np.ndarray)) and len(c) >= 3 else np.nan
                        else:
                            result_row[key_area] = np.nan
                            result_row[key_cx] = np.nan
                            result_row[key_cy] = np.nan
                            result_row[key_cz] = np.nan

                    if not (geo_data and area > 0):
                        result_row.update({
                            'Total_Area_Source_um2': np.nan,
                            'Total_Area_Target_um2': np.nan,
                            'Num_Contact_Vertices': 0,
                            'Num_Contact_Faces': 0
                        })
                    
                    all_results_combined.append(result_row)
        
        # Save combined results
        if all_results_combined:
            combined_df = pd.DataFrame(all_results_combined)
            combined_file = os.path.join(RESULTS_DIR, "all_results_combined.csv")
            combined_df.to_csv(combined_file, index=False)
            print(f"Final combined results saved: {combined_file}")
        
        # Save overall summary statistics
        summary_stats = []
        for threshold_um in thresholds_microns:
            if threshold_um in all_results:
                results = all_results[threshold_um]
                # Extract areas from tuples (area, geo_data)
                areas = []
                for result in results.values():
                    area = result[0] if isinstance(result, tuple) else result
                    if area > 0:
                        areas.append(area)
                
                total_pairs = len(results)
                connected_pairs = len(areas)
                
                summary_stats.append({
                    'Threshold_um': threshold_um,
                    'Total_Pairs': total_pairs,
                    'Connected_Pairs': connected_pairs,
                    'Connection_Percentage': (connected_pairs/total_pairs)*100 if total_pairs > 0 else 0,
                    'Mean_Contact_Area': np.mean(areas) if areas else 0,
                    'Std_Contact_Area': np.std(areas) if areas else 0,
                    'Min_Contact_Area': np.min(areas) if areas else 0,
                    'Max_Contact_Area': np.max(areas) if areas else 0,
                    'Total_Contact_Area': np.sum(areas) if areas else 0
                })
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            summary_file = os.path.join(RESULTS_DIR, "final_summary_statistics.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"Final summary statistics saved: {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"Error saving final results: {e}")
        return False

def print_summary(all_results, valid_names, thresholds_microns):
    """Print analysis summary"""
    print("\n" + "="*80)
    print("COMPREHENSIVE NEURON OVERLAP ANALYSIS SUMMARY")
    print("="*80)
    print(f"Analyzed neurons: {', '.join(valid_names)}")
    print(f"Total neurons: {len(valid_names)}")
    print(f"Thresholds analyzed: {thresholds_microns} um")
    print()
    
    for threshold_um in thresholds_microns:
        results = all_results[threshold_um]
        # Extract areas from tuples (area, geo_data)
        areas = []
        for result in results.values():
            area = result[0] if isinstance(result, tuple) else result
            if area > 0:
                areas.append(area)
        
        total_pairs = len(results)
        connected_pairs = len(areas)
        
        print(f"--- Threshold: {threshold_um} um ---")
        print(f"  Connected pairs: {connected_pairs}/{total_pairs} ({connected_pairs/total_pairs:.1%})")
        if areas:
            print(f"  Mean contact area: {np.mean(areas):.4f} ± {np.std(areas):.4f} um²")
            print(f"  Range: {np.min(areas):.4f} - {np.max(areas):.4f} um²")
            print(f"  Total contact area: {np.sum(areas):.4f} um²")
        print()

# Main execution
def create_interactive_comparison_matrix(all_results, valid_names, thresholds_microns):
    """Create interactive Plotly heatmap for target pairs only"""
    
    # Get target pairs
    target_pairs = generate_target_pairs()
    
    # Filter valid pairs
    valid_pairs = []
    for source, target in target_pairs:
        if source in valid_names and target in valid_names:
            valid_pairs.append((source, target))
    
    # Create a subset of neurons involved in the analysis
    involved_neurons = set()
    for source, target in valid_pairs:
        involved_neurons.add(source)
        involved_neurons.add(target)
    involved_neurons = sorted(list(involved_neurons))
    
    for threshold_um in thresholds_microns:
        print(f"Creating interactive matrix for threshold {threshold_um} um")
        
        # Create matrix for involved neurons only
        n = len(involved_neurons)
        matrix = np.zeros((n, n))
        results = all_results[threshold_um]
        
        for i, source in enumerate(involved_neurons):
            for j, target in enumerate(involved_neurons):
                if source != target:
                    pair_key = f"{source}→{target}"
                    if pair_key in results:
                        result = results[pair_key]
                        area = result[0] if isinstance(result, tuple) else result
                        matrix[i, j] = area
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=involved_neurons,
            y=involved_neurons,
            colorscale='Viridis',
            colorbar=dict(title="Contact Area (um²)"),
            hovertemplate='Source: %{y}<br>Target: %{x}<br>Contact Area: %{z:.3f} um²<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Neuron Contact Matrix - Threshold {threshold_um} um<br>MOT/MOS ↔ HS/VS Analysis',
            xaxis_title='Target Neuron',
            yaxis_title='Source Neuron',
            width=800,
            height=800,
            font=dict(size=10)
        )
        
        # Save interactive plot
        output_file = os.path.join(RESULTS_DIR, f'interactive_contact_matrix_{threshold_um}um.html')
        fig.write_html(output_file)
        print(f"Interactive matrix saved to: {output_file}")
        
        # Also create a summary plot showing only significant contacts
        significant_matrix = np.where(matrix > 0.1, matrix, 0)  # Only show contacts > 0.1 um²
        
        fig_sig = go.Figure(data=go.Heatmap(
            z=significant_matrix,
            x=involved_neurons,
            y=involved_neurons,
            colorscale='Viridis',
            colorbar=dict(title="Contact Area (um²)"),
            hovertemplate='Source: %{y}<br>Target: %{x}<br>Contact Area: %{z:.3f} um²<extra></extra>'
        ))
        
        fig_sig.update_layout(
            title=f'Significant Neuron Contacts (>0.1 um²) - Threshold {threshold_um} um<br>MOT/MOS ↔ HS/VS Analysis',
            xaxis_title='Target Neuron',
            yaxis_title='Source Neuron',
            width=800,
            height=800,
            font=dict(size=10)
        )
        
        # Save significant contacts plot
        sig_output_file = os.path.join(RESULTS_DIR, f'significant_contacts_matrix_{threshold_um}um.html')
        fig_sig.write_html(sig_output_file)
        print(f"Significant contacts matrix saved to: {sig_output_file}")

def export_meshes_and_generate_viewer(neurons, results_dir):
    """Export neuron meshes as OBJ files and generate EM overlay viewer.
    Recycles meshes from previous results when possible."""
    import subprocess
    import sys

    # Create mesh directory
    mesh_dir = os.path.join(results_dir, "neuron_meshes")
    os.makedirs(mesh_dir, exist_ok=True)

    print(f"Exporting meshes to: {mesh_dir}")
    exported_count = 0
    recycled_count = 0

    # Find meshes from previous runs
    previous_meshes = _find_previous_meshes()

    # Export each neuron mesh
    for neuron_id, neuron_name in neuron_ids.items():
        obj_path = os.path.join(mesh_dir, f"{neuron_id}.obj")

        # Skip if already in current results dir
        if os.path.exists(obj_path):
            exported_count += 1
            continue

        # Try copying from previous results
        if neuron_id in previous_meshes:
            try:
                shutil.copy2(previous_meshes[neuron_id], obj_path)
                recycled_count += 1
                exported_count += 1
                print(f"  Recycled {neuron_name} mesh from previous run")
                continue
            except Exception:
                pass

        # Export fresh from loaded neuron
        if neuron_name not in neurons or neurons[neuron_name] is None:
            print(f"  Skipping {neuron_name} (not loaded)")
            continue

        neuron = neurons[neuron_name]
        try:
            if hasattr(neuron, 'trimesh'):
                mesh = neuron.trimesh
            else:
                print(f"  Skipping {neuron_name} (no trimesh)")
                continue

            mesh.export(obj_path)
            exported_count += 1
            print(f"  Exported {neuron_name} ({neuron_id})")
        except Exception as e:
            print(f"  Failed to export {neuron_name}: {e}")

    print(f"\nExported {exported_count}/{len(neuron_ids)} meshes ({recycled_count} recycled)")

    # Recycle EM snapshots from previous runs
    em_snap_dir = os.path.join(results_dir, 'em_snaps')
    if not os.path.isdir(em_snap_dir) or not os.listdir(em_snap_dir):
        prev_snaps = _find_previous_em_snaps()
        if prev_snaps:
            print(f"\nRecycling EM snapshots from: {prev_snaps}")
            os.makedirs(em_snap_dir, exist_ok=True)
            snap_count = 0
            for fname in os.listdir(prev_snaps):
                if fname.endswith('.png'):
                    src = os.path.join(prev_snaps, fname)
                    dst = os.path.join(em_snap_dir, fname)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        snap_count += 1
            print(f"  Copied {snap_count} EM snapshots")

    # Recycle geometric data from previous runs (always fill missing files)
    geo_dir = os.path.join(results_dir, 'geometric_data')
    os.makedirs(geo_dir, exist_ok=True)
    critical_geo_files = ['contact_vertices.csv', 'contact_faces.csv', 'contact_patches.csv']
    missing = [f for f in critical_geo_files if not os.path.exists(os.path.join(geo_dir, f))]
    if missing:
        for prev_dir in _find_previous_results():
            prev_geo = os.path.join(prev_dir, 'geometric_data')
            if os.path.isdir(prev_geo):
                copied = 0
                for fname in os.listdir(prev_geo):
                    src = os.path.join(prev_geo, fname)
                    dst = os.path.join(geo_dir, fname)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        copied += 1
                if copied:
                    print(f"\nRecycled {copied} geometric data files from: {prev_geo}")
                break

    # DISABLED: em_viewer.html generation (duplicate).
    # The single comprehensive viewer is generated by skeleton_em_viewer.py.
    print("Skipping em_viewer.html (consolidated into skeleton_em_viewer.py)")

































if __name__ == "__main__":
    print("Starting comprehensive pairwise neuron overlap analysis...")
    print(f"Results directory: {RESULTS_DIR}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Report recycling sources
    prev_dirs = _find_previous_results()
    if prev_dirs:
        print(f"\nFound {len(prev_dirs)} previous result(s) to recycle from:")
        for d in prev_dirs[:3]:
            print(f"  - {os.path.basename(d)}")
    else:
        print("\nNo previous results found; will compute everything from scratch.")

    start_time = time.time()
    
    try:
        # Load neurons
        print("\n" + "="*60)
        print("STEP 1: Loading neurons")
        print("="*60)
        neurons = load_all_neurons(neuron_ids, LOD)
        
        # Immediately save a meshes-only HTML so a figure is always produced
        try:
            valid_names_mesh = [name for name, n in neurons.items() if n is not None]
            build_meshes_only_html(neurons, valid_names_mesh, RESULTS_DIR, title_suffix="All neuron meshes (pre-analysis)")
        except Exception as e:
            print(f"Meshes-only HTML failed (non-critical): {e}")
        
        # Analyze pairs
        print("\n" + "="*60)
        print("STEP 2: Analyzing all pairs")
        print("="*60)
        all_results, valid_names = analyze_all_pairs(neurons, THRESHOLDS_MICRONS)
        
        # Save final combined results
        print("\n" + "="*60)
        print("STEP 3: Saving final combined results")
        print("="*60)
        save_final_combined_results(all_results, valid_names, THRESHOLDS_MICRONS)

        # Fetch synapses once (optional overlay) — try recycling first
        syn_df_global = None
        prev_syn = _find_previous_synapses()
        if prev_syn:
            try:
                print(f"Recycling synapses from: {prev_syn}")
                syn_df_global = pd.read_csv(prev_syn)
                print(f"  Loaded {len(syn_df_global)} synapses from previous run")
                # Save copy to new results directory
                syn_copy = os.path.join(RESULTS_DIR, "synapses.csv")
                syn_df_global.to_csv(syn_copy, index=False)
            except Exception as e:
                print(f"  Synapse recycle failed: {e}")
                syn_df_global = None
        if syn_df_global is None:
            try:
                print("Fetching synapses for overlay (once)...")
                syn_df_global = fetch_synapses_for_neurons(neuron_ids)
            except Exception as e:
                print(f"Synapse fetch skipped: {e}")

        # Always build the high-quality meshes + overlaps HTML visualization
        print("\n" + "="*60)
        print("STEP 4: Building 3D meshes + overlap HTML (publication quality)")
        print("="*60)
        # Build heavy (publication quality)
        try:
            thr = THRESHOLDS_MICRONS[0]
            # Participants filters: plot only cells participating in contacts
            participant_sources = {"MOT", "MOS"}
            participant_targets = {"HS", "VS"}
            all_include = compute_contact_participants(all_results, thr, participant_sources, participant_targets)

            mot_include = compute_contact_participants(all_results, thr, {"MOT"}, {"HS", "VS"})
            mos_include = compute_contact_participants(all_results, thr, {"MOS"}, {"HS", "VS"})

            # All MOT+MOS (only participants)
            build_mesh_and_overlap_html(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                        filename_suffix="ALL", source_group_filter=participant_sources,
                                        target_group_filter=participant_targets, include_names=all_include,
                                        synapses_df=syn_df_global)
            # MOT-only (only HS targets)
            build_mesh_and_overlap_html(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                        filename_suffix="MOT_only", source_group_filter={"MOT"},
                                        target_group_filter={"HS","VS"}, include_names=mot_include,
                                        synapses_df=syn_df_global)
            # MOS-only (HS and VS targets)
            build_mesh_and_overlap_html(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                        filename_suffix="MOS_only", source_group_filter={"MOS"},
                                        target_group_filter={"HS","VS"}, include_names=mos_include,
                                        synapses_df=syn_df_global)
        except Exception as e:
            print(f"Meshes + overlaps HTML failed (non-critical): {e}")
            _log_viz_debug("HEAVY HTML ERROR:\n" + traceback.format_exc())
            _write_placeholder_html(RESULTS_DIR, name="meshes_and_overlaps_ERROR.html")
        
        # Always attempt to build LITE HTML (browser-friendly), even if heavy failed
        try:
            thr = THRESHOLDS_MICRONS[0]
            participant_sources = {"MOT", "MOS"}
            participant_targets = {"HS", "VS"}
            all_include = compute_contact_participants(all_results, thr, participant_sources, participant_targets)
            mot_include = compute_contact_participants(all_results, thr, {"MOT"}, {"HS","VS"})
            mos_include = compute_contact_participants(all_results, thr, {"MOS"}, {"HS","VS"})

            # All sources
            build_mesh_and_overlap_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                             filename_suffix="ALL", target_group_filter={"HS","VS"},
                                             include_names=all_include, synapses_df=syn_df_global)
            # MOT-only
            build_mesh_and_overlap_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                             filename_suffix="MOT_only", source_group_filter={"MOT"},
                                             target_group_filter={"HS"}, include_names=mot_include, synapses_df=syn_df_global)
            # MOS-only
            build_mesh_and_overlap_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                             filename_suffix="MOS_only", source_group_filter={"MOS"},
                                             target_group_filter={"HS","VS"}, include_names=mos_include, synapses_df=syn_df_global)
            # Per-hemisphere views with point-cloud meshes (as requested earlier)
            build_mesh_and_overlap_html_lite(
                neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                hemi_filter='L', mot_mos_faces=50000, hs_vs_faces=12000, filename_suffix="ALL",
                target_group_filter={"HS","VS"}, include_names=all_include, synapses_df=syn_df_global
            )
            build_mesh_and_overlap_html_lite(
                neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                hemi_filter='R', mot_mos_faces=50000, hs_vs_faces=12000, filename_suffix="ALL",
                target_group_filter={"HS","VS"}, include_names=all_include, synapses_df=syn_df_global
            )
            # Hemi + MOT-only
            build_mesh_and_overlap_html_lite(
                neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                hemi_filter='L', mot_mos_faces=50000, hs_vs_faces=12000, filename_suffix="MOT_only",
                source_group_filter={"MOT"}, target_group_filter={"HS"}, include_names=mot_include, synapses_df=syn_df_global
            )
            build_mesh_and_overlap_html_lite(
                neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                hemi_filter='R', mot_mos_faces=50000, hs_vs_faces=12000, filename_suffix="MOT_only",
                source_group_filter={"MOT"}, target_group_filter={"HS"}, include_names=mot_include, synapses_df=syn_df_global
            )
            # Hemi + MOS-only
            build_mesh_and_overlap_html_lite(
                neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                hemi_filter='L', mot_mos_faces=50000, hs_vs_faces=12000, filename_suffix="MOS_only",
                source_group_filter={"MOS"}, target_group_filter={"HS","VS"}, include_names=mos_include, synapses_df=syn_df_global
            )
            build_mesh_and_overlap_html_lite(
                neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                hemi_filter='R', mot_mos_faces=50000, hs_vs_faces=12000, filename_suffix="MOS_only",
                source_group_filter={"MOS"}, target_group_filter={"HS","VS"}, include_names=mos_include, synapses_df=syn_df_global
            )
        except Exception as e:
            print(f"Meshes + overlaps LITE HTML failed (non-critical): {e}")
            _log_viz_debug("LITE HTML ERROR:\n" + traceback.format_exc())
            _write_placeholder_html(RESULTS_DIR, name="meshes_and_overlaps_LITE_ERROR.html")

        # Also attempt a point-cloud visualization as a very robust fallback
        try:
            thr = THRESHOLDS_MICRONS[0]
            participant_sources = {"MOT", "MOS"}
            participant_targets = {"HS", "VS"}
            all_include = compute_contact_participants(all_results, thr, participant_sources, participant_targets)
            mot_include = compute_contact_participants(all_results, thr, {"MOT"}, {"HS","VS"})
            mos_include = compute_contact_participants(all_results, thr, {"MOS"}, {"HS","VS"})
            build_pointcloud_and_overlap_html(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              points_per_neuron=50000, patch_mode='points', max_patch_faces=100000,
                                              filename_suffix="ALL", target_group_filter={"HS","VS"}, include_names=all_include,
                                              synapses_df=syn_df_global)
            build_pointcloud_and_overlap_html(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              points_per_neuron=50000, patch_mode='points', max_patch_faces=100000,
                                              filename_suffix="MOT_only", source_group_filter={"MOT"}, target_group_filter={"HS","VS"}, include_names=mot_include,
                                              synapses_df=syn_df_global)
            build_pointcloud_and_overlap_html(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              points_per_neuron=50000, patch_mode='points', max_patch_faces=100000,
                                              filename_suffix="MOS_only", source_group_filter={"MOS"}, target_group_filter={"HS","VS"}, include_names=mos_include,
                                              synapses_df=syn_df_global)
        except Exception as e:
            print(f"Point-cloud HTML failed (non-critical): {e}")
            _log_viz_debug("POINTCLOUD HTML ERROR:\n" + traceback.format_exc())
            _write_placeholder_html(RESULTS_DIR, name="meshes_and_overlaps_POINTCLOUD_ERROR.html")

        # Additionally produce wireframe alternatives (very light)
        try:
            thr = THRESHOLDS_MICRONS[0]
            participant_sources = {"MOT", "MOS"}
            participant_targets = {"HS", "VS"}
            all_include = compute_contact_participants(all_results, thr, participant_sources, participant_targets)
            mot_include = compute_contact_participants(all_results, thr, {"MOT"}, {"HS","VS"})
            mos_include = compute_contact_participants(all_results, thr, {"MOS"}, {"HS","VS"})
            # ALL
            build_overlap_wireframe_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              filename_suffix="ALL", include_names=all_include,
                                              source_group_filter=participant_sources, target_group_filter=participant_targets, synapses_df=syn_df_global)
            # MOT-only
            build_overlap_wireframe_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              filename_suffix="MOT_only", include_names=mot_include,
                                              source_group_filter={"MOT"}, target_group_filter={"HS","VS"}, synapses_df=syn_df_global)
            # MOS-only
            build_overlap_wireframe_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              filename_suffix="MOS_only", include_names=mos_include,
                                              source_group_filter={"MOS"}, target_group_filter={"HS","VS"}, synapses_df=syn_df_global)
            # Hemi L/R
            build_overlap_wireframe_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              hemi_filter='L', filename_suffix="ALL", include_names=all_include,
                                              source_group_filter=participant_sources, target_group_filter=participant_targets, synapses_df=syn_df_global)
            build_overlap_wireframe_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              hemi_filter='R', filename_suffix="ALL", include_names=all_include,
                                              source_group_filter=participant_sources, target_group_filter=participant_targets, synapses_df=syn_df_global)
            # Hemi L/R MOT-only
            build_overlap_wireframe_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              hemi_filter='L', filename_suffix="MOT_only", include_names=mot_include,
                                              source_group_filter={"MOT"}, target_group_filter={"HS","VS"}, synapses_df=syn_df_global)
            build_overlap_wireframe_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              hemi_filter='R', filename_suffix="MOT_only", include_names=mot_include,
                                              source_group_filter={"MOT"}, target_group_filter={"HS","VS"}, synapses_df=syn_df_global)
            # Hemi L/R MOS-only
            build_overlap_wireframe_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              hemi_filter='L', filename_suffix="MOS_only", include_names=mos_include,
                                              source_group_filter={"MOS"}, target_group_filter={"HS","VS"}, synapses_df=syn_df_global)
            build_overlap_wireframe_html_lite(neurons, all_results, valid_names, THRESHOLDS_MICRONS, RESULTS_DIR,
                                              hemi_filter='R', filename_suffix="MOS_only", include_names=mos_include,
                                              source_group_filter={"MOS"}, target_group_filter={"HS","VS"}, synapses_df=syn_df_global)
        except Exception as e:
            print(f"Wireframe HTML failed (non-critical): {e}")

        # Build synapses-only quick verification pages
        try:
            if syn_df_global is not None:
                build_synapses_only_html(valid_names, RESULTS_DIR, syn_df_global, hemi_filter=None, title_suffix="ALL")
                build_synapses_only_html(valid_names, RESULTS_DIR, syn_df_global, hemi_filter='L', title_suffix="LEFT")
                build_synapses_only_html(valid_names, RESULTS_DIR, syn_df_global, hemi_filter='R', title_suffix="RIGHT")
        except Exception as e:
            print(f"Synapses-only HTML failed (non-critical): {e}")

        # Try to create matrices (may fail due to memory)
        print("\n" + "="*60)
        print("STEP 5: Creating matrices (optional)")
        print("="*60)
        try:
            matrices = create_matrices(all_results, valid_names, THRESHOLDS_MICRONS)
            print("Matrices created successfully")
            # Additional matrices: synapse counts (pre x post) and overlap areas (source x target) at first threshold
            try:
                thr = THRESHOLDS_MICRONS[0]
                n = len(valid_names)
                # Overlap area matrix
                overlap_mat = np.zeros((n, n))
                res_thr = all_results.get(thr, {})
                for i, pre in enumerate(valid_names):
                    for j, post in enumerate(valid_names):
                        if pre == post:
                            continue
                        val = res_thr.get(f"{pre}→{post}", 0.0)
                        area = val[0] if isinstance(val, tuple) else val
                        overlap_mat[i, j] = area
                overlap_df = pd.DataFrame(overlap_mat, index=valid_names, columns=valid_names)
                overlap_path = os.path.join(RESULTS_DIR, f"matrix_overlap_area_{thr}um.csv")
                overlap_df.to_csv(overlap_path)
                print(f"Saved overlap area matrix: {overlap_path}")

                # Synapse count matrix (pre x post) using syn_df_global
                if 'syn_df_global' in globals() and syn_df_global is not None:
                    syn = syn_df_global.dropna(subset=['pre_type','post_type'])
                    syn = syn[(syn['pre_type'].isin(valid_names)) & (syn['post_type'].isin(valid_names))]
                    syn_mat = pd.pivot_table(syn, index='pre_type', columns='post_type', values='id', aggfunc='count', fill_value=0)
                    # Ensure full index/columns ordering
                    syn_mat = syn_mat.reindex(index=valid_names, columns=valid_names, fill_value=0)
                    syn_path = os.path.join(RESULTS_DIR, "matrix_synapse_counts.csv")
                    syn_mat.to_csv(syn_path)
                    print(f"Saved synapse count matrix: {syn_path}")
            except Exception as e:
                print(f"Additional matrices failed (non-critical): {e}")
            
            # Try visualization (optional)
            print("\n" + "="*60)
            print("STEP 6: Creating visualizations (optional)")
            print("="*60)
            try:
                fig_static, fig_interactive = visualize_matrices(matrices, valid_names, THRESHOLDS_MICRONS)
                print("Visualizations created successfully")
            except Exception as e:
                print(f"Visualization failed (non-critical): {e}")
            
            # Create interactive comparison matrix for target pairs
            try:
                create_interactive_comparison_matrix(all_results, valid_names, THRESHOLDS_MICRONS)
                print("Interactive comparison matrix created successfully")
            except Exception as e:
                print(f"Interactive comparison matrix failed (non-critical): {e}")
        
        except Exception as e:
            print(f"Matrix creation failed (non-critical): {e}")
            matrices = None
        
        # Try to save detailed geometric data (optional)
        print("\n" + "="*60)
        print("STEP 6: Saving detailed geometric data (optional)")
        print("="*60)
        try:
            save_detailed_geometric_data(all_results, RESULTS_DIR)
        except Exception as e:
            print(f"Detailed geometric data saving failed (non-critical): {e}")
        
        # Save individual pair geometric data
            print(f"Individual pair geometric data saving failed (non-critical): {e}")
        
        # Print summary
        print_summary(all_results, valid_names, THRESHOLDS_MICRONS)
        
        # Export meshes and generate EM viewer
        print("\n" + "="*60)
        print("STEP 8: Exporting neuron meshes and generating EM viewer")
        print("="*60)
        try:
            export_meshes_and_generate_viewer(neurons, RESULTS_DIR)
        except Exception as e:
            print(f"Mesh export and EM viewer generation failed (non-critical): {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\nTotal analysis time: {time.time() - start_time:.2f} seconds")
        print("Analysis complete!")
        print(f"\nResults saved in: {RESULTS_DIR}/")
        print("Check the individual CSV files for each threshold and the combined results.")
        
    except Exception as e:
        print(f"\nCritical error during analysis: {e}")
        print("Checking for any saved incremental results...")
        
        # Try to at least show what was saved
        try:
            import glob
            saved_files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
            if saved_files:
                print(f"Found {len(saved_files)} saved result files:")
                for f in saved_files:
                    print(f"  - {f}")
            else:
                print("No result files found.")
        except:
            pass
        
        print(f"Analysis failed after {time.time() - start_time:.2f} seconds")
