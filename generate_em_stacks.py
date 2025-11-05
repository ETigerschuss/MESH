"""
Generate ALL EM snapshots with segmentation - Complete Clean Download
- All 240 contact patches (center + 41 Z-stack images each)
- All 67 synapses (center + 41 Z-stack images each)
- Consistent resolution (512x512 pixels)
- Consistent Z-range (-20 to +20, 41 images)
- All with colored segmentation overlays
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from tqdm import tqdm

try:
    from cloudvolume import CloudVolume
    HAVE_CLOUDVOLUME = True
except ImportError:
    print("[ERROR] CloudVolume not available. Install with: pip install cloud-volume")
    HAVE_CLOUDVOLUME = False
    exit(1)

# Neuron IDs and colors
NEURON_IDS = {
    720575940618519710: 'MOT_L', 720575940630139386: 'MOT_R',
    720575940622361270: 'MOS_L', 720575940622168052: 'MOS_R',
    720575940626477498: 'VS1_L', 720575940619878961: 'VS1_R',
    720575940640722851: 'VS2_L', 720575940613126835: 'VS2_R',
    720575940622831740: 'VS3_L', 720575940641812699: 'VS3_R',
    720575940624273919: 'VS4_L', 720575940659799937: 'VS4_R',
    720575940626457406: 'VS5_L', 720575940639151694: 'VS5_R',
    720575940647311651: 'VS6_L', 720575940626928521: 'VS6_R',
    720575940624931564: 'VS7_L', 720575940618681709: 'VS7_R',
    720575940633923298: 'VS8_L', 720575940636972400: 'VS8_R',
    720575940628031249: 'HSN_L', 720575940615933919: 'HSN_R',
    720575940629153020: 'HSE_L', 720575940629148007: 'HSE_R',
    720575940622312965: 'HSS_L', 720575940628743496: 'HSS_R',
}

# Color scheme based on cell types with nuances
# VS (Visual System) = #1B9E77 (teal/green)
# HS (Horizontal System) = #D95F02 (orange)
# MNs (Motor Neurons) = #7570B3 (purple)

NEURON_COLORS = {
    # Motor Neurons (MNs) - Purple #7570B3 with nuances
    'MOT_L': (117, 112, 179),      # Base purple
    'MOT_R': (147, 142, 209),      # Lighter purple
    'MOS_L': (87, 82, 149),        # Darker purple
    'MOS_R': (107, 102, 169),      # Medium purple
    
    # Visual System (VS) - Teal #1B9E77 with nuances
    'VS1_L': (27, 158, 119),       # Base teal
    'VS1_R': (57, 188, 149),       # Lighter teal
    'VS2_L': (17, 138, 99),        # Darker teal
    'VS2_R': (37, 168, 129),       # Medium-light teal
    'VS3_L': (22, 148, 109),       # Slightly darker
    'VS3_R': (47, 178, 139),       # Slightly lighter
    'VS4_L': (12, 128, 89),        # More saturated dark
    'VS4_R': (52, 198, 159),       # More saturated light
    'VS5_L': (27, 143, 104),       # Shifted darker
    'VS5_R': (42, 173, 134),       # Shifted lighter
    'VS6_L': (17, 153, 114),       # Variation 1
    'VS6_R': (37, 183, 144),       # Variation 1 light
    'VS7_L': (22, 133, 94),        # Variation 2
    'VS7_R': (47, 163, 124),       # Variation 2 light
    'VS8_L': (27, 148, 109),       # Variation 3
    'VS8_R': (52, 178, 139),       # Variation 3 light
    
    # Horizontal System (HS) - Orange #D95F02 with nuances
    'HSN_L': (217, 95, 2),         # Base orange
    'HSN_R': (237, 125, 42),       # Lighter orange
    'HSE_L': (197, 75, 0),         # Darker orange
    'HSE_R': (227, 105, 22),       # Medium orange
    'HSS_L': (207, 85, 0),         # Variation 1
    'HSS_R': (237, 115, 32),       # Variation 1 light
}


def create_segmented_snapshot(em_vol, seg_vol, center_nm, source_id, target_id, 
                              source_name, target_name, z_offset_slices=0, size_pixels=512):
    """
    Create EM snapshot with colored neuron segmentation overlay
    PROVEN WORKING VERSION - using exact code from create_segmented_em_snapshots.py
    """
    # Convert nm to voxel coordinates using each volume's resolution
    em_center_vox = np.array(center_nm) / np.array(em_vol.resolution)
    seg_center_vox = np.array(center_nm) / np.array(seg_vol.resolution)
    
    # Account for resolution difference: EM is 8nm/voxel, Seg is 16nm/voxel (2x larger)
    resolution_ratio = seg_vol.resolution[0] / em_vol.resolution[0]  # = 2.0
    
    em_half_size = size_pixels // 2
    seg_half_size = int(em_half_size / resolution_ratio)  # = 256 / 2 = 128
    
    # EM patch bounds
    em_x_start = int(em_center_vox[0] - em_half_size)
    em_x_end = int(em_center_vox[0] + em_half_size)
    em_y_start = int(em_center_vox[1] - em_half_size)
    em_y_end = int(em_center_vox[1] + em_half_size)
    em_z = int(em_center_vox[2])
    
    # Segmentation patch bounds
    seg_x_start = int(seg_center_vox[0] - seg_half_size)
    seg_x_end = int(seg_center_vox[0] + seg_half_size)
    seg_y_start = int(seg_center_vox[1] - seg_half_size)
    seg_y_end = int(seg_center_vox[1] + seg_half_size)
    seg_z = int(seg_center_vox[2])
    
    # Apply Z-offset if provided
    seg_z += z_offset_slices
    em_z += z_offset_slices
    
    try:
        # Fetch EM and segmentation at the SAME Z-offset
        em_data = em_vol[em_x_start:em_x_end, em_y_start:em_y_end, em_z:em_z+1]
        em_slice = em_data[:, :, 0, 0]
        
        seg_data = seg_vol[seg_x_start:seg_x_end, seg_y_start:seg_y_end, seg_z:seg_z+1]
        seg_slice = seg_data[:, :, 0, 0]
        
        # Create RGB image from grayscale EM
        em_rgb = np.stack([em_slice, em_slice, em_slice], axis=-1)
        
        # Color the source and target neurons
        overlay_small = np.zeros((seg_slice.shape[0], seg_slice.shape[1], 3), dtype=np.uint8)
        alpha_mask_small = np.zeros(seg_slice.shape, dtype=np.float32)
        
        neurons_to_color = {
            source_id: source_name,
            target_id: target_name
        }
        
        for neuron_id, neuron_name in neurons_to_color.items():
            color = NEURON_COLORS.get(neuron_name, (255, 255, 255))
            mask_small = (seg_slice == neuron_id)
            overlay_small[mask_small] = color
            alpha_mask_small[mask_small] = 0.5
        
        # Resize the colored overlay and alpha mask to match EM size
        if seg_slice.shape != em_slice.shape:
            zoom_factors = (em_slice.shape[0] / seg_slice.shape[0], 
                           em_slice.shape[1] / seg_slice.shape[1])
            overlay = np.zeros_like(em_rgb, dtype=np.uint8)
            for c in range(3):
                overlay[:,:,c] = ndimage.zoom(overlay_small[:,:,c], zoom_factors, order=0)
            alpha_mask = ndimage.zoom(alpha_mask_small, zoom_factors, order=0)
        else:
            overlay = overlay_small
            alpha_mask = alpha_mask_small
        
        # Blend EM and overlay
        alpha_3d = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=-1)
        blended = (em_rgb * (1 - alpha_3d) + overlay * alpha_3d).astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(blended)
        
        # Add labels
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        label = f"{source_name} → {target_name}"
        draw.text((10, 10), label, fill=(255, 255, 255), font=font)
        
        if z_offset_slices != 0:
            z_label = f"Z: {z_offset_slices:+d} ({z_offset_slices * 40:+d}nm)"
            draw.text((10, 30), z_label, fill=(255, 255, 0), font=font)
        
        return img
        
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


def main():
    results_dir = 'comprehensive_overlap_results'
    em_snap_dir = os.path.join(results_dir, 'em_snaps')
    os.makedirs(em_snap_dir, exist_ok=True)
    
    print("="*70)
    print("Generate ALL EM Snapshots - Complete Clean Download")
    print("="*70)
    
    # Initialize CloudVolume connections
    print("\n[1/5] Connecting to data sources...")
    em_vol = CloudVolume('https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14', 
                         mip=1, use_https=True, progress=False)
    seg_vol = CloudVolume('precomputed://gs://flywire_v141_m783', 
                          mip=0, use_https=True, progress=False)
    print("  [OK] Connected to EM and segmentation volumes")
    
    # ========== PROCESS CONTACTS ==========
    print("\n[2/5] Loading contact data...")
    contacts_file = os.path.join(results_dir, 'all_results_combined.csv')
    
    if not os.path.exists(contacts_file):
        print(f"[ERROR] File not found: {contacts_file}")
        return
    
    df = pd.read_csv(contacts_file)
    df = df[df['Has_Contact'] == True]
    
    print(f"  Found {len(df)} contact pairs")
    
    # Expand all patches (Top1-Top6)
    print("\n[3/5] Processing ALL contact patches...")
    
    all_patches = []
    patch_idx = 0
    
    for _, row in df.iterrows():
        source_name = row['Source_Neuron']
        target_name = row['Target_Neuron']
        
        # Get neuron IDs
        source_id = None
        target_id = None
        for fid, fname in NEURON_IDS.items():
            if fname == source_name:
                source_id = fid
            if fname == target_name:
                target_id = fid
        
        if source_id is None or target_id is None:
            continue
        
        # Process each Top patch (Top1-Top6)
        for patch_num in range(1, 7):
            x_col = f'Top{patch_num}_Patch_Centroid_X'
            y_col = f'Top{patch_num}_Patch_Centroid_Y'
            z_col = f'Top{patch_num}_Patch_Centroid_Z'
            
            if all(col in df.columns for col in [x_col, y_col, z_col]):
                if not pd.isna(row[x_col]):
                    patch_center = (row[x_col], row[y_col], row[z_col])
                    all_patches.append({
                        'idx': patch_idx,
                        'source_name': source_name,
                        'target_name': target_name,
                        'source_id': source_id,
                        'target_id': target_id,
                        'center': patch_center,
                        'patch_num': patch_num
                    })
                    patch_idx += 1
    
    print(f"  Total contact patches: {len(all_patches)}")
    print(f"  Will generate {len(all_patches)} center + {len(all_patches) * 41} Z-stack = {len(all_patches) * 42} images")
    
    # Generate contact snapshots with Z-stacks
    contact_success = 0
    total_contact_images = 0
    skipped_contact_images = 0
    
    for patch in tqdm(all_patches, desc="Contacts (center + Z-stacks)"):
        idx = patch['idx']
        
        # Generate Z-stack: -20 to +20 (41 images)
        for z_offset in range(-20, 21):
            # Determine output path
            if z_offset == 0:
                # Center slice - save as _segmented
                output_path = os.path.join(em_snap_dir, f"contact_{idx}_segmented.png")
            else:
                # Z-stack slice
                z_str = f"z{z_offset:+04d}"
                output_path = os.path.join(em_snap_dir, f"contact_{idx}_{z_str}.png")
            
            # Skip if already exists
            if os.path.exists(output_path):
                skipped_contact_images += 1
                continue
            
            img = create_segmented_snapshot(
                em_vol, seg_vol,
                patch['center'],
                patch['source_id'], patch['target_id'],
                patch['source_name'], patch['target_name'],
                z_offset_slices=z_offset,
                size_pixels=512
            )
            
            if img is not None:
                img.save(output_path, 'PNG')
                total_contact_images += 1
        
        contact_success += 1
    
    print(f"\n  [OK] Contacts: {contact_success}/{len(all_patches)} patches, {total_contact_images} new + {skipped_contact_images} skipped")
    
    # ========== PROCESS SYNAPSES ==========
    print("\n[4/5] Loading synapse data...")
    synapses_file = os.path.join(results_dir, 'synapses.csv')
    
    if not os.path.exists(synapses_file):
        print(f"[ERROR] File not found: {synapses_file}")
        return
    
    syn_df = pd.read_csv(synapses_file)
    syn_df = syn_df.dropna(subset=['x', 'y', 'z'])
    
    # Filter for MOT/MOS synapses (same approach as working script)
    mot_mos_neurons = ['MOT_L', 'MOT_R', 'MOS_L', 'MOS_R']
    
    # Handle both pre_type/post_type and pre/post column names
    if 'pre_type' in syn_df.columns and 'post_type' in syn_df.columns:
        syn_df['source'] = syn_df['pre_type']
        syn_df['target'] = syn_df['post_type']
    elif 'pre' in syn_df.columns and 'post' in syn_df.columns:
        syn_df['source'] = syn_df['pre']
        syn_df['target'] = syn_df['post']
    else:
        print(f"[ERROR] Could not find source/target columns in synapses.csv")
        return
    
    mot_mos = syn_df[
        syn_df['source'].isin(mot_mos_neurons) | 
        syn_df['target'].isin(mot_mos_neurons)
    ]
    
    print(f"  Found {len(mot_mos)} MOT/MOS synapses")
    print(f"  Will generate {len(mot_mos)} center + {len(mot_mos) * 41} Z-stack = {len(mot_mos) * 42} images")
    
    # Generate synapse snapshots with Z-stacks
    synapse_success = 0
    total_synapse_images = 0
    skipped_synapse_images = 0
    
    for idx, row in tqdm(mot_mos.iterrows(), total=len(mot_mos), desc="Synapses (center + Z-stacks)"):
        source_name = row['source']
        target_name = row['target']
        
        # Get neuron IDs
        source_id = None
        target_id = None
        for fid, fname in NEURON_IDS.items():
            if fname == source_name:
                source_id = fid
            if fname == target_name:
                target_id = fid
        
        if source_id is None or target_id is None:
            continue
        
        center_nm = (row['x'], row['y'], row['z'])
        
        # Generate Z-stack: -20 to +20 (41 images)
        for z_offset in range(-20, 21):
            # Determine output path
            if z_offset == 0:
                # Center slice - save as _segmented
                output_path = os.path.join(em_snap_dir, f"synapse_{idx}_segmented.png")
            else:
                # Z-stack slice
                z_str = f"z{z_offset:+04d}"
                output_path = os.path.join(em_snap_dir, f"synapse_{idx}_{z_str}.png")
            
            # Skip if already exists
            if os.path.exists(output_path):
                skipped_synapse_images += 1
                continue
            
            img = create_segmented_snapshot(
                em_vol, seg_vol,
                center_nm,
                source_id, target_id,
                source_name, target_name,
                z_offset_slices=z_offset,
                size_pixels=512
            )
            
            if img is not None:
                img.save(output_path, 'PNG')
                total_synapse_images += 1
        
        synapse_success += 1
    
    print(f"\n  [OK] Synapses: {synapse_success}/{len(mot_mos)} synapses, {total_synapse_images} new + {skipped_synapse_images} skipped")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("[5/5] COMPLETE!")
    print("="*70)
    print(f"  Contact patches: {contact_success} ({total_contact_images} images)")
    print(f"  Synapses: {synapse_success} ({total_synapse_images} images)")
    print(f"  TOTAL: {total_contact_images + total_synapse_images} images")
    print(f"  Resolution: 512x512 pixels (4096nm × 4096nm)")
    print(f"  Z-range: -20 to +20 (±800nm depth, 41 slices)")
    print(f"  All images saved in: {em_snap_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
