"""
Generate ALL EM snapshots with segmentation overlays
====================================================
- Overlap faces  → spatially clustered (each cluster = separate overlap idx)
- Contact patches (Top1-Top6 per pair, ±20 Z-stack)
- Synapses (±20 Z-stack)
- Writes overlap_em_meta.json consumed by skeleton_em_viewer.py

Uses neurons.json for neuron IDs and colors (no hardcoded lists).
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, fcluster
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time

try:
    from cloudvolume import CloudVolume
except ImportError:
    print("[ERROR] CloudVolume not available. Install with: pip install cloud-volume")
    exit(1)


# ─── CloudVolume download with timeout + retry ──────────────────────

_cv_pool = ThreadPoolExecutor(max_workers=1)

DOWNLOAD_TIMEOUT = 60       # seconds per CloudVolume slice request
MAX_RETRIES = 3             # retry on timeout/error
RETRY_BACKOFF = 5           # seconds between retries (doubles each retry)


def _cv_fetch(vol, slc):
    """Fetch a single CloudVolume slice (runs in thread for timeout)."""
    return vol[slc]


def cv_download(vol, x0, x1, y0, y1, z0, z1, label=""):
    """Download a CloudVolume cutout with timeout + retry.

    Returns numpy array or raises after MAX_RETRIES failures.
    """
    slc = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        fut = _cv_pool.submit(_cv_fetch, vol, slc)
        try:
            return fut.result(timeout=DOWNLOAD_TIMEOUT)
        except FuturesTimeoutError:
            fut.cancel()
            if attempt < MAX_RETRIES:
                print(f"    [TIMEOUT] {label} attempt {attempt}/{MAX_RETRIES}, "
                      f"retrying in {backoff}s ...")
                time.sleep(backoff)
                backoff *= 2
            else:
                raise TimeoutError(
                    f"CloudVolume download timed out after {MAX_RETRIES} attempts: {label}")
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"    [ERROR] {label} attempt {attempt}/{MAX_RETRIES}: {e}, "
                      f"retrying in {backoff}s ...")
                time.sleep(backoff)
                backoff *= 2
            else:
                raise


# ─── Configuration ────────────────────────────────────────────────────

def load_config():
    """Load neuron IDs, names, colors from neurons.json."""
    base = os.path.dirname(os.path.abspath(__file__))
    nf = os.path.join(base, 'neurons.json')
    with open(nf, 'r') as f:
        cfg = json.load(f)

    neuron_ids = {}   # flyid -> name
    neuron_colors = {}  # name -> (r, g, b)
    for name, info in cfg['neurons'].items():
        fid = info['id']
        neuron_ids[fid] = name
        neuron_colors[name] = tuple(info['color_rgb'])
    return neuron_ids, neuron_colors


def resolve_results_dir():
    """Find the latest comprehensive_overlap_results_* directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d))
                  and d.startswith('comprehensive_overlap_results_')]
    if candidates:
        return os.path.join(base, sorted(candidates)[-1])
    return os.path.join(base, 'comprehensive_overlap_results')


# ─── EM Snapshot Creator ─────────────────────────────────────────────

def create_segmented_snapshot(em_vol, seg_vol, center_nm,
                              source_id, target_id,
                              source_name, target_name,
                              neuron_colors,
                              z_offset_slices=0, size_pixels=512):
    """
    Create EM snapshot with coloured neuron segmentation overlay.
    Returns PIL Image or None on error.
    """
    em_center_vox = np.array(center_nm) / np.array(em_vol.resolution)
    seg_center_vox = np.array(center_nm) / np.array(seg_vol.resolution)

    resolution_ratio = seg_vol.resolution[0] / em_vol.resolution[0]  # typically 2.0

    em_half = size_pixels // 2
    seg_half = int(em_half / resolution_ratio)

    em_x0 = int(em_center_vox[0] - em_half)
    em_x1 = int(em_center_vox[0] + em_half)
    em_y0 = int(em_center_vox[1] - em_half)
    em_y1 = int(em_center_vox[1] + em_half)
    em_z  = int(em_center_vox[2]) + z_offset_slices

    seg_x0 = int(seg_center_vox[0] - seg_half)
    seg_x1 = int(seg_center_vox[0] + seg_half)
    seg_y0 = int(seg_center_vox[1] - seg_half)
    seg_y1 = int(seg_center_vox[1] + seg_half)
    seg_z  = int(seg_center_vox[2]) + z_offset_slices

    try:
        label = f"{source_name}->{target_name} z={em_z}"
        em_data = cv_download(em_vol, em_x0, em_x1, em_y0, em_y1, em_z, em_z+1,
                              label=f"EM {label}")
        em_slice = em_data[:, :, 0, 0]

        seg_data = cv_download(seg_vol, seg_x0, seg_x1, seg_y0, seg_y1, seg_z, seg_z+1,
                               label=f"SEG {label}")
        seg_slice = seg_data[:, :, 0, 0]

        em_rgb = np.stack([em_slice, em_slice, em_slice], axis=-1)

        overlay_sm = np.zeros((*seg_slice.shape, 3), dtype=np.uint8)
        alpha_sm   = np.zeros(seg_slice.shape, dtype=np.float32)

        for nid, nname in [(source_id, source_name), (target_id, target_name)]:
            colour = neuron_colors.get(nname, (255, 255, 255))
            mask = (seg_slice == nid)
            overlay_sm[mask] = colour
            alpha_sm[mask] = 0.5

        if seg_slice.shape != em_slice.shape:
            zf = (em_slice.shape[0] / seg_slice.shape[0],
                  em_slice.shape[1] / seg_slice.shape[1])
            overlay = np.zeros_like(em_rgb, dtype=np.uint8)
            for c in range(3):
                overlay[:, :, c] = ndimage.zoom(overlay_sm[:, :, c], zf, order=0)
            alpha_mask = ndimage.zoom(alpha_sm, zf, order=0)
        else:
            overlay = overlay_sm
            alpha_mask = alpha_sm

        a3 = np.stack([alpha_mask]*3, axis=-1)
        blended = (em_rgb * (1 - a3) + overlay * a3).astype(np.uint8)

        img = Image.fromarray(blended)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        draw.text((10, 10), f"{source_name} \u2192 {target_name}",
                  fill=(255, 255, 255), font=font)
        if z_offset_slices != 0:
            draw.text((10, 30),
                      f"Z: {z_offset_slices:+d} ({z_offset_slices*40:+d}nm)",
                      fill=(255, 255, 0), font=font)
        return img

    except Exception as e:
        print(f"    [ERROR] snapshot: {e}")
        return None


# ─── Spatial Clustering for Overlap Faces ────────────────────────────

CLUSTER_THRESHOLD_NM = 10_000  # 10 µm — faces farther apart form new cluster


def build_overlap_plan(results_dir, neuron_ids):
    """
    Read contact_faces.csv, spatially cluster each pair's faces,
    and return a list of *sub-cluster* overlap entries.

    Returns list of dicts, each with:
        idx, source, target, source_id, target_id,
        faces  (DataFrame slice),
        centroid_xyz,
        z_lo, z_hi,
        slice_detail [{z_offset, cx, cy, n_faces, area_um2}, ...]
    """
    csv_path = os.path.join(results_dir, 'geometric_data', 'contact_faces.csv')
    if not os.path.exists(csv_path):
        print(f"[ERROR] contact_faces.csv not found at {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    print(f"[overlap plan] {len(df)} total faces across "
          f"{df.groupby(['neuron_a','neuron_b']).ngroups} directed pairs")

    # Reverse lookup: name → flyid
    name2id = {v: k for k, v in neuron_ids.items()}

    # Deduplicate: merge both directions (A→B, B→A) into one undirected pair
    # Use sorted pair key so each undirected pair is processed once
    df['pair_key'] = df.apply(
        lambda r: tuple(sorted([r['neuron_a'], r['neuron_b']])), axis=1)

    plan = []
    idx_counter = 0

    for pair_key, grp in df.groupby('pair_key'):
        na, nb = pair_key  # alphabetically sorted
        centroids = grp[['centroid_x', 'centroid_y', 'centroid_z']].values

        # --- Spatial clustering ---
        if len(centroids) > 1:
            Z = linkage(centroids, method='single')
            labels = fcluster(Z, t=CLUSTER_THRESHOLD_NM, criterion='distance')
        else:
            labels = np.array([1])

        n_clusters = int(labels.max())

        for cl in range(1, n_clusters + 1):
            mask = labels == cl
            cl_faces = grp[mask].copy()

            cx = cl_faces['centroid_x'].values
            cy = cl_faces['centroid_y'].values
            cz = cl_faces['centroid_z'].values

            # Z in nm → z-slice offset: round(z / 40)
            z_slices = np.round(cz / 40).astype(int)
            z_lo = int(z_slices.min())
            z_hi = int(z_slices.max())

            # Per-z-slice detail (used for EM centroids per slice)
            slice_detail = []
            for z_off in range(z_lo, z_hi + 1):
                zmask = z_slices == z_off
                if zmask.sum() == 0:
                    continue
                slice_detail.append({
                    'z_offset': int(z_off),
                    'cx': float(cx[zmask].mean()),
                    'cy': float(cy[zmask].mean()),
                    'n_faces': int(zmask.sum()),
                    'area_um2': float(cl_faces['face_area_um2'].values[zmask].sum()),
                })

            global_cx = float(cx.mean())
            global_cy = float(cy.mean())
            global_cz = float(cz.mean())

            plan.append({
                'idx': idx_counter,
                'source': na,
                'target': nb,
                'source_id': name2id.get(na),
                'target_id': name2id.get(nb),
                'x': global_cx,
                'y': global_cy,
                'z': global_cz,
                'z_lo': z_lo,
                'z_hi': z_hi,
                'n_slices': len(slice_detail),
                'total_faces': int(mask.sum()),
                'total_area_um2': float(cl_faces['face_area_um2'].sum()),
                'slice_detail': slice_detail,
                'cluster_label': cl,
                'n_clusters_in_pair': n_clusters,
            })
            idx_counter += 1

    n_undirected = df['pair_key'].nunique()
    print(f"[overlap plan] {idx_counter} sub-clusters "
          f"(from {n_undirected} undirected pairs)")
    return plan


# ─── Main ────────────────────────────────────────────────────────────

def main():
    neuron_ids, neuron_colors = load_config()
    results_dir = resolve_results_dir()
    em_snap_dir = os.path.join(results_dir, 'em_snaps')
    os.makedirs(em_snap_dir, exist_ok=True)

    print("=" * 70)
    print("Generate EM Snapshots – Overlaps (clustered) + Contacts + Synapses")
    print("=" * 70)

    # ── Connect to CloudVolume ──────────────────────────────────
    print("\n[1/6] Connecting to data sources ...")
    em_vol = CloudVolume(
        'https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14',
        mip=1, use_https=True, progress=False)
    seg_vol = CloudVolume(
        'precomputed://gs://flywire_v141_m783',
        mip=0, use_https=True, progress=False)
    print("  [OK] Connected to EM and segmentation volumes")

    # ════════════════════════════════════════════════════════════
    # A.  OVERLAP FACES  (spatially clustered)
    # ════════════════════════════════════════════════════════════
    print("\n[2/6] Building spatially-clustered overlap plan ...")
    plan = build_overlap_plan(results_dir, neuron_ids)

    if plan:
        total_overlap_imgs = sum(p['n_slices'] for p in plan)
        print(f"  Will generate up to {total_overlap_imgs} overlap images "
              f"({len(plan)} sub-clusters)")

        # Compute which overlap indices are in the current plan
        plan_idxs = set(p['idx'] for p in plan)
        # Only remove orphaned overlap images whose idx is NOT in the current plan
        # This allows resuming after a crash without re-downloading everything
        old_ov_files = [f for f in os.listdir(em_snap_dir)
                        if f.startswith('overlap_')]
        orphaned = []
        for f in old_ov_files:
            try:
                file_idx = int(f.split('_')[1])
                if file_idx not in plan_idxs:
                    orphaned.append(f)
            except (ValueError, IndexError):
                orphaned.append(f)  # malformed filename
        if orphaned:
            print(f"  Removing {len(orphaned)} orphaned overlap images "
                  f"(keeping {len(old_ov_files) - len(orphaned)} valid) ...")
            for f in orphaned:
                os.remove(os.path.join(em_snap_dir, f))
        else:
            existing = len(old_ov_files)
            if existing:
                print(f"  Found {existing} existing overlap images — will skip those")

        print("\n[3/6] Downloading overlap EM snapshots ...")
        overlap_ok = 0
        overlap_new = 0
        overlap_skip = 0

        for entry in tqdm(plan, desc="Overlap clusters"):
            idx = entry['idx']
            src_id = entry['source_id']
            tgt_id = entry['target_id']
            src_name = entry['source']
            tgt_name = entry['target']

            if src_id is None or tgt_id is None:
                print(f"  [WARN] No fly-ID for {src_name}/{tgt_name}, skipping")
                continue

            for sd in entry['slice_detail']:
                z_off = sd['z_offset']
                center_nm = (sd['cx'], sd['cy'], z_off * 40)

                # File naming
                if z_off == entry['z_lo']:
                    # treat the first slice as "center" (segmented)
                    fname_center = f"overlap_{idx}_segmented.png"
                    opath_c = os.path.join(em_snap_dir, fname_center)
                    if not os.path.exists(opath_c):
                        img = create_segmented_snapshot(
                            em_vol, seg_vol, center_nm,
                            src_id, tgt_id, src_name, tgt_name,
                            neuron_colors, z_offset_slices=0,
                            size_pixels=512)
                        if img is not None:
                            img.save(opath_c, 'PNG')
                            overlap_new += 1
                    else:
                        overlap_skip += 1

                # Z-offset file (relative to cluster z_lo)
                rel_z = z_off - entry['z_lo']
                sign = '+' if rel_z >= 0 else '-'
                fname = f"overlap_{idx}_z{sign}{abs(rel_z):03d}.png"
                opath = os.path.join(em_snap_dir, fname)

                if os.path.exists(opath):
                    overlap_skip += 1
                    continue

                img = create_segmented_snapshot(
                    em_vol, seg_vol, center_nm,
                    src_id, tgt_id, src_name, tgt_name,
                    neuron_colors, z_offset_slices=0,
                    size_pixels=512)
                if img is not None:
                    img.save(opath, 'PNG')
                    overlap_new += 1

            overlap_ok += 1

        print(f"\n  [OK] Overlaps: {overlap_ok}/{len(plan)} clusters, "
              f"{overlap_new} new + {overlap_skip} skipped")

        # Write overlap_em_meta.json
        meta_path = os.path.join(results_dir, 'overlap_em_meta.json')
        meta_out = []
        for entry in plan:
            # Convert slice_detail z_offsets to relative (z_lo-based)
            rel_slices = []
            for sd in entry['slice_detail']:
                rel_z = sd['z_offset'] - entry['z_lo']
                rel_slices.append({
                    'z_offset': rel_z,
                    'cx': sd['cx'],
                    'cy': sd['cy'],
                    'n_faces': sd['n_faces'],
                    'area_um2': sd['area_um2'],
                })

            meta_out.append({
                'idx': entry['idx'],
                'source': entry['source'],
                'target': entry['target'],
                'x': entry['x'],
                'y': entry['y'],
                'z': entry['z'],
                'z_base_nm': entry['z_lo'] * 40,
                'z_lo': 0,
                'z_hi': entry['z_hi'] - entry['z_lo'],
                'n_slices': entry['n_slices'],
                'total_faces': entry['total_faces'],
                'total_area_um2': entry['total_area_um2'],
                'cluster_label': entry['cluster_label'],
                'n_clusters_in_pair': entry['n_clusters_in_pair'],
                'slice_detail': rel_slices,
            })

        with open(meta_path, 'w') as f:
            json.dump(meta_out, f, indent=2)
        print(f"  [OK] Wrote {meta_path}  ({len(meta_out)} entries)")
    else:
        print("  [WARN] No overlap plan – skipping overlap EM snapshots")

    # ════════════════════════════════════════════════════════════
    # B.  CONTACT PATCHES  (Top1–Top6 per pair, ±20 Z-stack)
    # ════════════════════════════════════════════════════════════
    print("\n[4/6] Loading contact patch data ...")
    contacts_file = os.path.join(results_dir, 'all_results_combined.csv')
    if not os.path.exists(contacts_file):
        print(f"  [SKIP] {contacts_file} not found")
    else:
        cdf = pd.read_csv(contacts_file)
        cdf = cdf[cdf['Has_Contact'] == True]
        print(f"  Found {len(cdf)} contact pairs")

        # Reverse lookup name → flyid
        name2id = {v: k for k, v in neuron_ids.items()}

        all_patches = []
        patch_idx = 0
        for _, row in cdf.iterrows():
            sn = row['Source_Neuron']
            tn = row['Target_Neuron']
            sid = name2id.get(sn)
            tid = name2id.get(tn)
            if sid is None or tid is None:
                continue
            for pn in range(1, 7):
                xcol = f'Top{pn}_Patch_Centroid_X'
                if xcol in cdf.columns and not pd.isna(row.get(xcol)):
                    all_patches.append({
                        'idx': patch_idx,
                        'source_name': sn, 'target_name': tn,
                        'source_id': sid, 'target_id': tid,
                        'center': (row[xcol],
                                   row[f'Top{pn}_Patch_Centroid_Y'],
                                   row[f'Top{pn}_Patch_Centroid_Z']),
                        'patch_num': pn,
                    })
                    patch_idx += 1

        print(f"  Total contact patches: {len(all_patches)} "
              f"→ {len(all_patches)*41} Z-stack images")

        print("\n[5/6] Downloading contact EM snapshots ...")
        c_new = c_skip = 0
        for patch in tqdm(all_patches, desc="Contacts"):
            pidx = patch['idx']
            for z_off in range(-20, 21):
                if z_off == 0:
                    fname = f"contact_{pidx}_segmented.png"
                else:
                    fname = f"contact_{pidx}_z{z_off:+04d}.png"
                opath = os.path.join(em_snap_dir, fname)
                if os.path.exists(opath):
                    c_skip += 1
                    continue
                img = create_segmented_snapshot(
                    em_vol, seg_vol, patch['center'],
                    patch['source_id'], patch['target_id'],
                    patch['source_name'], patch['target_name'],
                    neuron_colors, z_offset_slices=z_off,
                    size_pixels=512)
                if img is not None:
                    img.save(opath, 'PNG')
                    c_new += 1
        print(f"  [OK] Contacts: {c_new} new + {c_skip} skipped")

    # ════════════════════════════════════════════════════════════
    # C.  SYNAPSES  (±20 Z-stack per synapse)
    # ════════════════════════════════════════════════════════════
    print("\n[6/6] Loading synapse data ...")
    syn_file = os.path.join(results_dir, 'synapses.csv')
    if not os.path.exists(syn_file):
        print(f"  [SKIP] {syn_file} not found")
    else:
        sdf = pd.read_csv(syn_file).dropna(subset=['x', 'y', 'z'])

        # Resolve column names
        if 'pre_type' in sdf.columns:
            sdf['source'] = sdf['pre_type']
            sdf['target'] = sdf['post_type']
        elif 'pre' in sdf.columns:
            sdf['source'] = sdf['pre']
            sdf['target'] = sdf['post']
        else:
            print("  [SKIP] Cannot find source/target columns")
            return

        # Include all synapses (all neuron groups)
        name2id = {v: k for k, v in neuron_ids.items()}
        valid_names = set(name2id.keys())
        sdf = sdf[sdf['source'].isin(valid_names) & sdf['target'].isin(valid_names)]
        print(f"  Found {len(sdf)} synapses for EM stacks")

        s_new = s_skip = 0
        for orig_idx, row in tqdm(sdf.iterrows(), total=len(sdf), desc="Synapses"):
            sn = row['source']
            tn = row['target']
            sid = name2id.get(sn)
            tid = name2id.get(tn)
            if sid is None or tid is None:
                continue
            center_nm = (row['x'], row['y'], row['z'])
            for z_off in range(-20, 21):
                if z_off == 0:
                    fname = f"synapse_{orig_idx}_segmented.png"
                else:
                    fname = f"synapse_{orig_idx}_z{z_off:+04d}.png"
                opath = os.path.join(em_snap_dir, fname)
                if os.path.exists(opath):
                    s_skip += 1
                    continue
                img = create_segmented_snapshot(
                    em_vol, seg_vol, center_nm,
                    sid, tid, sn, tn,
                    neuron_colors, z_offset_slices=z_off,
                    size_pixels=512)
                if img is not None:
                    img.save(opath, 'PNG')
                    s_new += 1
        print(f"  [OK] Synapses: {s_new} new + {s_skip} skipped")

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    total_files = len([f for f in os.listdir(em_snap_dir) if f.endswith('.png')])
    print(f"  Total PNG files in em_snaps/: {total_files}")
    print(f"  Resolution: 512×512 px  (4096 nm × 4096 nm)")
    print(f"  Overlap clusters: {len(plan)} (spatially separated)")
    print("=" * 70)


if __name__ == "__main__":
    main()
