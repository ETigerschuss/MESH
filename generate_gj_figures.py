#!/usr/bin/env python
"""
Putative gap-junction composite figures (publication panels)
============================================================
For each motor neuron (MOT_L/R, MOS_L/R) builds one composite figure:

  * centre : the neuron skeleton (in its config colour) with partner-coloured
             overlap "blobs" (HS for MOT; HS + VS1-4 for MOS)
  * around : up to N_SIDE EM insets on EACH of the four sides (left/right/top/
             bottom). Each inset is the densest contact with a partner that is
             >= MIN_SYN_DIST_NM from any annotated chemical synapse (so the site
             is a putative *gap junction*, not a chemical synapse). Each inset is
             labelled with the partner only and linked by a connector line to its
             location (ringed) on the skeleton. Insets are placed on the side
             closest to their site so connector lines do not cross.

EM is the FlyWire-aligned volume at 8 nm (the only EM that registers with the
v14.1 segmentation); for native-4 nm ultrastructure use the printed coordinates
in FlyWire's Neuroglancer.

Inputs  : latest comprehensive_overlap_results_* dir (contact_patches.csv,
          synapses.csv, neuron_meshes/) + the active neuron config.
Outputs : <results_dir>/gj_figures/FIGURE_<motor>.pdf (+ .png)
          <results_dir>/gj_figures/sites/SITE_<motor>__<partner>_<k>.png
          <results_dir>/gj_figures/gj_sites.json
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from mesh_config import load_config
from proofreading import drop_rejected

# Partner groups whose contacts are eligible as GJ examples / drawn as blobs.
BLOB_GROUPS = {"MOT": {"HS"}, "MOS": {"HS", "VS"}}
# Partners for the compact 3-example figure (one inset each, stacked on the right).
CANONICAL = {"MOT": ["HSN", "HSE", "HSS"], "MOS": ["HSN", "VS2", "VS3"]}
N_SIDE = 3                  # insets per side (left/right/top/bottom)
N_TOTAL = 4 * N_SIDE        # 12 insets per figure
MAX_PER_PARTNER = 8         # cap sites from any single partner cell (variety)
MIN_SYN_DIST_NM = 2000      # contact site must be >= this from any chem synapse
DEDUPE_NM = 2000            # drop sites closer than this to a kept one (finer -> more sites)
DENSITY_BIN_NM = 1200
INSET_HALF_PX = 150         # EM inset half-size in 8 nm px (~2.4 um field)
SKELETON_ALPHA = 0.55       # higher -> neuron more visible

# Confirmed gap junctions to force-include (motor, partner-type) -> nm centroid.
CONFIRMED_SITES = {
    ("MOT_R", "HSN"): [618792, 267816, 202720],   # voxel 154698,66954,5068 @4x4x40
}

EM_CLOUDPATH = "https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14"
EM_MIP = 1
SEG_CLOUDPATH = "precomputed://gs://flywire_v141_m783"
SEG_MIP = 0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_results_dir():
    cands = [d for d in os.listdir(SCRIPT_DIR)
             if os.path.isdir(os.path.join(SCRIPT_DIR, d))
             and d.startswith("comprehensive_overlap_results_")]
    if not cands:
        sys.exit("ERROR: no comprehensive_overlap_results_* directory found.")
    return os.path.join(SCRIPT_DIR, sorted(cands)[-1])


def _clahe(img):
    from skimage import exposure
    f = img.astype(np.float32) / max(1, int(img.max()))
    return (exposure.equalize_adapthist(f, clip_limit=0.02) * 255).astype(np.uint8)


def _syn_dist(syn, a, b, cen):
    m = (((syn.pre_type == a) & (syn.post_type == b)) |
         ((syn.pre_type == b) & (syn.post_type == a)))
    if not m.any():
        return float("inf")
    return float(np.linalg.norm(syn.loc[m, ["x", "y", "z"]].values - cen, axis=1).min())


def _pair_sites(pat, syn, a, b, k):
    """Top-k densest contact bins between a,b that avoid chemical synapses."""
    m = (((pat.neuron_a == a) & (pat.neuron_b == b)) |
         ((pat.neuron_a == b) & (pat.neuron_b == a)))
    sub = pat[m]
    if len(sub) == 0:
        return []
    bins = list(zip((sub.patch_centroid_x // DENSITY_BIN_NM).astype(int),
                    (sub.patch_centroid_y // DENSITY_BIN_NM).astype(int),
                    (sub.patch_centroid_z // DENSITY_BIN_NM).astype(int)))
    g = (sub.assign(_b=bins).groupby("_b")
            .agg(area=("patch_area_um2", "sum"), cx=("patch_centroid_x", "mean"),
                 cy=("patch_centroid_y", "mean"), cz=("patch_centroid_z", "mean"))
            .reset_index().sort_values("area", ascending=False))
    out = []
    for _, r in g.iterrows():
        cen = np.array([r.cx, r.cy, r.cz])
        sd = _syn_dist(syn, a, b, cen)
        if sd >= MIN_SYN_DIST_NM:
            out.append({"partner": b, "center": cen, "area": float(r.area), "syn": sd})
        if len(out) >= k:
            break
    return out


def _curate(motor, partners, pat, syn, grp):
    pool = []
    for p in partners:
        pool.extend(_pair_sites(pat, syn, motor, p, MAX_PER_PARTNER))
    # force-include confirmed sites for this motor
    forced = []
    for (mm, ptype), nm in CONFIRMED_SITES.items():
        if mm == motor:
            full = next((p for p in partners if p.split("_")[0] == ptype), None)
            if full:
                forced.append({"partner": full, "center": np.array(nm, float),
                               "area": 999.0, "syn": float("inf"), "confirmed": True})
    pool = forced + sorted(pool, key=lambda s: -s["area"])
    # spatial dedupe + per-partner cap
    kept, per = [], {}
    for s in pool:
        if any(np.linalg.norm(s["center"] - k2["center"]) < DEDUPE_NM for k2 in kept):
            continue
        if per.get(s["partner"], 0) >= MAX_PER_PARTNER and not s.get("confirmed"):
            continue
        kept.append(s)
        per[s["partner"]] = per.get(s["partner"], 0) + 1
        if len(kept) >= N_TOTAL:
            break
    return kept


def _assign_sides(sites, cx, cy):
    """Assign sites to left/right/top/bottom (N_SIDE each) minimising crossings.
    Display y increases downward (skeleton y-axis is inverted)."""
    for s in sites:
        dx = s["center"][0] - cx
        dy = s["center"][1] - cy          # data-y: larger = lower on screen
        s["theta"] = math.atan2(-dy, dx)  # screen angle (up = +)
    sides = {"right": [], "top": [], "left": [], "bottom": []}

    def sector(t):
        d = math.degrees(t)
        if -45 <= d < 45:
            return "right"
        if 45 <= d < 135:
            return "top"
        if d >= 135 or d < -135:
            return "left"
        return "bottom"
    for s in sorted(sites, key=lambda s: -s["area"]):
        order = sorted(["right", "top", "left", "bottom"],
                       key=lambda sd: abs(((math.degrees(s["theta"]) -
                                            {"right": 0, "top": 90, "left": 180, "bottom": -90}[sd] + 180) % 360) - 180))
        for sd in order:
            if len(sides[sd]) < N_SIDE:
                sides[sd].append(s)
                break
    # order insets along each edge by position so connectors don't cross
    sides["left"].sort(key=lambda s: s["center"][1])
    sides["right"].sort(key=lambda s: s["center"][1])
    sides["top"].sort(key=lambda s: s["center"][0])
    sides["bottom"].sort(key=lambda s: s["center"][0])
    return sides


def _inset(em, seg, nid, ncol, a, b, cen):
    from scipy import ndimage
    er, sr = np.array(em.resolution), np.array(seg.resolution)
    h = INSET_HALF_PX
    c = (np.array(cen) / er).astype(int)
    emc = np.asarray(em[c[0]-h:c[0]+h, c[1]-h:c[1]+h, c[2]])[:, :, 0, 0]
    sc = (np.array(cen) / sr).astype(int)
    sh = max(6, int(h * er[0] / sr[0]))
    sg = np.asarray(seg[sc[0]-sh:sc[0]+sh, sc[1]-sh:sc[1]+sh, sc[2]])[:, :, 0, 0]
    enh = _clahe(emc)
    rgb = np.stack([enh]*3, -1).astype(float)
    for nm in (a, b):
        mask = ndimage.zoom((sg == nid[nm]).astype(float),
                            (enh.shape[0]/sg.shape[0], enh.shape[1]/sg.shape[1]), order=0) > 0.5
        rgb[mask] = 0.78*rgb[mask] + 0.22*np.array(ncol[nm])
    return rgb.astype(np.uint8)


def _build(motor, sides, imgs, mesh_obj, chex, blob_df, syn, out_base, show_synapses=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    import navis, trimesh
    tm = trimesh.load(mesh_obj, process=False)
    mn = navis.downsample_neuron(navis.MeshNeuron(tm, name=motor), downsampling_factor=12)
    fig = plt.figure(figsize=(19, 19), facecolor="white")
    axS = fig.add_axes([0.255, 0.235, 0.49, 0.53])
    navis.plot2d(mn, color=chex[motor], alpha=SKELETON_ALPHA, view=("x", "-y"),
                 method="2d", ax=axS, linewidth=0.6)
    for prt, g in blob_df.groupby("partner"):
        axS.scatter(g.patch_centroid_x, g.patch_centroid_y, s=150, c=chex[prt],
                    alpha=0.14, edgecolors="none", zorder=80)
        axS.scatter(g.patch_centroid_x, g.patch_centroid_y, s=38, c=chex[prt],
                    alpha=0.40, edgecolors="none", zorder=81)
    if show_synapses:
        sm = syn[(syn.pre_type == motor) | (syn.post_type == motor)].copy()
        sm["pgrp"] = [b if a == motor else a for a, b in zip(sm.pre_group, sm.post_group)]
        sm = sm[sm.pgrp.isin(BLOB_GROUPS[motor[:3]])]   # only the same partners as the blobs
        if len(sm):
            axS.scatter(sm.x, sm.y, s=20, c="#FFE000", edgecolors="#6b5500",
                        linewidths=0.4, alpha=0.95, zorder=150)
    V = tm.vertices
    axS.set_xlim(V[:, 0].min(), V[:, 0].max())
    axS.set_ylim(V[:, 1].max(), V[:, 1].min())
    axS.set_aspect("equal"); axS.axis("off")
    axS.set_title(motor, fontsize=24, color=chex[motor], fontweight="bold", loc="left")

    iw, ih = 0.165, 0.165
    # inset rectangles per side slot + which inset-edge faces the skeleton
    coords = {
        "left":   [(0.035, y, (1.0, 0.5)) for y in (0.60, 0.40, 0.20)],
        "right":  [(0.800, y, (0.0, 0.5)) for y in (0.60, 0.40, 0.20)],
        "top":    [(x, 0.805, (0.5, 0.0)) for x in (0.30, 0.46, 0.62)],
        "bottom": [(x, 0.030, (0.5, 1.0)) for x in (0.30, 0.46, 0.62)],
    }
    for side, slots in coords.items():
        for s, (x, y, edge) in zip(sides[side], slots):
            ax = fig.add_axes([x, y, iw, ih]); ax.imshow(imgs[id(s)]); ax.axis("off")
            ax.set_title(s["partner"].split("_")[0] + (" ★" if s.get("confirmed") else ""),
                         fontsize=14, fontweight="bold", color=chex[s["partner"]])
            cen = s["center"]
            axS.scatter([cen[0]], [cen[1]], s=300, facecolors="none",
                        edgecolors="#111", linewidths=2.1, zorder=200)
            fig.add_artist(ConnectionPatch(xyA=(cen[0], cen[1]), coordsA=axS.transData,
                                           xyB=edge, coordsB=ax.transAxes, color="#333", lw=1.2))
    fig.savefig(out_base + ".pdf", facecolor="white")
    fig.savefig(out_base + ".png", dpi=130, facecolor="white")
    plt.close(fig)


def _canonical_sites(motor, partners, pat, syn):
    """Best synapse-free site for each canonical partner (confirmed preferred)."""
    side = motor.split("_")[1]
    out = []
    for ptype in CANONICAL[motor[:3]]:
        cell = ptype + "_" + side
        if cell not in partners:
            continue
        conf = CONFIRMED_SITES.get((motor, ptype))
        if conf is not None:
            out.append({"partner": cell, "center": np.array(conf, float),
                        "area": 999.0, "confirmed": True})
            continue
        found = _pair_sites(pat, syn, motor, cell, 1)
        if found:
            out.append(found[0])
    return out


def _build_simple(motor, sites, imgs, mesh_obj, chex, blob_df, syn, out_base, show_synapses=False):
    """Compact figure: skeleton + 3 partner insets stacked on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    import navis, trimesh
    tm = trimesh.load(mesh_obj, process=False)
    mn = navis.downsample_neuron(navis.MeshNeuron(tm, name=motor), downsampling_factor=12)
    fig = plt.figure(figsize=(16, 11), facecolor="white")
    axS = fig.add_axes([0.04, 0.05, 0.58, 0.90])
    navis.plot2d(mn, color=chex[motor], alpha=SKELETON_ALPHA, view=("x", "-y"),
                 method="2d", ax=axS, linewidth=0.6)
    for prt, g in blob_df.groupby("partner"):
        axS.scatter(g.patch_centroid_x, g.patch_centroid_y, s=150, c=chex[prt],
                    alpha=0.14, edgecolors="none", zorder=80)
        axS.scatter(g.patch_centroid_x, g.patch_centroid_y, s=38, c=chex[prt],
                    alpha=0.40, edgecolors="none", zorder=81)
    if show_synapses:
        sm = syn[(syn.pre_type == motor) | (syn.post_type == motor)].copy()
        sm["pgrp"] = [b if a == motor else a for a, b in zip(sm.pre_group, sm.post_group)]
        sm = sm[sm.pgrp.isin(BLOB_GROUPS[motor[:3]])]   # only the same partners as the blobs
        if len(sm):
            axS.scatter(sm.x, sm.y, s=20, c="#FFE000", edgecolors="#6b5500",
                        linewidths=0.4, alpha=0.95, zorder=150)
    V = tm.vertices
    axS.set_xlim(V[:, 0].min(), V[:, 0].max())
    axS.set_ylim(V[:, 1].max(), V[:, 1].min())
    axS.set_aspect("equal"); axS.axis("off")
    axS.set_title(motor, fontsize=22, color=chex[motor], fontweight="bold", loc="left")
    ys = [0.69, 0.40, 0.11]
    for k, s in enumerate(sites):
        ax = fig.add_axes([0.66, ys[k], 0.30, 0.26]); ax.imshow(imgs[id(s)]); ax.axis("off")
        ax.set_title(s["partner"].split("_")[0] + (" ★" if s.get("confirmed") else ""),
                     fontsize=15, fontweight="bold", color=chex[s["partner"]])
        cen = s["center"]
        axS.scatter([cen[0]], [cen[1]], s=320, facecolors="none",
                    edgecolors="#111", linewidths=2.2, zorder=200)
        fig.add_artist(ConnectionPatch(xyA=(cen[0], cen[1]), coordsA=axS.transData,
                                       xyB=(0, 0.5), coordsB=ax.transAxes, color="#333", lw=1.3))
    fig.savefig(out_base + ".pdf", facecolor="white")
    fig.savefig(out_base + ".png", dpi=130, facecolor="white")
    plt.close(fig)


def main():
    from cloudvolume import CloudVolume
    from PIL import Image
    cfg, cfg_path = load_config()
    neurons = cfg["neurons"]
    nid = {n: i["id"] for n, i in neurons.items()}
    chex = {n: i["color_hex"] for n, i in neurons.items()}
    ncol = {n: tuple(i["color_rgb"]) for n, i in neurons.items()}
    grp = {n: i["group"] for n, i in neurons.items()}
    rd = _find_results_dir()
    print(f"Config: {os.path.basename(str(cfg_path))} | Results: {os.path.basename(rd)}")
    out_dir = os.path.join(rd, "gj_figures"); sites_dir = os.path.join(out_dir, "sites")
    os.makedirs(sites_dir, exist_ok=True)
    pat = pd.read_csv(os.path.join(rd, "geometric_data", "contact_patches.csv"),
                      usecols=["neuron_a", "neuron_b", "patch_centroid_x", "patch_centroid_y",
                               "patch_centroid_z", "patch_area_um2"])
    pat = drop_rejected(pat, rd)
    sp = os.path.join(rd, "synapses.csv")
    syn = pd.read_csv(sp) if os.path.exists(sp) else pd.DataFrame(columns=["pre_type", "post_type", "x", "y", "z"])
    em = CloudVolume(EM_CLOUDPATH, mip=EM_MIP, use_https=True, progress=False)
    seg = CloudVolume(SEG_CLOUDPATH, mip=SEG_MIP, use_https=True, progress=False)
    motors = [m for m in ["MOT_L", "MOT_R", "MOS_L", "MOS_R"] if m in neurons]
    all_sites = {}
    for motor in motors:
        side = motor.split("_")[1]
        partners = [n for n, info in neurons.items()
                    if info["group"] in BLOB_GROUPS[motor[:3]] and n.endswith("_" + side)]
        print(f"\n[{motor}] curating sites from {len(partners)} partners ...")
        sites = _curate(motor, partners, pat, syn, grp)
        print(f"  {len(sites)} sites kept")
        cx = pat_cx = np.mean([s["center"][0] for s in sites]) if sites else 0
        cy = pat_cy = np.mean([s["center"][1] for s in sites]) if sites else 0
        imgs = {}
        for i, s in enumerate(sites):
            img = _inset(em, seg, nid, ncol, motor, s["partner"], s["center"])
            imgs[id(s)] = img
            Image.fromarray(img).save(os.path.join(sites_dir, f"SITE_{motor}__{s['partner']}_{i}.png"))
            all_sites[f"{motor}_{i}_{s['partner']}"] = {
                "partner": s["partner"], "center_nm": s["center"].astype(int).tolist(),
                "voxel_4x4x40": [int(round(s["center"][0]/4)), int(round(s["center"][1]/4)), int(round(s["center"][2]/40))],
                "contact_area_um2": round(s["area"], 2),
                "confirmed": bool(s.get("confirmed", False))}
        sides = _assign_sides(sites, cx, cy)
        blob = pat[(pat.neuron_a == motor) | (pat.neuron_b == motor)].copy()
        blob["partner"] = [b if a == motor else a for a, b in zip(blob.neuron_a, blob.neuron_b)]
        blob = blob[blob.partner.map(lambda n: n in grp and grp[n] in BLOB_GROUPS[motor[:3]])]
        mesh_obj = os.path.join(rd, "neuron_meshes", f"{nid[motor]}.obj")
        if not os.path.exists(mesh_obj):
            print("  mesh missing, skipping figure"); continue
        base_ext = os.path.join(out_dir, f"FIGURE_{motor}_extended")
        _build(motor, sides, imgs, mesh_obj, chex, blob, syn, base_ext, show_synapses=False)
        _build(motor, sides, imgs, mesh_obj, chex, blob, syn, base_ext + "_synapses", show_synapses=True)
        # compact 3-example figure (canonical partners, right side)
        can = _canonical_sites(motor, partners, pat, syn)
        can_imgs = {id(s): _inset(em, seg, nid, ncol, motor, s["partner"], s["center"]) for s in can}
        base_s = os.path.join(out_dir, f"FIGURE_{motor}_3examples")
        _build_simple(motor, can, can_imgs, mesh_obj, chex, blob, syn, base_s, show_synapses=False)
        _build_simple(motor, can, can_imgs, mesh_obj, chex, blob, syn, base_s + "_synapses", show_synapses=True)
        print(f"  -> {motor}: extended + 3examples (+ _synapses variants) [{', '.join(s['partner'] for s in can)}]")
    json.dump(all_sites, open(os.path.join(out_dir, "gj_sites.json"), "w"), indent=2)
    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
