#!/usr/bin/env python
"""
Putative gap-junction composite figures (publication panels)
============================================================
For each motor neuron (MOT_L/R, MOS_L/R) builds one composite figure:

  * left  : the neuron skeleton (in its config color) with partner-coloured
            overlap "blobs" (HS for MOT; HS + VS1-4 for MOS)
  * right : 3 small EM insets, each the densest contact with a chosen partner
            that lies >= MIN_SYN_DIST_NM from any annotated chemical synapse
            (so the site is a putative *gap junction*, not a chemical synapse).
            Each inset is labelled with the partner only and is linked by a
            connector line to its location (ringed) on the skeleton.

EM is the FlyWire-aligned volume at 8 nm (the only EM that registers with the
v14.1 segmentation); for native-4 nm ultrastructure use the printed coordinates
in FlyWire's Neuroglancer.

Inputs  : latest comprehensive_overlap_results_* dir (contact_patches.csv,
          synapses.csv, neuron_meshes/) + the active neuron config.
Outputs : <results_dir>/gj_figures/FIGURE_<motor>.png
          <results_dir>/gj_figures/sites/SITE_<motor>__<partner>.png
          <results_dir>/gj_figures/gj_sites.json  (centroids, areas, syn dist)

Usage:  python generate_gj_figures.py
"""

import json
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

# ── Which partners to show per motor neuron (3 each) ──────────────────
# Edit here to change the example partners shown in each figure.
PARTNER_PLAN = {
    "MOT_L": ["HSN_L", "HSE_L", "HSS_L"],
    "MOT_R": ["HSN_R", "HSE_R", "HSS_R"],
    "MOS_L": ["HSN_L", "VS2_L", "VS3_L"],
    "MOS_R": ["HSN_R", "VS2_R", "VS3_R"],
}
# Groups whose overlaps are drawn as blobs on each motor's skeleton.
BLOB_GROUPS = {"MOT": {"HS"}, "MOS": {"HS", "VS"}}

MIN_SYN_DIST_NM = 2000     # contact site must be >= this from any chem synapse
DENSITY_BIN_NM = 1500      # bin size for finding the densest contact region
INSET_HALF_PX = 150        # EM inset half-size in 8 nm pixels (~2.4 um field)

# FlyWire-aligned EM (8 nm, v14.1) + segmentation — same sources as the viewer.
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


def _densest_site(pat, syn, a, b):
    """Densest contact region between a and b that avoids chemical synapses."""
    m = (((pat.neuron_a == a) & (pat.neuron_b == b)) |
         ((pat.neuron_a == b) & (pat.neuron_b == a)))
    sub = pat[m]
    if len(sub) == 0:
        return None
    bins = list(zip((sub.patch_centroid_x // DENSITY_BIN_NM).astype(int),
                    (sub.patch_centroid_y // DENSITY_BIN_NM).astype(int),
                    (sub.patch_centroid_z // DENSITY_BIN_NM).astype(int)))
    g = (sub.assign(_b=bins)
            .groupby("_b")
            .agg(area=("patch_area_um2", "sum"),
                 cx=("patch_centroid_x", "mean"),
                 cy=("patch_centroid_y", "mean"),
                 cz=("patch_centroid_z", "mean"))
            .reset_index().sort_values("area", ascending=False))
    for _, r in g.iterrows():
        cen = np.array([r.cx, r.cy, r.cz])
        if _syn_dist(syn, a, b, cen) >= MIN_SYN_DIST_NM:
            return float(r.area), cen, _syn_dist(syn, a, b, cen)
    r = g.iloc[0]
    cen = np.array([r.cx, r.cy, r.cz])
    return float(r.area), cen, _syn_dist(syn, a, b, cen)


def _inset_image(em, seg, nid, ncol, a, b, cen):
    """Aligned 8 nm EM crop with faint colour overlay of the two cells."""
    from scipy import ndimage
    er, sr = np.array(em.resolution), np.array(seg.resolution)
    h = INSET_HALF_PX
    c = (np.array(cen) / er).astype(int)
    emc = np.asarray(em[c[0]-h:c[0]+h, c[1]-h:c[1]+h, c[2]])[:, :, 0, 0]
    sc = (np.array(cen) / sr).astype(int)
    sh = max(6, int(h * er[0] / sr[0]))
    sg = np.asarray(seg[sc[0]-sh:sc[0]+sh, sc[1]-sh:sc[1]+sh, sc[2]])[:, :, 0, 0]
    enh = _clahe(emc)
    rgb = np.stack([enh] * 3, -1).astype(float)
    for nm in (a, b):
        mask = ndimage.zoom((sg == nid[nm]).astype(float),
                            (enh.shape[0] / sg.shape[0], enh.shape[1] / sg.shape[1]),
                            order=0) > 0.5
        rgb[mask] = 0.78 * rgb[mask] + 0.22 * np.array(ncol[nm])
    return rgb.astype(np.uint8)


def _build_figure(motor, partners, sites, inset_imgs, mesh_obj, chex, blob_df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    import navis
    import trimesh

    tm = trimesh.load(mesh_obj, process=False)
    mn = navis.downsample_neuron(navis.MeshNeuron(tm, name=motor), downsampling_factor=12)
    fig = plt.figure(figsize=(16, 11), facecolor="white")
    axS = fig.add_axes([0.04, 0.05, 0.58, 0.90])
    navis.plot2d(mn, color=chex[motor], alpha=0.28, view=("x", "-y"),
                 method="2d", ax=axS, linewidth=0.5)
    for prt, grp in blob_df.groupby("partner"):
        axS.scatter(grp.patch_centroid_x, grp.patch_centroid_y, s=170,
                    c=chex[prt], alpha=0.16, edgecolors="none", zorder=80)
        axS.scatter(grp.patch_centroid_x, grp.patch_centroid_y, s=42,
                    c=chex[prt], alpha=0.42, edgecolors="none", zorder=81)
    V = tm.vertices
    axS.set_xlim(V[:, 0].min(), V[:, 0].max())
    axS.set_ylim(V[:, 1].max(), V[:, 1].min())
    axS.set_aspect("equal")
    axS.axis("off")
    axS.set_title(motor, fontsize=20, color=chex[motor], fontweight="bold", loc="left")

    ys = [0.69, 0.40, 0.11]
    for k, p in enumerate(partners):
        ax = fig.add_axes([0.66, ys[k], 0.30, 0.26])
        ax.imshow(inset_imgs[p])
        ax.axis("off")
        # partner-only label on top (simple, for a small subfigure)
        ax.set_title(p.split("_")[0], fontsize=14, fontweight="bold",
                     color=chex[p])
        cen = sites[p][1]
        axS.scatter([cen[0]], [cen[1]], s=340, facecolors="none",
                    edgecolors="#111", linewidths=2.3, zorder=200)
        fig.add_artist(ConnectionPatch(
            xyA=(cen[0], cen[1]), coordsA=axS.transData,
            xyB=(0, 0.5), coordsB=ax.transAxes, color="#333", lw=1.3))
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)


def main():
    from cloudvolume import CloudVolume
    from PIL import Image

    cfg, cfg_path = load_config()
    neurons = cfg["neurons"]
    nid = {n: i["id"] for n, i in neurons.items()}
    chex = {n: i["color_hex"] for n, i in neurons.items()}
    grp = {n: i["group"] for n, i in neurons.items()}

    rd = _find_results_dir()
    print(f"Config      : {os.path.basename(str(cfg_path))}")
    print(f"Results dir : {os.path.basename(rd)}")
    out_dir = os.path.join(rd, "gj_figures")
    sites_dir = os.path.join(out_dir, "sites")
    os.makedirs(sites_dir, exist_ok=True)

    pat = pd.read_csv(os.path.join(rd, "geometric_data", "contact_patches.csv"),
                      usecols=["neuron_a", "neuron_b", "patch_centroid_x",
                               "patch_centroid_y", "patch_centroid_z", "patch_area_um2"])
    syn_path = os.path.join(rd, "synapses.csv")
    syn = (pd.read_csv(syn_path) if os.path.exists(syn_path)
           else pd.DataFrame(columns=["pre_type", "post_type", "x", "y", "z"]))

    em = CloudVolume(EM_CLOUDPATH, mip=EM_MIP, use_https=True, progress=False)
    seg = CloudVolume(SEG_CLOUDPATH, mip=SEG_MIP, use_https=True, progress=False)

    all_sites = {}
    for motor, partners in PARTNER_PLAN.items():
        if motor not in neurons:
            print(f"  skip {motor} (not in config)")
            continue
        print(f"\n[{motor}] curating + rendering insets ...")
        sites, inset_imgs = {}, {}
        for p in partners:
            site = _densest_site(pat, syn, motor, p)
            if site is None:
                print(f"    {motor}<->{p}: no contact found, skipping")
                continue
            area, cen, sd = site
            sites[p] = (area, cen, sd)
            img = _inset_image(em, seg, nid, {n: tuple(neurons[n]["color_rgb"]) for n in neurons},
                               motor, p, cen)
            inset_imgs[p] = img
            Image.fromarray(img).save(os.path.join(sites_dir, f"SITE_{motor}__{p}.png"))
            all_sites[f"{motor}__{p}"] = {
                "partner": p, "center_nm": cen.astype(int).tolist(),
                "voxel_4x4x40": [int(round(cen[0] / 4)), int(round(cen[1] / 4)), int(round(cen[2] / 40))],
                "contact_area_um2": round(area, 2), "nearest_chem_synapse_um": round(sd / 1000, 2)}
            print(f"    {motor}<->{p}: {area:.1f} um2, {sd/1000:.1f} um from nearest synapse")

        # overlap blobs for this motor (relevant groups only)
        blob = pat[(pat.neuron_a == motor) | (pat.neuron_b == motor)].copy()
        blob["partner"] = [b if a == motor else a for a, b in zip(blob.neuron_a, blob.neuron_b)]
        allow = BLOB_GROUPS[motor[:3]]
        blob = blob[blob.partner.map(lambda n: n in grp and grp[n] in allow)]

        mesh_obj = os.path.join(rd, "neuron_meshes", f"{nid[motor]}.obj")
        if not os.path.exists(mesh_obj):
            print(f"    mesh not found for {motor}, skipping figure")
            continue
        out_path = os.path.join(out_dir, f"FIGURE_{motor}.png")
        _build_figure(motor, [p for p in partners if p in sites], sites,
                      inset_imgs, mesh_obj, chex, blob, out_path)
        print(f"    -> {out_path}")

    with open(os.path.join(out_dir, "gj_sites.json"), "w") as f:
        json.dump(all_sites, f, indent=2)
    print(f"\nDone. Figures + sites in {out_dir}")


if __name__ == "__main__":
    main()
