#!/usr/bin/env python
"""
Extended putative gap-junction composite figures (publication panels)
======================================================================
For a user-specified GJ coordinate, builds an extended composite figure with 9
EM example insets arranged left/top/bottom to minimize arrow crossings:

  * center  : skeleton of the nearest motor neuron with all relevant overlap
              blobs (HS for MOT; HS + VS1-4 for MOS)
  * left    : 3 insets (top-to-bottom)
  * top     : 3 insets (left-to-right)
  * bottom  : 3 insets (left-to-right)

Each inset shows the densest contact site, colored with the partner, and linked
via a connector line + ringed location on the skeleton. No arrows on insets to
keep them simple. Site must be >= MIN_SYN_DIST_NM from any chemical synapse.

EM is the FlyWire-aligned volume at 8 nm (the only EM that registers with the
v14.1 segmentation); for native-4 nm ultrastructure use the coordinates in
FlyWire's Neuroglancer.

Inputs  : GJ coordinate (voxel 4x4x40), latest comprehensive_overlap_results_*,
          active neuron config.
Outputs : <results_dir>/gj_figures_extended/FIGURE_<motor>_extended.png
          <results_dir>/gj_figures_extended/sites/SITE_*.png
          <results_dir>/gj_figures_extended/gj_sites_extended.json

Usage:  python generate_gj_figures_extended.py <voxel_x> <voxel_y> <voxel_z>
        e.g. python generate_gj_figures_extended.py 154698 66954 5068
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

MIN_SYN_DIST_NM = 2000
DENSITY_BIN_NM = 1500
INSET_HALF_PX = 150
SEARCH_RADIUS_NM = 15000

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


def _densest_near(pat, syn, a, b, center_nm, search_nm=SEARCH_RADIUS_NM,
                  binnm=DENSITY_BIN_NM, avoid=MIN_SYN_DIST_NM):
    """Densest contact site between a and b near center_nm, synapse-free."""
    m = (((pat.neuron_a == a) & (pat.neuron_b == b)) |
         ((pat.neuron_a == b) & (pat.neuron_b == a)))
    sub = pat[m]
    if len(sub) == 0:
        return None
    C = sub[["patch_centroid_x", "patch_centroid_y", "patch_centroid_z"]].values
    d = np.linalg.norm(C - center_nm, axis=1)
    near = d <= search_nm
    if not near.any():
        return None
    sub = sub[near]
    bins = list(zip((sub.patch_centroid_x // binnm).astype(int),
                    (sub.patch_centroid_y // binnm).astype(int),
                    (sub.patch_centroid_z // binnm).astype(int)))
    g = (sub.assign(_b=bins)
            .groupby("_b")
            .agg(area=("patch_area_um2", "sum"),
                 cx=("patch_centroid_x", "mean"),
                 cy=("patch_centroid_y", "mean"),
                 cz=("patch_centroid_z", "mean"))
            .reset_index().sort_values("area", ascending=False))
    for _, r in g.iterrows():
        cen = np.array([r.cx, r.cy, r.cz])
        if _syn_dist(syn, a, b, cen) >= avoid:
            return float(r.area), cen, _syn_dist(syn, a, b, cen), float(np.linalg.norm(cen - center_nm))
    if len(g):
        r = g.iloc[0]
        cen = np.array([r.cx, r.cy, r.cz])
        return float(r.area), cen, _syn_dist(syn, a, b, cen), float(np.linalg.norm(cen - center_nm))
    return None


def _inset_image(em, seg, nid, ncol, a, b, cen):
    """Aligned 8 nm EM crop with faint colour overlay."""
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


def _build_figure(motor, sites_9, inset_imgs, mesh_obj, chex, blob_df, out_path, nid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    import navis
    import trimesh

    tm = trimesh.load(mesh_obj, process=False)
    mn = navis.downsample_neuron(navis.MeshNeuron(tm, name=motor), downsampling_factor=12)

    fig = plt.figure(figsize=(20, 18), facecolor="white")
    # center skeleton (larger)
    axS = fig.add_axes([0.25, 0.25, 0.50, 0.50])
    navis.plot2d(mn, color=chex[motor], alpha=0.28, view=("x", "-y"),
                 method="2d", ax=axS, linewidth=0.5)

    # overlap blobs
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
    axS.set_title(motor, fontsize=24, color=chex[motor], fontweight="bold", loc="center")

    # Inset positions: left (3), top (3), bottom (3)
    inset_size = 0.22
    positions = {
        # left: x=0.02, y varies from top to bottom
        "left": [(0.02, 0.70, "left"), (0.02, 0.48, "left"), (0.02, 0.26, "left")],
        # top: y=0.78, x varies left to right
        "top": [(0.26, 0.78, "top"), (0.48, 0.78, "top"), (0.70, 0.78, "top")],
        # bottom: y=0.02, x varies left to right
        "bottom": [(0.26, 0.02, "bottom"), (0.48, 0.02, "bottom"), (0.70, 0.02, "bottom")],
    }

    all_pos = []
    for loc_list in positions.values():
        for x, y, loc in loc_list:
            all_pos.append((x, y, loc))

    # assign the 9 sites to positions
    for k, site_tuple in enumerate(sites_9[:9]):
        if k >= len(all_pos):
            break
        d_from_gj, a, b, area, cen, sd = site_tuple
        x, y, loc = all_pos[k]
        ax = fig.add_axes([x, y, inset_size, inset_size])
        img = inset_imgs[(a, b)]
        ax.imshow(img)
        ax.axis("off")
        # Partner label only (e.g. "HSN" or "VS2")
        label = a.split("_")[0] if a != motor else b.split("_")[0]
        ax.set_title(label, fontsize=13, fontweight="bold", color=chex[a if a != motor else b])

        # ring on skeleton
        axS.scatter([cen[0]], [cen[1]], s=340, facecolors="none",
                    edgecolors="#111", linewidths=2.3, zorder=200)
        # connector line
        if loc == "left":
            xyB = (1, 0.5)
        elif loc == "top":
            xyB = (0.5, 0)
        else:  # bottom
            xyB = (0.5, 1)
        fig.add_artist(ConnectionPatch(
            xyA=(cen[0], cen[1]), coordsA=axS.transData,
            xyB=xyB, coordsB=ax.transAxes, color="#333", lw=1.5))

    fig.savefig(out_path, dpi=120, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    from cloudvolume import CloudVolume
    from PIL import Image

    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    try:
        voxel = np.array([int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])])
        center_nm = voxel * np.array([4, 4, 40])
    except ValueError:
        print(f"ERROR: invalid voxel coordinates: {sys.argv[1:]}")
        sys.exit(1)

    cfg, cfg_path = load_config()
    neurons = cfg["neurons"]
    nid = {n: i["id"] for n, i in neurons.items()}
    chex = {n: i["color_hex"] for n, i in neurons.items()}
    ncol_rgb = {n: tuple(i["color_rgb"]) for n, i in neurons.items()}
    grp = {n: i["group"] for n, i in neurons.items()}

    rd = _find_results_dir()
    print(f"Config      : {os.path.basename(str(cfg_path))}")
    print(f"Results dir : {os.path.basename(rd)}")
    print(f"GJ coord (voxel): {voxel.tolist()}")
    print(f"GJ coord (nm):   {center_nm.tolist()}")

    out_dir = os.path.join(rd, "gj_figures_extended")
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

    # Find 9 nearest synapse-free sites to the GJ coordinate
    all_neurons = sorted(set(pat.neuron_a) | set(pat.neuron_b))
    cands = []
    for a in all_neurons:
        for b in all_neurons:
            if a >= b:
                continue
            site = _densest_near(pat, syn, a, b, center_nm)
            if site:
                area, cen, sd, d = site
                cands.append((d, a, b, area, cen, sd))
    cands.sort(key=lambda x: x[0])
    sites_9 = cands[:9]

    if len(sites_9) < 9:
        print(f"WARNING: only {len(sites_9)} sites found within {SEARCH_RADIUS_NM} nm")

    # Determine which motor to use (the nearest site's motor)
    if sites_9:
        _, a, b, _, _, _ = sites_9[0]
        # figure out which is the motor
        motor = a if a.startswith("MOT") or a.startswith("MOS") else b
        if not (motor.startswith("MOT") or motor.startswith("MOS")):
            motor = a if "L" in a or "R" in a else "MOT_R"
    else:
        motor = "MOT_R"

    print(f"\nMotor neuron: {motor}")
    print(f"9 nearest synapse-free contact sites:")

    all_sites = {}
    inset_imgs = {}
    for idx, site_tuple in enumerate(sites_9):
        d, a, b, area, cen, sd = site_tuple
        print(f"  {idx+1}. {a:7s}<->{b:7s}  {d/1000:5.2f}um away  {area:4.1f}um2  syn {sd/1000:5.1f}um")
        img = _inset_image(em, seg, nid, ncol_rgb, a, b, cen)
        inset_imgs[(a, b)] = img
        Image.fromarray(img).save(os.path.join(sites_dir, f"SITE_{a}__{b}.png"))
        all_sites[f"{a}__{b}"] = {
            "partner_a": a, "partner_b": b, "center_nm": cen.astype(int).tolist(),
            "voxel_4x4x40": [int(round(cen[0] / 4)), int(round(cen[1] / 4)), int(round(cen[2] / 40))],
            "contact_area_um2": round(area, 2), "nearest_chem_synapse_um": round(sd / 1000, 2),
            "distance_to_gj_um": round(d / 1000, 2)}

    # Build the extended figure
    if motor in neurons:
        mesh_obj = os.path.join(rd, "neuron_meshes", f"{nid[motor]}.obj")
        if os.path.exists(mesh_obj):
            # Get overlap blobs for this motor
            blob = pat[(pat.neuron_a == motor) | (pat.neuron_b == motor)].copy()
            blob["partner"] = [b if a == motor else a for a, b in zip(blob.neuron_a, blob.neuron_b)]
            allow = {"HS"} if motor.startswith("MOT") else {"HS", "VS"}
            blob = blob[blob.partner.map(lambda n: n in grp and grp[n] in allow)]

            out_path = os.path.join(out_dir, f"FIGURE_{motor}_extended.png")
            _build_figure(motor, sites_9, inset_imgs, mesh_obj, chex, blob, out_path, nid)
            print(f"\n-> {out_path}")

    with open(os.path.join(out_dir, "gj_sites_extended.json"), "w") as f:
        json.dump(all_sites, f, indent=2)
    print(f"Done. Extended figures + sites in {out_dir}")


if __name__ == "__main__":
    main()
