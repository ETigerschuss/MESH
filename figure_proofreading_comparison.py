#!/usr/bin/env python
"""
figure_proofreading_comparison.py — before/after EM proofreading
================================================================
Side-by-side skeleton figure for one motor neuron showing the geometric contact
sites BEFORE proofreading and the sites that survive AFTER a human rejected the
EM slices that show no real membrane apposition.

Left panel  : every contact patch the geometric detector produced.
Right panel : the same, minus the patches in ``deleted_regions.json``
              (written by apply_proofreading.py from the viewer's export).

Blob styling matches generate_gj_figures.py so the panels are comparable to the
publication figures.

Usage:
    python figure_proofreading_comparison.py [MOT_R] [--results-dir DIR]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh_config import load_config
from proofreading import rejected_patch_mask, pair_areas_um2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The neuron is drawn faintly so the true-scale contact areas stay legible:
# a contact is ~100 nm across on a ~130 um neuron, so it only reads against a
# low-contrast background. (Enlarging the contacts instead would misstate area.)
SKELETON_ALPHA = 0.22
BLOB_GROUPS = {"MOT": {"HS"}, "MOS": {"HS", "VS"}}


def _find_results_dir(explicit=None):
    if explicit:
        return explicit
    c = [d for d in os.listdir(SCRIPT_DIR)
         if os.path.isdir(os.path.join(SCRIPT_DIR, d))
         and d.startswith("comprehensive_overlap_results_")]
    if not c:
        sys.exit("ERROR: no results directory found.")
    return os.path.join(SCRIPT_DIR, sorted(c)[-1])


REJECT_COLOR = "#111111"


def _tris(df):
    """(N, 3, 2) projected triangles for the ('x', '-y') view."""
    return np.stack([
        df[["vertex1_x", "vertex1_y"]].values,
        df[["vertex2_x", "vertex2_y"]].values,
        df[["vertex3_x", "vertex3_y"]].values,
    ], axis=1)


def _panel(ax, mn, tm, faces, chex, motor, title, subtitle,
           rejected=None, dim_kept=False, xlim=None, ylim=None, stroke=0.0,
           outline=False):
    """Draw the neuron with its true contact areas painted on the membrane.

    The overlap triangles from contact_faces.csv are drawn at their real size
    (PolyCollection), not as scatter markers: a mean patch is ~100 nm across,
    which a scatter marker inflates roughly 6x in diameter on this scale.
    """
    import navis
    from matplotlib.collections import PolyCollection
    navis.plot2d(mn, color=chex[motor], alpha=SKELETON_ALPHA, view=("x", "-y"),
                 method="2d", ax=ax, linewidth=0.6)
    # Kept contacts stay at (near) full opacity even in the "rejected" panel.
    # Dimming them made the opaque black rejections look like a majority: the
    # rejected set is 20% of the true area and 28% of the projected pixels
    # (contacts at different z stack up in projection), so the two colours have
    # to compete on equal footing or the panel misreads.
    alpha = 0.80 if dim_kept else 0.95
    for prt, g in faces.groupby("partner"):
        ax.add_collection(PolyCollection(
            _tris(g), facecolors=chex[prt],
            edgecolors=(chex[prt] if stroke else "none"), linewidths=stroke,
            alpha=alpha, zorder=80))
    if rejected is not None and len(rejected):
        if outline:
            # Outline-only: marks WHERE contacts were rejected without covering
            # the kept contacts underneath, so the surviving area stays readable.
            ax.add_collection(PolyCollection(
                _tris(rejected), facecolors="none", edgecolors=REJECT_COLOR,
                linewidths=max(stroke, 0.4), alpha=0.9, zorder=90))
        else:
            ax.add_collection(PolyCollection(
                _tris(rejected), facecolors=REJECT_COLOR,
                edgecolors=(REJECT_COLOR if stroke else "none"), linewidths=stroke,
                alpha=0.95, zorder=90))
    V = tm.vertices
    ax.set_xlim(*(xlim if xlim else (V[:, 0].min(), V[:, 0].max())))
    ax.set_ylim(*(ylim if ylim else (V[:, 1].max(), V[:, 1].min())))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=20, fontweight="bold", color="#111", loc="left")
    ax.text(0.0, -0.02, subtitle, transform=ax.transAxes, fontsize=13,
            color="#444", va="top")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("motor", nargs="?", default="MOT_R")
    ap.add_argument("--results-dir")
    ap.add_argument("--out")
    ap.add_argument("--zoom", action="store_true",
                    help="crop to the contact-bearing region (contacts are ~100 nm "
                         "on a ~130 um neuron, so true-scale areas are invisible "
                         "at full-neuron zoom)")
    ap.add_argument("--outline", action="store_true",
                    help="draw rejected contacts as outlines instead of solid fill, "
                         "so they mark the rejected areas without hiding what was kept")
    ap.add_argument("--stroke", type=float, default=0.0,
                    help="outline width for overlap triangles; makes true-scale "
                         "contacts visible without enlarging their area")
    args = ap.parse_args()

    import navis
    import trimesh

    cfg, _ = load_config()
    neurons = cfg["neurons"]
    chex = {n: i["color_hex"] for n, i in neurons.items()}
    rd = _find_results_dir(args.results_dir)
    motor = args.motor
    if motor not in neurons:
        sys.exit(f"ERROR: {motor} not in config.")
    side = motor.split("_")[1]

    # Real overlap triangles (not patch centroids) so the painted areas are true
    # membrane footprints rather than inflated markers.
    faces = pd.read_csv(
        os.path.join(rd, "geometric_data", "contact_faces.csv"),
        usecols=["neuron_a", "neuron_b", "face_area_um2",
                 "centroid_x", "centroid_y", "centroid_z",
                 "vertex1_x", "vertex1_y", "vertex2_x", "vertex2_y",
                 "vertex3_x", "vertex3_y"])
    partners = [n for n, i in neurons.items()
                if i["group"] in BLOB_GROUPS[motor[:3]] and n.endswith("_" + side)]
    m = (((faces.neuron_a == motor) & (faces.neuron_b.isin(partners))) |
         ((faces.neuron_b == motor) & (faces.neuron_a.isin(partners))))
    blobs = faces[m].copy()
    blobs["partner"] = np.where(blobs.neuron_a == motor, blobs.neuron_b, blobs.neuron_a)

    # the proofreading mask works on patch_centroid_* names
    blobs = blobs.rename(columns={"centroid_x": "patch_centroid_x",
                                  "centroid_y": "patch_centroid_y",
                                  "centroid_z": "patch_centroid_z"})
    drop = rejected_patch_mask(blobs, rd)
    kept = blobs[~drop]

    # Areas come from the EM cluster metadata, which is exact: a cluster's area
    # is the sum of its slice areas, so removing a slice subtracts precisely its
    # own contribution. (Summing the drawn patches would only approximate this,
    # because patches and EM cluster faces are separate derivations of the same
    # contact.) The metadata is already post-proofreading, so "before" is
    # reconstructed by adding back what was rejected.
    areas_now = pair_areas_um2(rd)
    pairs = {" <-> ".join(sorted((motor, p))) for p in partners}
    a_after = sum(v for k, v in areas_now.items() if k in pairs)
    pr_csv = os.path.join(rd, "proofread_areas.csv")
    a_before = a_after
    if os.path.exists(pr_csv):
        import csv as _csv
        delta = 0.0
        for row in _csv.DictReader(open(pr_csv, newline="")):
            if row["pair"] in pairs:
                delta += float(row["area_um2_before"]) - float(row["area_um2_after"])
        a_before = a_after + delta
    print(f"{motor}: {len(blobs)} overlap triangles -> {len(kept)} kept "
          f"({int(drop.sum())} rejected); area {a_before:.1f} -> {a_after:.1f} um2")

    mesh_obj = os.path.join(rd, "neuron_meshes", f"{neurons[motor]['id']}.obj")
    tm = trimesh.load(mesh_obj, process=False)
    mn = navis.downsample_neuron(navis.MeshNeuron(tm, name=motor), downsampling_factor=12)

    rej = blobs[drop]
    n_rej = int(drop.sum())
    pct = 100 * (1 - a_after / a_before) if a_before else 0.0

    xlim = ylim = None
    if args.zoom:
        pad = 3000.0
        xlim = (blobs.patch_centroid_x.min() - pad, blobs.patch_centroid_x.max() + pad)
        ylim = (blobs.patch_centroid_y.max() + pad, blobs.patch_centroid_y.min() - pad)

    fig, axes = plt.subplots(1, 3, figsize=(30, 11), facecolor="white")
    _panel(axes[0], mn, tm, blobs, chex, motor, f"{motor} — before",
           f"{len(blobs)} overlap faces   ·   {a_before:.1f} µm²",
           xlim=xlim, ylim=ylim, stroke=args.stroke)
    _panel(axes[1], mn, tm, kept, chex, motor, f"{motor} — rejected (black)",
           f"{n_rej} faces removed   ·   −{a_before - a_after:.1f} µm² (−{pct:.0f}%)",
           rejected=rej, dim_kept=True, xlim=xlim, ylim=ylim, stroke=args.stroke,
           outline=args.outline)
    _panel(axes[2], mn, tm, kept, chex, motor, f"{motor} — after",
           f"{len(kept)} overlap faces   ·   {a_after:.1f} µm²",
           xlim=xlim, ylim=ylim, stroke=args.stroke)
    fig.suptitle(f"{motor} contact sites with HS cells, before and after EM proofreading",
                 fontsize=16, color="#222", y=0.965)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = args.out or os.path.join(rd, "gj_figures", f"FIGURE_{motor}_proofreading_comparison")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out + ".pdf", facecolor="white")
    fig.savefig(out + ".png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()
