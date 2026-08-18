#!/usr/bin/env python
"""
proofreading.py — shared helpers for applying EM proofreading decisions
=======================================================================
When a contact is rejected in the EM viewer, the decision is recorded per
(cluster, Z-slice) in ``deleted_regions.json``. Figures, however, are drawn from
the raw geometric ``contact_patches.csv``, which is never modified. This module
maps one onto the other so every downstream product can honour the proofreading
from a single implementation.

Matching rule
-------------
A patch is rejected when it is:
  * of the same neuron pair as a rejected slice,
  * on the same Z section (+/- Z_TOL_NM, sections are 40 nm), and
  * inside the XY footprint of the cluster that slice belonged to
    (bounding box of all its slice centroids, plus BBOX_MARGIN_NM).

Using the cluster's own footprint rather than a fixed window matters: contact
patches on a section can lie far from that section's centroid, so a fixed box
silently under-removes (18.6 of the expected 28.6 um2 on the MOT_R test case,
versus 29.2 for this rule).

The EM cluster metadata (``overlap_em_meta.json``) remains the authoritative
source for *areas* — it is exact, since a cluster's area is the sum of its slice
areas. This module decides only which patches to *draw*.
"""

import json
import os

import numpy as np

Z_TOL_NM = 20.0          # half a 40 nm section
BBOX_MARGIN_NM = 1000.0  # tolerance around a cluster's XY footprint


def load_deleted_regions(results_dir):
    """[{pair, x, y, z, cluster}, ...] — empty if nothing was proofread."""
    path = os.path.join(results_dir, "deleted_regions.json")
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path))
    except Exception:
        return []


def _cluster_bboxes(results_dir, regions):
    """{cluster_idx: [xmin, xmax, ymin, ymax]} over surviving AND deleted slices."""
    bbox = {}

    def _add(idx, x, y):
        b = bbox.setdefault(idx, [np.inf, -np.inf, np.inf, -np.inf])
        b[0] = min(b[0], x); b[1] = max(b[1], x)
        b[2] = min(b[2], y); b[3] = max(b[3], y)

    meta_path = os.path.join(results_dir, "overlap_em_meta.json")
    if os.path.exists(meta_path):
        try:
            for m in json.load(open(meta_path)):
                for sd in m["slice_detail"]:
                    _add(m["idx"], sd["cx"], sd["cy"])
        except Exception:
            pass
    for r in regions:
        _add(r.get("cluster"), r["x"], r["y"])
    return bbox


def rejected_patch_mask(pat, results_dir, regions=None):
    """Boolean mask over ``pat`` marking patches rejected during proofreading.

    ``pat`` needs columns neuron_a, neuron_b, patch_centroid_x/y/z.
    """
    regions = load_deleted_regions(results_dir) if regions is None else regions
    drop = np.zeros(len(pat), dtype=bool)
    if not regions or len(pat) == 0:
        return drop
    bbox = _cluster_bboxes(results_dir, regions)
    keys = pat.apply(
        lambda r: " <-> ".join(sorted((r.neuron_a, r.neuron_b))), axis=1).values
    xyz = pat[["patch_centroid_x", "patch_centroid_y", "patch_centroid_z"]].values
    for pair in {r["pair"] for r in regions}:
        m = (keys == pair)
        if not m.any():
            continue
        sub = xyz[m]
        hit = np.zeros(len(sub), dtype=bool)
        for r in regions:
            if r["pair"] != pair:
                continue
            b = bbox.get(r.get("cluster"))
            c = np.abs(sub[:, 2] - r["z"]) <= Z_TOL_NM
            if b is not None:
                c &= ((sub[:, 0] >= b[0] - BBOX_MARGIN_NM) &
                      (sub[:, 0] <= b[1] + BBOX_MARGIN_NM) &
                      (sub[:, 1] >= b[2] - BBOX_MARGIN_NM) &
                      (sub[:, 1] <= b[3] + BBOX_MARGIN_NM))
            hit |= c
        drop[np.where(m)[0][hit]] = True
    return drop


def drop_rejected(pat, results_dir, verbose=True):
    """Return ``pat`` without the patches rejected during proofreading."""
    regions = load_deleted_regions(results_dir)
    if not regions:
        return pat
    drop = rejected_patch_mask(pat, results_dir, regions)
    if verbose and drop.any():
        print(f"  [proofreading] dropping {int(drop.sum())} of {len(pat)} contact "
              f"patches rejected in the EM viewer ({len(regions)} rejected slices)")
    return pat.loc[~drop].reset_index(drop=True)


def pair_areas_um2(results_dir):
    """{pair: area_um2} from the EM cluster metadata — exact, post-proofreading."""
    out = {}
    meta_path = os.path.join(results_dir, "overlap_em_meta.json")
    if not os.path.exists(meta_path):
        return out
    try:
        for m in json.load(open(meta_path)):
            k = " <-> ".join(sorted((str(m["source"]), str(m["target"]))))
            out[k] = out.get(k, 0.0) + float(m.get("total_area_um2", 0.0))
    except Exception:
        pass
    return out
