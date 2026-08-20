#!/usr/bin/env python
"""
Confine overlap areas by segmentation adjacency
===============================================
The geometric overlap detector flags mesh proximity (<=0.1 um), which
over-counts: two meshes can be near without their membranes actually touching.
This script validates each motor<->partner contact against the FlyWire
segmentation: for a sample of contact patches it checks whether the two cells'
segments are truly adjacent (touch within DILATE_PX). It reports, per pair, the
truly-apposed area fraction and the confined area, and writes the validated
(real-contact) patch centroids so downstream code can surface only the EM images
that show a real contact.

Inputs : latest comprehensive_overlap_results_* (contact_patches.csv) + config.
Outputs: <results_dir>/overlap_validation.json
         <results_dir>/validated_patches.csv
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

N_SAMPLE = 80            # patches sampled per pair to estimate the real fraction
DILATE_PX = 3           # segment adjacency tolerance (seg px; 16 nm -> ~48 nm)
SEG_HALF_PX = 16        # half-size of the seg cutout per patch (px)
SEG_CLOUDPATH = "precomputed://gs://flywire_v141_m783"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_results_dir():
    cands = [d for d in os.listdir(SCRIPT_DIR)
             if os.path.isdir(os.path.join(SCRIPT_DIR, d))
             and d.startswith("comprehensive_overlap_results_")]
    if not cands:
        sys.exit("ERROR: no comprehensive_overlap_results_* directory found.")
    return os.path.join(SCRIPT_DIR, sorted(cands)[-1])


def main():
    from cloudvolume import CloudVolume
    from scipy.ndimage import binary_dilation

    cfg, _ = load_config()
    neurons = cfg["neurons"]
    nid = {n: i["id"] for n, i in neurons.items()}
    grp = {n: i["group"] for n, i in neurons.items()}
    rd = _find_results_dir()
    print(f"Results: {os.path.basename(rd)}")

    pat = pd.read_csv(os.path.join(rd, "geometric_data", "contact_patches.csv"),
                      usecols=["neuron_a", "neuron_b", "patch_centroid_x",
                               "patch_centroid_y", "patch_centroid_z", "patch_area_um2"])
    seg = CloudVolume(SEG_CLOUDPATH, mip=0, use_https=True, progress=False, fill_missing=True)
    sr = np.array(seg.resolution)
    H = SEG_HALF_PX

    # motor <-> partner pairs (HS for MOT; HS + VS1-4 for MOS)
    BLOB = {"MOT": {"HS"}, "MOS": {"HS", "VS"}}
    pairs = []
    for motor in ["MOT_L", "MOT_R", "MOS_L", "MOS_R"]:
        if motor not in neurons:
            continue
        side = motor.split("_")[1]
        for p, info in neurons.items():
            if info["group"] in BLOB[motor[:3]] and p.endswith("_" + side):
                pairs.append((motor, p))

    results = {}
    real_rows = []
    for a, b in pairs:
        m = (((pat.neuron_a == a) & (pat.neuron_b == b)) |
             ((pat.neuron_a == b) & (pat.neuron_b == a)))
        sub = pat[m].reset_index(drop=True)
        mesh_area = float(sub.patch_area_um2.sum())
        if len(sub) == 0:
            continue
        ida, idb = nid[a], nid[b]
        idx = np.unique(np.linspace(0, len(sub) - 1, min(N_SAMPLE, len(sub))).astype(int))
        touch = notouch = absent = 0
        real_area = samp_area = 0.0
        for i in idx:
            r = sub.iloc[i]
            c = (np.array([r.patch_centroid_x, r.patch_centroid_y, r.patch_centroid_z]) / sr).astype(int)
            try:
                sg = np.asarray(seg[c[0]-H:c[0]+H, c[1]-H:c[1]+H, c[2]])[:, :, 0, 0]
            except Exception:
                continue
            ma, mb = (sg == ida), (sg == idb)
            samp_area += r.patch_area_um2
            if ma.sum() == 0 or mb.sum() == 0:
                absent += 1
                continue
            if (binary_dilation(ma, iterations=DILATE_PX) & mb).sum() > 0:
                touch += 1
                real_area += r.patch_area_um2
                real_rows.append({"neuron_a": a, "neuron_b": b,
                                  "x": float(r.patch_centroid_x), "y": float(r.patch_centroid_y),
                                  "z": float(r.patch_centroid_z), "area_um2": float(r.patch_area_um2)})
            else:
                notouch += 1
        frac = (real_area / samp_area) if samp_area > 0 else 0.0
        confined = mesh_area * frac
        results[f"{a}__{b}"] = {"mesh_area_um2": round(mesh_area, 2),
                                "real_fraction": round(frac, 3),
                                "confined_area_um2": round(confined, 2),
                                "n_patches": int(len(sub)), "n_sampled": int(len(idx)),
                                "touch": touch, "not_touch": notouch, "cell_absent": absent}
        print(f"  {a}<->{b}: mesh {mesh_area:6.1f}  real {frac*100:3.0f}%  "
              f"-> confined {confined:6.1f} um2  (touch {touch}/{len(idx)})")

    json.dump(results, open(os.path.join(rd, "overlap_validation.json"), "w"), indent=2)
    pd.DataFrame(real_rows).to_csv(os.path.join(rd, "validated_patches.csv"), index=False)
    tot_mesh = sum(v["mesh_area_um2"] for v in results.values())
    tot_conf = sum(v["confined_area_um2"] for v in results.values())
    print(f"\nTOTAL motor<->partner area: mesh {tot_mesh:.0f} -> confined {tot_conf:.0f} um2 "
          f"({tot_conf/tot_mesh*100:.0f}%)")
    print(f"Wrote overlap_validation.json + validated_patches.csv to {os.path.basename(rd)}")


if __name__ == "__main__":
    main()
