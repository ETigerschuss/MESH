#!/usr/bin/env python
"""
apply_proofreading.py — make viewer proofreading stick
======================================================
The EM viewer lets you delete spurious overlaps while proofreading (Delete
Slice / Delete All) and mark confirmed gap junctions, then exports them to
``viewer_annotations.json``. That file, on its own, changes nothing: rebuilding
the viewer brings the deleted areas back. This script APPLIES those decisions to
the dataset so the removal is permanent and every downstream product (viewer,
gap-junction candidate list, figures) reflects it.

For each exported decision it:
  * overlap_all   — drops the whole cluster from ``overlap_em_meta.json``,
                    deletes its ``em_snaps/overlap_<idx>_*`` images, and removes
                    any ``gj_candidates.csv`` sites inside it.
  * overlap_slice — removes that single Z-slice from the cluster's slice list
                    (and deletes the matching image), so the viewer skips it.
  * contact       — logged (contacts are not gap-junction candidates).
  * gapJunction   — appended to ``confirmed_gj.csv``.

Originals are backed up (``*.bak_<timestamp>``) before anything is changed.
After running, rebuild the viewer (``python skeleton_em_viewer.py``) and, if you
use them, regenerate the candidate list / figures.

Usage:
    python apply_proofreading.py [viewer_annotations.json] [--results-dir DIR] [--dry-run]
If no path is given, the newest viewer_annotations.json in the results dir or in
~/Downloads is used.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_PURGE_NM = 2000.0   # gj_candidate removed if within this of a deleted cluster (same pair)


def _find_results_dir(explicit=None):
    if explicit:
        return explicit
    cands = [d for d in os.listdir(SCRIPT_DIR)
             if os.path.isdir(os.path.join(SCRIPT_DIR, d))
             and d.startswith("comprehensive_overlap_results_")]
    if not cands:
        sys.exit("ERROR: no comprehensive_overlap_results_* directory found.")
    return os.path.join(SCRIPT_DIR, sorted(cands)[-1])


def _find_annotations(explicit, results_dir):
    if explicit:
        if not os.path.isfile(explicit):
            sys.exit(f"ERROR: annotations file not found: {explicit}")
        return explicit
    search = []
    search += glob.glob(os.path.join(results_dir, "viewer_annotations*.json"))
    downloads = os.path.join(os.path.expanduser("~"), "Downloads")
    search += glob.glob(os.path.join(downloads, "viewer_annotations*.json"))
    if not search:
        sys.exit("ERROR: no viewer_annotations.json found (pass its path explicitly).")
    newest = max(search, key=os.path.getmtime)
    print(f"Using annotations: {newest}")
    return newest


def _backup(path, stamp):
    if os.path.exists(path):
        shutil.copy2(path, f"{path}.bak_{stamp}")


def _pair_key(a, b):
    return " <-> ".join(sorted((str(a), str(b))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations", nargs="?", help="viewer_annotations.json (default: newest found)")
    ap.add_argument("--results-dir", help="results dir (default: latest)")
    ap.add_argument("--dry-run", action="store_true", help="report only; change nothing")
    args = ap.parse_args()

    rd = _find_results_dir(args.results_dir)
    ann_path = _find_annotations(args.annotations, rd)
    ann = json.load(open(ann_path, encoding="utf-8"))
    deleted = ann.get("deletedItems", [])
    gjs = ann.get("gapJunctions", [])
    print(f"Results dir : {os.path.basename(rd)}")
    print(f"Decisions   : {len(deleted)} deletions, {len(gjs)} confirmed gap junctions")

    del_clusters = {int(d["idx"]) for d in deleted if d.get("kind") == "overlap_all"}
    del_slices = {}
    for d in deleted:
        if d.get("kind") == "overlap_slice":
            del_slices.setdefault(int(d["idx"]), set()).add(int(d["z_offset"]))
    n_contacts = sum(1 for d in deleted if d.get("kind") == "contact")
    print(f"  overlap clusters removed : {len(del_clusters)}")
    print(f"  overlap slices removed   : {sum(len(v) for v in del_slices.values())} "
          f"in {len(del_slices)} clusters")
    print(f"  contacts flagged (logged): {n_contacts}")

    meta_path = os.path.join(rd, "overlap_em_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8"))

    # collect deleted-cluster points (nm) per pair, for candidate purging
    del_points = {}
    for m in meta:
        if m["idx"] in del_clusters:
            key = _pair_key(m["source"], m["target"])
            for sd in m["slice_detail"]:
                z = m["z_base_nm"] + sd["z_offset"] * 40
                del_points.setdefault(key, []).append((sd["cx"], sd["cy"], z))
    del_points = {k: np.array(v, float) for k, v in del_points.items()}

    # build new meta: drop clusters, prune slices
    new_meta, pruned_slice_files = [], []
    for m in meta:
        if m["idx"] in del_clusters:
            continue
        if m["idx"] in del_slices:
            zoffs = del_slices[m["idx"]]
            m["slice_detail"] = [sd for sd in m["slice_detail"]
                                 if sd["z_offset"] not in zoffs]
            m["n_slices"] = len(m["slice_detail"])
            for z in zoffs:
                sign = "+" if z >= 0 else "-"
                pruned_slice_files += glob.glob(os.path.join(
                    rd, "em_snaps", f"overlap_{m['idx']}_*_z{sign}{abs(z):03d}.png"))
        new_meta.append(m)

    # em_snaps to delete: all files of removed clusters + pruned slices
    cluster_files = []
    for idx in del_clusters:
        cluster_files += glob.glob(os.path.join(rd, "em_snaps", f"overlap_{idx}_*"))
    to_delete = sorted(set(cluster_files) | set(pruned_slice_files))

    # gj_candidates purge
    cand_path = os.path.join(rd, "gj_candidates.csv")
    kept_rows, removed_cand = [], 0
    if os.path.exists(cand_path):
        rows = list(csv.DictReader(open(cand_path, newline="")))
        for r in rows:
            key = _pair_key(r["neuron_a"], r["neuron_b"])
            p = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
            pts = del_points.get(key)
            if pts is not None and np.sqrt(((pts - p) ** 2).sum(1)).min() <= CANDIDATE_PURGE_NM:
                removed_cand += 1
            else:
                kept_rows.append(r)

    print(f"\nWould remove: {len(to_delete)} EM images, "
          f"{len(meta) - len(new_meta)} clusters from meta, "
          f"{removed_cand} gj_candidates.")
    if args.dry_run:
        print("[dry-run] nothing changed.")
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    _backup(meta_path, stamp)
    _backup(cand_path, stamp)

    json.dump(new_meta, open(meta_path, "w"), indent=2)
    for f in to_delete:
        try:
            os.remove(f)
        except OSError:
            pass
    if os.path.exists(cand_path):
        with open(cand_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(kept_rows)

    # confirmed gap junctions
    if gjs:
        conf_path = os.path.join(rd, "confirmed_gj.csv")
        exists = os.path.exists(conf_path)
        with open(conf_path, "a", newline="") as fh:
            w = csv.writer(fh)
            if not exists:
                w.writerow(["source", "target", "x_nm", "y_nm", "z_nm",
                            "vx", "vy", "vz", "kind", "idx", "timestamp"])
            for g in gjs:
                x, y, z = g["x"], g["y"], g["z"]
                w.writerow([g.get("source", ""), g.get("target", ""), x, y, z,
                            int(round(x / 4)), int(round(y / 4)), int(round(z / 40)),
                            g.get("kind", ""), g.get("idx", ""), g.get("timestamp", "")])
        print(f"Appended {len(gjs)} confirmed gap junction(s) -> confirmed_gj.csv")

    # log
    with open(os.path.join(rd, "proofreading_applied.log"), "a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {os.path.basename(ann_path)}: "
                 f"-{len(del_clusters)} clusters, -{len(to_delete)} images, "
                 f"-{removed_cand} candidates, +{len(gjs)} confirmed GJ\n")

    print(f"\nApplied. Backups: *.bak_{stamp}")
    print("Next: rebuild the viewer  ->  python skeleton_em_viewer.py")
    print("      (and re-run phase_a_gj_candidates.py if you want the summary refreshed)")


if __name__ == "__main__":
    main()
