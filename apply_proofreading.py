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


def _read_manifest(rd):
    path = os.path.join(rd, "graveyard_manifest.csv")
    if not os.path.exists(path):
        return []
    return list(csv.DictReader(open(path, newline="")))


def _list_graveyard(rd):
    rows = _read_manifest(rd)
    if not rows:
        print("Graveyard is empty.")
        return
    grave = os.path.join(rd, "em_snaps_graveyard")
    present = {r["filename"] for r in rows
               if os.path.exists(os.path.join(grave, r["filename"]))}
    by_burial = {}
    for r in rows:
        if r["filename"] not in present:
            continue
        b = by_burial.setdefault(r["buried"], {"n": 0, "clusters": set(), "reasons": set()})
        b["n"] += 1
        b["clusters"].add(r["cluster_idx"])
        b["reasons"].add(r["reason"])
    print(f"Buried EM images: {len(present)}")
    print(f"  {'buried (timestamp)':<22}{'images':>8}  clusters")
    for stamp, b in sorted(by_burial.items()):
        cl = ",".join(sorted(b["clusters"])[:6]) + ("..." if len(b["clusters"]) > 6 else "")
        print(f"  {stamp:<22}{b['n']:>8}  {cl}")
    print("\nRevive with:  python apply_proofreading.py --restore all")
    print("          or: --restore <cluster_idx>   |   --restore <timestamp>")


def _restore(rd, what, dry_run=False):
    """Move images back out of the graveyard. Restores the EM images only; the
    metadata entries are recovered from the *.bak_<timestamp> backups."""
    rows = _read_manifest(rd)
    if not rows:
        sys.exit("Graveyard is empty — nothing to restore.")
    grave = os.path.join(rd, "em_snaps_graveyard")
    snaps = os.path.join(rd, "em_snaps")
    sel = [r for r in rows
           if what == "all" or r["cluster_idx"] == what or r["buried"] == what]
    sel = [r for r in sel if os.path.exists(os.path.join(grave, r["filename"]))]
    if not sel:
        sys.exit(f"Nothing buried matches '{what}'. Try --list-graveyard.")
    print(f"Restoring {len(sel)} image(s) matching '{what}'")
    if dry_run:
        print("[dry-run] nothing moved.")
        return
    n = 0
    for r in sel:
        src = os.path.join(grave, r["filename"])
        try:
            shutil.move(src, os.path.join(snaps, r["filename"]))
            n += 1
        except OSError:
            pass
    print(f"Restored {n} image(s) to em_snaps/.")
    stamps = sorted({r["buried"] for r in sel})
    print("\nTo also restore the metadata, copy the matching backup back:")
    for s in stamps:
        print(f"  copy overlap_em_meta.json.bak_{s}  ->  overlap_em_meta.json")
    print("Then rebuild the viewer: python skeleton_em_viewer.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotations", nargs="?", help="viewer_annotations.json (default: newest found)")
    ap.add_argument("--results-dir", help="results dir (default: latest)")
    ap.add_argument("--dry-run", action="store_true", help="report only; change nothing")
    ap.add_argument("--restore", metavar="WHAT",
                    help="revive from the graveyard instead of applying: 'all', "
                         "a cluster index, or a burial timestamp from "
                         "graveyard_manifest.csv")
    ap.add_argument("--list-graveyard", action="store_true",
                    help="show what is currently buried and exit")
    args = ap.parse_args()

    rd = _find_results_dir(args.results_dir)
    if args.list_graveyard:
        _list_graveyard(rd)
        return
    if args.restore:
        _restore(rd, args.restore, args.dry_run)
        return
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

    # Index em_snaps/ ONCE: {cluster_idx: {z_suffix: [filenames]}}. The folder
    # holds ~190k files, so globbing per deleted slice would rescan it hundreds
    # of times.
    snaps_dir = os.path.join(rd, "em_snaps")
    snap_index, snap_all = {}, {}
    if os.path.isdir(snaps_dir):
        for fn in os.listdir(snaps_dir):
            if not fn.startswith("overlap_"):
                continue
            parts = fn.split("_")
            try:
                cidx = int(parts[1])
            except (ValueError, IndexError):
                continue
            snap_all.setdefault(cidx, []).append(fn)
            tail = fn.rsplit("_", 1)[-1]           # e.g. 'z+012.png'
            if tail.startswith("z"):
                snap_index.setdefault(cidx, {}).setdefault(tail[:-4], []).append(fn)

    def _files_for_slice(cidx, z):
        sign = "+" if z >= 0 else "-"
        key = f"z{sign}{abs(z):03d}"
        return [os.path.join(snaps_dir, f)
                for f in snap_index.get(cidx, {}).get(key, [])]

    # area before proofreading, per pair (cluster areas are the exact sum of
    # their slice areas, so removing a slice subtracts exactly its own area)
    area_before = {}
    for m in meta:
        area_before[_pair_key(m["source"], m["target"])] = \
            area_before.get(_pair_key(m["source"], m["target"]), 0.0) + m["total_area_um2"]

    # build new meta: drop clusters, prune slices, recompute areas.
    # NOTE: the deleted regions are captured HERE, while slice_detail still
    # contains the rejected slices — collecting them afterwards would find
    # nothing, since the pruning below rewrites slice_detail in place.
    new_meta, pruned_slice_files, del_regions = [], [], []
    for m in meta:
        if m["idx"] in del_clusters:
            for sd in m["slice_detail"]:
                del_regions.append({"pair": _pair_key(m["source"], m["target"]),
                                    "x": sd["cx"], "y": sd["cy"],
                                    "z": m["z_base_nm"] + sd["z_offset"] * 40,
                                    "cluster": m["idx"]})
            continue
        if m["idx"] in del_slices:
            zoffs = del_slices[m["idx"]]
            for sd in m["slice_detail"]:
                if sd["z_offset"] in zoffs:
                    del_regions.append({"pair": _pair_key(m["source"], m["target"]),
                                        "x": sd["cx"], "y": sd["cy"],
                                        "z": m["z_base_nm"] + sd["z_offset"] * 40,
                                        "cluster": m["idx"]})
            m["slice_detail"] = [sd for sd in m["slice_detail"]
                                 if sd["z_offset"] not in zoffs]
            m["n_slices"] = len(m["slice_detail"])
            # recompute the cluster area from the slices that remain
            m["total_area_um2"] = float(sum(sd["area_um2"] for sd in m["slice_detail"]))
            m["total_faces"] = int(sum(sd["n_faces"] for sd in m["slice_detail"]))
            for z in zoffs:
                pruned_slice_files += _files_for_slice(m["idx"], z)
        new_meta.append(m)

    # area after proofreading, per pair
    area_after = {}
    for m in new_meta:
        area_after[_pair_key(m["source"], m["target"])] = \
            area_after.get(_pair_key(m["source"], m["target"]), 0.0) + m["total_area_um2"]

    changed = []
    for pair, before in sorted(area_before.items()):
        after = area_after.get(pair, 0.0)
        if abs(after - before) > 1e-9:
            changed.append((pair, before, after))
    if changed:
        print("\nOverlap area change (um2, from EM cluster metadata):")
        print(f"  {'pair':<22}{'before':>10}{'after':>10}{'delta':>10}")
        for pair, b, a in sorted(changed, key=lambda t: t[1] - t[2], reverse=True):
            print(f"  {pair:<22}{b:>10.3f}{a:>10.3f}{a-b:>+10.3f}")

    # em_snaps to delete: all files of removed clusters + pruned slices
    cluster_files = []
    for idx in del_clusters:
        cluster_files += [os.path.join(snaps_dir, f) for f in snap_all.get(idx, [])]
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

    # Graveyard: EM images are expensive to re-download (the host is capped at
    # ~1.3 crops/s), so proofreading MOVES them aside instead of erasing them.
    # Everything here can be revived with --restore.
    grave = os.path.join(rd, "em_snaps_graveyard")
    os.makedirs(grave, exist_ok=True)
    manifest = os.path.join(rd, "graveyard_manifest.csv")
    new_manifest = not os.path.exists(manifest)
    moved = 0
    with open(manifest, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_manifest:
            w.writerow(["filename", "cluster_idx", "reason", "buried"])
        for f in to_delete:
            base = os.path.basename(f)
            try:
                idx = int(base.split("_")[1])
            except (ValueError, IndexError):
                idx = ""
            reason = "cluster_deleted" if idx in del_clusters else "slice_deleted"
            dest = os.path.join(grave, base)
            try:
                if os.path.exists(dest):
                    os.remove(f)          # already buried once
                else:
                    shutil.move(f, dest)
                w.writerow([base, idx, reason, stamp])
                moved += 1
            except OSError:
                pass
    print(f"Moved {moved} EM image(s) to em_snaps_graveyard/ (revive: --restore)")

    # Record deleted regions so figure regeneration can honour the proofreading
    # (generate_gj_figures.py reads the raw contact_patches.csv, which is not
    # itself modified — this file tells it what to leave out).
    regions_path = os.path.join(rd, "deleted_regions.json")
    regions = []
    if os.path.exists(regions_path):
        try:
            regions = json.load(open(regions_path))
        except Exception:
            regions = []
    for r in del_regions:
        r["buried"] = stamp
    regions.extend(del_regions)
    json.dump(regions, open(regions_path, "w"), indent=2)
    print(f"Recorded {len(regions)} deleted region(s) -> deleted_regions.json")
    if os.path.exists(cand_path):
        with open(cand_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(kept_rows)

    # proofread area table (authoritative post-proofreading overlap areas).
    # NOTE: matrix_overlap_area_*.csv is the raw geometric analysis output and is
    # deliberately NOT rewritten here — this file records the proofread values.
    if changed:
        with open(os.path.join(rd, "proofread_areas.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pair", "area_um2_before", "area_um2_after", "delta_um2", "applied"])
            for pair, b, a in changed:
                w.writerow([pair, f"{b:.6f}", f"{a:.6f}", f"{a-b:.6f}", stamp])
        print(f"Wrote proofread_areas.csv ({len(changed)} pair(s) changed)")

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
