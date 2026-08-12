#!/usr/bin/env python
"""
Phase A — chemical-vs-candidate filter for putative gap junctions
=================================================================
Turns the seg-validated motor<->LPTC appositions into a ranked shortlist of
gap-junction *candidates* by removing appositions that coincide with an
annotated chemical synapse.

Rationale
---------
FlyWire annotates chemical synapses (T-bars) with coordinates, but NOT gap
junctions. So the exact first step needs no learning: for each validated
apposition, ask whether a chemical synapse between the SAME two cells lies within
a small distance of it. If yes -> chemical contact (exclude). If no -> a
membrane apposition with no chemical synapse = gap-junction candidate.

This is the deterministic ground-truth filter that precedes any EM-image model
(Phase B): it converts 937 raw appositions into a small, ranked candidate list.

Inputs  (latest comprehensive_overlap_results_*):
    validated_patches.csv   seg-validated motor<->LPTC appositions (nm)
    synapses.csv            FlyWire chemical synapses (nm) among these cells
    configs/*.json          neuron id<->name
Outputs:
    gj_candidates.csv       deduplicated candidate sites, ranked by area
    phase_a_summary.csv     per-pair chemical vs candidate counts
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from mesh_config import load_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# A validated apposition is called "chemical" if an annotated chemical synapse
# between the same two cells lies within this distance of its centroid (nm).
CHEM_MATCH_NM = 1000.0
# Distinct candidate sites of one pair must be at least this far apart (nm).
SITE_SEP_NM = 3000.0
# Confirmed reference gap junction (MOT_R<->HSN_R), FlyWire voxel * (4,4,40).
CONFIRMED_GJ_NM = np.array([154698 * 4, 66954 * 4, 5068 * 40], dtype=float)


def _find_results_dir():
    cands = [d for d in os.listdir(SCRIPT_DIR)
             if os.path.isdir(os.path.join(SCRIPT_DIR, d))
             and d.startswith("comprehensive_overlap_results_")]
    if not cands:
        sys.exit("ERROR: no comprehensive_overlap_results_* directory found.")
    return os.path.join(SCRIPT_DIR, sorted(cands)[-1])


def main():
    cfg, _ = load_config()
    id2name = {str(info["id"]): name for name, info in cfg["neurons"].items()}
    rd = _find_results_dir()
    print(f"Results: {os.path.basename(rd)}")

    app = pd.read_csv(os.path.join(rd, "validated_patches.csv"))
    syn = pd.read_csv(os.path.join(rd, "synapses.csv"))
    syn["pre_name"] = syn["pre"].astype(str).map(id2name)
    syn["post_name"] = syn["post"].astype(str).map(id2name)
    syn = syn.dropna(subset=["pre_name", "post_name"])
    print(f"Appositions: {len(app)}   annotated chemical synapses: {len(syn)}")

    # index synapses by unordered cell pair -> Nx3 nm coordinates
    syn_xyz = {}
    for _, r in syn.iterrows():
        key = frozenset((r.pre_name, r.post_name))
        syn_xyz.setdefault(key, []).append((r.x, r.y, r.z))
    syn_xyz = {k: np.array(v, float) for k, v in syn_xyz.items()}

    def nearest_chem_nm(a, b, xyz):
        pts = syn_xyz.get(frozenset((a, b)))
        if pts is None:
            return np.inf
        return float(np.sqrt(((pts - xyz) ** 2).sum(1)).min())

    # classify every apposition
    d = []
    for _, r in app.iterrows():
        xyz = np.array([r.x, r.y, r.z], float)
        d.append(nearest_chem_nm(r.neuron_a, r.neuron_b, xyz))
    app["nearest_chem_nm"] = d
    app["is_chemical"] = app["nearest_chem_nm"] <= CHEM_MATCH_NM
    app["pair"] = app.apply(
        lambda r: " <-> ".join(sorted((r.neuron_a, r.neuron_b))), axis=1)

    # sensitivity of the exclusion to the match distance
    print("\nChemical-exclusion sensitivity (appositions flagged chemical):")
    for thr in (250, 500, 1000, 2000):
        n = int((app["nearest_chem_nm"] <= thr).sum())
        print(f"  <= {thr:>4} nm : {n:4d} / {len(app)}  "
              f"-> {len(app)-n} candidates")

    cand = app[~app["is_chemical"]].copy()

    # deduplicate candidates per pair into distinct sites (largest area first)
    sites = []
    for pair, sub in cand.groupby("pair"):
        sub = sub.sort_values("area_um2", ascending=False)
        kept = []
        for _, r in sub.iterrows():
            p = np.array([r.x, r.y, r.z])
            if all(np.sqrt(((p - k) ** 2).sum()) >= SITE_SEP_NM for k in kept):
                kept.append(p)
                sites.append({
                    "pair": pair, "neuron_a": r.neuron_a, "neuron_b": r.neuron_b,
                    "x": r.x, "y": r.y, "z": r.z, "area_um2": round(r.area_um2, 4),
                    "nearest_chem_nm": round(r.nearest_chem_nm, 0),
                    "vx": int(round(r.x / 4)), "vy": int(round(r.y / 4)),
                    "vz": int(round(r.z / 40)),
                })
    sites = pd.DataFrame(sites).sort_values("area_um2", ascending=False)
    sites.to_csv(os.path.join(rd, "gj_candidates.csv"), index=False)

    # per-pair summary
    summ = (app.groupby("pair")
            .agg(n_appositions=("is_chemical", "size"),
                 n_chemical=("is_chemical", "sum"))
            .reset_index())
    summ["n_candidate_appositions"] = summ["n_appositions"] - summ["n_chemical"]
    site_counts = sites.groupby("pair").size().rename("n_candidate_sites")
    summ = summ.merge(site_counts, on="pair", how="left").fillna({"n_candidate_sites": 0})
    summ["n_candidate_sites"] = summ["n_candidate_sites"].astype(int)
    summ = summ.sort_values("n_candidate_sites", ascending=False)
    summ.to_csv(os.path.join(rd, "phase_a_summary.csv"), index=False)

    print(f"\nCandidate sites (deduped, >= {SITE_SEP_NM/1000:g} um apart): "
          f"{len(sites)} across {sites['pair'].nunique()} pairs")
    print(summ.to_string(index=False))

    # ── positive controls (adversarial sanity check) ───────────────────────
    print("\n" + "=" * 60 + "\nPOSITIVE CONTROLS\n" + "=" * 60)
    # 1) the confirmed GJ must survive the filter (it is NOT a chemical synapse)
    dd = np.sqrt(((app[["x", "y", "z"]].values - CONFIRMED_GJ_NM) ** 2).sum(1))
    i = int(dd.argmin())
    row = app.iloc[i]
    print(f"Confirmed MOT_R<->HSN_R GJ: nearest validated apposition {dd[i]:.0f} nm "
          f"({row.pair}); classified = "
          f"{'CHEMICAL (FAIL)' if row.is_chemical else 'candidate (PASS)'}; "
          f"nearest chem synapse {row.nearest_chem_nm:.0f} nm")
    # 2) pairs that DO have chemical synapses should produce chemical exclusions
    chem_pairs = {" <-> ".join(sorted(list(k))) for k in syn_xyz}
    got = set(app[app.is_chemical]["pair"])
    print(f"Pairs with annotated chemistry: {len(chem_pairs)}; "
          f"pairs where appositions were excluded as chemical: {len(got)}")
    print(f"\nWrote gj_candidates.csv ({len(sites)} sites) + phase_a_summary.csv")


if __name__ == "__main__":
    main()
