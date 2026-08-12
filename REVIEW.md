# MESH overlap-analysis pipeline — review

_Reviewed 2026-07 (publication branch). Scope: correctness, reproducibility,
documentation, efficiency across the full pipeline._

## Verdict

The pipeline is in good shape: modular scripts, a clean `run_all.py` driver,
config-driven neuron sets (`configs/*.json`), and generally solid docstrings. The
core geometric result (mesh proximity → contact patches → seg-adjacency
confirmation) is sound and now validated. The issues below are mostly
**efficiency** and **reproducibility hardening**, not correctness errors that
would invalidate results.

Severity: 🔴 fix before reuse · 🟠 should fix · 🟢 nice-to-have. ✅ = fixed in this pass.

---

## Correctness & reproducibility

**🟠 "Any-vertex" face inclusion over-counts contact area.**
`calculate_neuron_overlap_simple` counts a face as "in contact" if *any* of its 3
vertices is within 100 nm of the partner (`overlap_analysis.py`). A face with one
close vertex and two distant ones still contributes its *full* triangle area. This
systematically inflates the mesh contact area at patch boundaries — a real
contributor to the ~30 % overestimate we measured against seg-adjacency
(389 → 272 µm²). Options: weight each face by the fraction of close vertices, or
require ≥2 close vertices. Keep the current behaviour if you treat the seg-
adjacency `confined_area` (in `overlap_validation.json`) as the authoritative
number — which is the right call for the paper.

**🟠 Contact area is one-sided (mesh A only).** The area is the area of A's faces
near B, not the mutual apposition. Downstream (`reduced_matrix._area`) takes
`max(A→B, B→A)`, which papers over it, but the per-direction numbers in
`matrix_overlap_area_*.csv` are not symmetric by construction. Document this, or
average the two directions.

**🟠 Non-deterministic large-mesh path.** `calculate_large_mesh_overlap` uses
`np.random.choice(...)` with no seed for vertex/face subsampling and then
extrapolates area by a scale factor — so areas change run-to-run and carry high
variance. Only triggers for >1 M-vertex meshes (decimated LPTCs are usually
under that, so the publication cells use the exact path — good). If any control
cell trips it, seed the RNG (`np.random.default_rng(0)`) and record it.

**🟢 Coordinate/scale conventions.** The 4×4×40 vs 4×4×80 nm confusion bit us
earlier (confirmed GJ marker). Now centralised: `generate_em_stacks._coord_tag`
and the viewer's `_coord_tag_xyz` both use 4×4×40. Worth a one-paragraph
"coordinate conventions" note at the top of `mesh_config.py` so it can't drift
again.

## Efficiency

**✅ EM snapshot download parallelised** (`generate_em_stacks.py`). Was serial
(`_cv_pool = ThreadPoolExecutor(max_workers=1)`), ~15 h for a full rebuild.
Now `EM_WORKERS` (env `MESH_EM_WORKERS`, default 6) concurrent CloudVolume
requests, and the overlap loop dispatches one task per Z-slice through a thread
pool. Verified output-identical (pixel- and md5-equal) to the serial path on a
6-file sample; a skip-only pass over 52 k slices now takes ~11 s.
Further headroom (not yet done): fetch the whole z-range of a cluster as **one**
3-D cutout and slice locally, to also cut per-request latency.

**🟠 Stale overlay colours in `em_snaps/`.** The verification above surfaced a
pre-existing inconsistency: some overlap snapshots were rendered with an older VS
colour (189,53,0) before it was changed to the current `#D14900` = (209,73,0), so
the viewer's EM library mixes two VS oranges. The publication *figures*
(`generate_gj_figures.py`) render EM insets fresh from the config, so they're
unaffected — this is viewer-only. To make the viewer fully colour-consistent,
delete `em_snaps/overlap_*` and re-run `generate_em_stacks.py` (now fast).

**✅ Vectorised face-selection.** The main overlap path did
`any(v in close_vertices for v in face)` with `close_vertices` a NumPy array — an
O(N) scan per vertex per face (hundreds of millions of Python ops per LPTC pair).
Replaced with a boolean vertex mask + `mask[faces].any(axis=1)`. Identical output,
orders of magnitude faster.

**🟢 `overlap_analysis.py` is 3.3 k lines** with four near-duplicate HTML figure
builders (mesh/points/wireframe/lite) that share ~80 % of their code. Factoring the
shared trace/checkbox assembly into one helper would cut ~600 lines and the
maintenance surface.

## Pipeline wiring & housekeeping

**🟠 New analysis steps aren't in `run_all.py`.** `validate_overlaps.py`
(seg-adjacency confinement) and `generate_gj_figures_extended.py` exist but aren't
in the `SCRIPTS` list, so a fresh `run_all` won't produce `overlap_validation.json`
or the extended figures. Add them (validation after `overlap_analysis`, before the
viewer, so the viewer's axonal markers are available).

**🟢 Two viewers coexist.** `em_viewer.py` (1.4 k lines) appears superseded by
`skeleton_em_viewer.py`. If it's legacy, move it to `legacy/` or delete; if it's
still used, note the split-of-responsibility in the header.

**🟢 Reproducibility bundle.** For "others can adapt it," add a short `README` with:
the two config files and what each covers, the exact `run_all` invocation
(`PYTHONUTF8=1 MESH_NEURON_CONFIG=... python run_all.py`), the FlyWire data
version (v14.1 / `flywire_v141_m783`), and the fact that aligned EM is only 8 nm
(native 4 nm is in v14 space and misaligned). Pin versions in a
`requirements.txt` (navis, cloud-volume, trimesh, pyfqmr, scikit-image, scipy).

## Documentation quality

Good overall — most functions have docstrings and the script headers explain the
data flow. Gaps worth closing: (a) the coordinate-convention note above; (b) a
top-level data-flow diagram (raw seg → meshes → contact patches → validation →
EM snapshots → viewer/figures); (c) the meaning of the two area numbers
(mesh vs confined) wherever both appear.
