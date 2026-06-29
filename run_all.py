#!/usr/bin/env python
"""
Run All MESH Pipeline Scripts
==============================
Executes all analysis and visualization scripts in the correct order:
  1. overlap_analysis.py   – Neuron overlap analysis + 3D HTML figures + EM viewer
  2. generate_skeleton_plots.py – Skeleton-based plots
  3. skeleton_em_viewer.py     – Skeleton EM viewer
  4. generate_em_stacks.py     – EM stack montages

Usage:
    python run_all.py
"""

import os
import sys
import subprocess
import time
import json
import argparse
from datetime import datetime

# Ensure UTF-8 output even when redirected to a file on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Force UTF-8 in the child scripts too. When run_all's output is redirected to a
# file/pipe, a child's stdout defaults to cp1252 on Windows and crashes on
# Unicode like '→' (used in overlap pair keys) or 'µm²'. PYTHONUTF8 is read at
# each child interpreter's startup, so set it before launching subprocesses.
os.environ['PYTHONUTF8'] = '1'
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

from pathlib import Path

from mesh_config import CONFIG_ENV_VAR, config_is_default, load_config, resolve_config_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Auto-load FlyWire token from cave-secret.json if not already set ──
def _ensure_flywire_token():
    """Load FlyWire token into env so child processes inherit it."""
    if os.environ.get("FLYWIRE_TOKEN"):
        return  # already set
    secret_path = Path.home() / ".cloudvolume" / "secrets" / "cave-secret.json"
    if secret_path.is_file():
        try:
            data = json.loads(secret_path.read_text())
            tok = data.get("token", "")
            if tok:
                os.environ["FLYWIRE_TOKEN"] = tok
                print(f"  Token loaded from {secret_path}  ({tok[:8]}…)")
                return
        except Exception as e:
            print(f"  ⚠ Could not read token from {secret_path}: {e}")
    print("  ⚠ FLYWIRE_TOKEN not set and cave-secret.json not found!")
    print("    Set env var FLYWIRE_TOKEN before running.")
    sys.exit(1)

SCRIPTS = [
    ("overlap_analysis.py",        "Overlap analysis + 3D figures"),
    ("reduced_matrix.py",          "Reduced MOT/MOS x HS/VS overlap matrix"),
    ("generate_skeleton_plots.py", "Skeleton plots"),
    ("generate_em_stacks.py",      "EM stack montages (contacts + synapses + overlaps)"),
    ("generate_gj_figures.py",     "Putative gap-junction composite figures (MOT/MOS x partners)"),
    ("skeleton_em_viewer.py",      "Final comprehensive EM viewer (runs last)"),
]


def _find_latest_results():
    """Find the latest comprehensive_overlap_results_* directory."""
    candidates = [d for d in os.listdir(SCRIPT_DIR)
                  if os.path.isdir(os.path.join(SCRIPT_DIR, d))
                  and d.startswith('comprehensive_overlap_results_')]
    if candidates:
        return os.path.join(SCRIPT_DIR, sorted(candidates)[-1])
    return None


def _can_skip_overlap_analysis():
    """Check if overlap_analysis.py can be skipped (key outputs exist)."""
    rd = _find_latest_results()
    if not rd:
        return False, "no results directory"
    # Check key output files
    faces = os.path.join(rd, 'geometric_data', 'contact_faces.csv')
    combined = os.path.join(rd, 'all_results_combined.csv')
    meshes = os.path.join(rd, 'neuron_meshes')
    if not os.path.exists(faces):
        return False, "missing contact_faces.csv"
    if not os.path.exists(combined):
        return False, "missing all_results_combined.csv"
    if not os.path.isdir(meshes) or len(os.listdir(meshes)) == 0:
        return False, "missing neuron meshes"
    # Count neurons in config
    cfg, _ = load_config()
    expected = len(cfg.get('viewer_neurons', cfg.get('neurons', {})))
    n_meshes = len([f for f in os.listdir(meshes) if f.endswith('.obj')])
    if n_meshes < expected:
        return False, f"only {n_meshes}/{expected} meshes"
    return True, rd


def _can_skip_skeleton_plots():
    """Check if skeleton plot dir exists."""
    rd = _find_latest_results()
    if not rd:
        return False, "no results directory"
    plots = os.path.join(rd, 'overlap_plots_skeleton')
    if os.path.isdir(plots) and len(os.listdir(plots)) > 0:
        return True, rd
    return False, "missing skeleton plots"


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the full MESH pipeline.")
    parser.add_argument(
        "--config",
        help="Path to a neuron config JSON file. Defaults to neurons.json or MESH_NEURON_CONFIG.",
    )
    parser.add_argument(
        "--results-dir",
        help="Optional results directory to export via MESH_RESULTS_DIR for all child scripts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Disable cached-step skipping for this run.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config_path = resolve_config_path(args.config)
    if not config_path.is_file():
        print(f"ERROR: config file not found: {config_path}")
        sys.exit(1)

    os.environ[CONFIG_ENV_VAR] = str(config_path)
    if args.results_dir:
        os.environ["MESH_RESULTS_DIR"] = args.results_dir

    start_all = time.time()
    print("=" * 72)
    print("  MESH Pipeline — Run All Scripts")
    print(f"  Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config: {config_path.name}")
    if not config_is_default(config_path):
        print(f"  Config path: {config_path}")
    if os.environ.get("MESH_RESULTS_DIR"):
        print(f"  Results dir override: {os.environ['MESH_RESULTS_DIR']}")
    print("=" * 72)

    _ensure_flywire_token()

    results = []
    for i, (script, description) in enumerate(SCRIPTS, 1):
        script_path = os.path.join(SCRIPT_DIR, script)
        if not os.path.isfile(script_path):
            print(f"\n[{i}/{len(SCRIPTS)}] SKIP {script} — file not found")
            results.append((script, "SKIP"))
            continue

        # Smart skip: avoid re-running expensive steps if outputs exist
        skip = False
        if not args.force and script == "overlap_analysis.py":
            can_skip, info = _can_skip_overlap_analysis()
            if can_skip:
                print(f"\n[{i}/{len(SCRIPTS)}] SKIP {script} — outputs already exist in {os.path.basename(info)}")
                print(f"         (contact_faces.csv, all_results_combined.csv, all meshes present)")
                print(f"         Delete the results dir to force re-run.")
                results.append((script, "SKIP (cached)"))
                skip = True
        elif not args.force and script == "generate_skeleton_plots.py":
            can_skip, info = _can_skip_skeleton_plots()
            if can_skip:
                print(f"\n[{i}/{len(SCRIPTS)}] SKIP {script} — skeleton plots already exist")
                results.append((script, "SKIP (cached)"))
                skip = True

        if skip:
            continue

        print(f"\n{'─' * 72}")
        print(f"[{i}/{len(SCRIPTS)}] {description}")
        print(f"         Running: {script}")
        print(f"{'─' * 72}")

        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=SCRIPT_DIR,
            )
            elapsed = time.time() - t0
            status = "OK" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
        except Exception as e:
            elapsed = time.time() - t0
            status = f"ERROR: {e}"

        m, s = divmod(int(elapsed), 60)
        print(f"\n  → {script}: {status}  ({m}m {s}s)")
        results.append((script, status))

    # Summary
    total = time.time() - start_all
    m, s = divmod(int(total), 60)
    print(f"\n{'=' * 72}")
    print(f"  Pipeline complete — total time: {m}m {s}s")
    print(f"  {'─' * 40}")
    for script, status in results:
        icon = "✓" if status == "OK" else "✗" if "FAIL" in status else "⚠"
        print(f"  {icon}  {script:35s}  {status}")
    print(f"{'=' * 72}")

    # Return non-zero if any script failed
    if any("FAIL" in s or "ERROR" in s for _, s in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
