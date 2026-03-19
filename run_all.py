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
from datetime import datetime

# Ensure UTF-8 output even when redirected to a file on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

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
    ("generate_skeleton_plots.py", "Skeleton plots"),
    ("generate_em_stacks.py",      "EM stack montages (contacts + synapses + overlaps)"),
    ("skeleton_em_viewer.py",      "Final comprehensive EM viewer (runs last)"),
]


def main():
    start_all = time.time()
    print("=" * 72)
    print("  MESH Pipeline — Run All Scripts")
    print(f"  Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    _ensure_flywire_token()

    results = []
    for i, (script, description) in enumerate(SCRIPTS, 1):
        script_path = os.path.join(SCRIPT_DIR, script)
        if not os.path.isfile(script_path):
            print(f"\n[{i}/{len(SCRIPTS)}] SKIP {script} — file not found")
            results.append((script, "SKIP"))
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
