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
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    ("overlap_analysis.py",       "Overlap analysis + 3D figures + EM viewer"),
    ("generate_skeleton_plots.py", "Skeleton plots"),
    ("skeleton_em_viewer.py",      "Skeleton EM viewer"),
    ("generate_em_stacks.py",      "EM stack montages"),
]


def main():
    start_all = time.time()
    print("=" * 72)
    print("  MESH Pipeline — Run All Scripts")
    print(f"  Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

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
                timeout=14400,  # 4-hour timeout
            )
            elapsed = time.time() - t0
            status = "OK" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            status = "TIMEOUT"
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
