#!/usr/bin/env python
"""
Reduced MOT/MOS x HS/VS overlap matrix (publication panel j)
============================================================
Compact binary overlap matrix in the style of the publication figure:

    rows    = MOS, MOT  x  right / left          (4 rows; hemisphere as a bracket)
    columns = HSN, HSS, HSE | VS1-4 | VS5-8       (cell TYPES, L/R collapsed)

A cell is filled (solid, non-gradient) when the row's motor neuron overlaps the
same-hemisphere LPTC of that column type:
    MOS rows -> MOS color (#4D9221)
    MOT rows -> MOT color (#5E3C99)
Empty cells are white. Column-group color bars sit on top
(HS #C51B7D, VS1-4 #D14900, VS5-8 #007F5F control).

Overlap is geometric and (near-)symmetric; the pipeline stores both directions
(source->target and target->source) and we use the larger of the two.

Input : matrix_overlap_area_*.csv from the latest (or a given) results dir.
Output: reduced_overlap_matrix.png / .svg (white background, publication ready).

Usage:
    python reduced_matrix.py [--results-dir DIR] [--config CFG] [--out-dir DIR]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mesh_config import load_config

# Group / motor colors (publication spec).
COLOR_MOT = "#5E3C99"
COLOR_MOS = "#4D9221"
COLOR_HS = "#C51B7D"
COLOR_VS1_4 = "#D14900"
COLOR_VS5_8 = "#007F5F"

MIN_AREA_UM2 = 0.5  # contacts smaller than this (um^2) count as no overlap

# Rows: (motor cell, motor color, hemisphere label). Top -> bottom.
ROW_SPECS = [
    ("MOS_R", COLOR_MOS, "right"),
    ("MOT_R", COLOR_MOT, "right"),
    ("MOS_L", COLOR_MOS, "left"),
    ("MOT_L", COLOR_MOT, "left"),
]

# Columns: (cell type, group key, group color). Left -> right.
COL_SPECS = [
    ("HSN", "HS", COLOR_HS), ("HSS", "HS", COLOR_HS), ("HSE", "HS", COLOR_HS),
    ("VS1", "VS1-4", COLOR_VS1_4), ("VS2", "VS1-4", COLOR_VS1_4),
    ("VS3", "VS1-4", COLOR_VS1_4), ("VS4", "VS1-4", COLOR_VS1_4),
    ("VS5", "VS5-8", COLOR_VS5_8), ("VS6", "VS5-8", COLOR_VS5_8),
    ("VS7", "VS5-8", COLOR_VS5_8), ("VS8", "VS5-8", COLOR_VS5_8),
]


def _find_latest_results():
    base = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(base, d) for d in os.listdir(base)
             if os.path.isdir(os.path.join(base, d))
             and d.startswith("comprehensive_overlap_results_")]
    return sorted(cands)[-1] if cands else None


def _find_area_matrix(results_dir):
    hits = sorted(glob.glob(os.path.join(results_dir, "matrix_overlap_area_*.csv")))
    return hits[-1] if hits else None


def _area(area_df, a, b):
    """Direction-agnostic overlap area between cells a and b (0 if absent)."""
    v = 0.0
    if a in area_df.index and b in area_df.columns:
        v = max(v, float(area_df.at[a, b]))
    if b in area_df.index and a in area_df.columns:
        v = max(v, float(area_df.at[b, a]))
    return v


def build_overlap_grid(area_df):
    """Return a boolean grid [n_rows x n_cols]: motor(row hemi) overlaps type(row hemi)."""
    n_rows, n_cols = len(ROW_SPECS), len(COL_SPECS)
    grid = np.zeros((n_rows, n_cols), dtype=bool)
    for i, (motor_cell, _c, hemi) in enumerate(ROW_SPECS):
        suffix = "_R" if hemi == "right" else "_L"
        for j, (ctype, _g, _cc) in enumerate(COL_SPECS):
            col_cell = ctype + suffix
            grid[i, j] = _area(area_df, motor_cell, col_cell) >= MIN_AREA_UM2
    return grid


def render(grid, out_dir):
    n_rows, n_cols = grid.shape
    cell = 1.0  # cell size in data units

    fig_w = 0.46 * n_cols + 2.4
    fig_h = 0.46 * n_rows + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.set_facecolor("white")

    # ── matrix cells (binary solid fill) ──
    for i in range(n_rows):
        fill = ROW_SPECS[i][1]
        for j in range(n_cols):
            x, y = j, n_rows - 1 - i  # row 0 at top
            ax.add_patch(Rectangle((x, y), cell, cell,
                                   facecolor=(fill if grid[i, j] else "white"),
                                   edgecolor="#444444", linewidth=0.8))

    # ── top column-group color bars + group labels ──
    bar_y = n_rows + 0.18
    bar_h = 0.42
    j = 0
    while j < n_cols:
        g = COL_SPECS[j][1]
        k = j
        while k < n_cols and COL_SPECS[k][1] == g:
            k += 1
        ax.add_patch(Rectangle((j, bar_y), (k - j), bar_h,
                               facecolor=COL_SPECS[j][2], edgecolor="none"))
        ax.text((j + k) / 2.0, bar_y + bar_h + 0.18, g, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=COL_SPECS[j][2])
        j = k

    # ── right-side hemisphere brackets (right = top 2 rows, left = bottom 2) ──
    bx = n_cols + 0.25
    for hemi, top_row in (("right", 0), ("left", 2)):
        y_hi = n_rows - top_row
        y_lo = n_rows - (top_row + 2)
        ax.plot([bx, bx], [y_lo + 0.06, y_hi - 0.06], color="#222222", lw=1.4)
        ax.plot([bx, bx - 0.12], [y_hi - 0.06, y_hi - 0.06], color="#222222", lw=1.4)
        ax.plot([bx, bx - 0.12], [y_lo + 0.06, y_lo + 0.06], color="#222222", lw=1.4)
        ax.text(bx + 0.18, (y_lo + y_hi) / 2.0, hemi, ha="left", va="center",
                fontsize=9, rotation=270, color="#111111")

    # ── row labels (motor type only; hemisphere is in the bracket) ──
    for i, (motor_cell, color, _hemi) in enumerate(ROW_SPECS):
        y = n_rows - 1 - i + 0.5
        ax.text(-0.18, y, motor_cell.split("_")[0], ha="right", va="center",
                fontsize=10, color=color, fontweight="bold")

    # ── column labels (cell type, tinted by group) ──
    for j, (ctype, _g, cc) in enumerate(COL_SPECS):
        ax.text(j + 0.5, -0.18, ctype, ha="center", va="top",
                fontsize=8, rotation=90, color=cc)

    ax.set_xlim(-1.4, n_cols + 1.2)
    ax.set_ylim(-1.6, n_rows + bar_h + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "reduced_overlap_matrix.png")
    svg = os.path.join(out_dir, "reduced_overlap_matrix.svg")
    fig.savefig(png, dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(svg, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png, svg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", help="results dir (default: latest)")
    ap.add_argument("--config", help="neuron config (default: active / neurons.json)")
    ap.add_argument("--out-dir", help="output dir (default: the results dir)")
    args = ap.parse_args()

    cfg, cfg_path = load_config(args.config)

    results_dir = args.results_dir or _find_latest_results()
    if not results_dir or not os.path.isdir(results_dir):
        sys.exit("ERROR: no results directory found.")
    area_csv = _find_area_matrix(results_dir)
    if not area_csv:
        sys.exit(f"ERROR: no matrix_overlap_area_*.csv in {results_dir}")

    print(f"Config      : {os.path.basename(str(cfg_path))}")
    print(f"Results dir : {os.path.basename(results_dir)}")
    print(f"Area matrix : {os.path.basename(area_csv)}")

    area_df = pd.read_csv(area_csv, index_col=0)
    grid = build_overlap_grid(area_df)
    filled = int(grid.sum())
    print(f"Overlap cells filled: {filled}/{grid.size}")

    out_dir = args.out_dir or results_dir
    png, svg = render(grid, out_dir)
    print(f"Wrote: {png}")
    print(f"Wrote: {svg}")


if __name__ == "__main__":
    main()
