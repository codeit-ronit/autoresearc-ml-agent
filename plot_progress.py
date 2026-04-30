#!/usr/bin/env python3
"""
plot_progress.py — Autoresearch Progress Visualizer
====================================================
Reads results.tsv and generates progress.png in the style of
Karpathy's autoresearch progress chart.

Usage:
    uv run plot_progress.py            # generate progress.png once
    uv run plot_progress.py --watch    # regenerate every 60s
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

RESULTS_FILE = "results.tsv"
OUTPUT_FILE = "progress.png"


def load_results(path: str = RESULTS_FILE) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [r for r in reader if r.get("status", "") != ""]
    return rows


def plot(rows: list[dict], save_path: str = OUTPUT_FILE) -> None:
    if not rows:
        print("No data in results.tsv yet — nothing to plot.")
        return

    # ── Parse data ────────────────────────────────────────────────────────────
    exp_nums   = []
    accuracies = []
    statuses   = []
    hypotheses = []

    for i, r in enumerate(rows):
        try:
            acc = float(r["accuracy"])
        except (ValueError, KeyError):
            continue
        exp_nums.append(i)
        accuracies.append(acc)
        statuses.append(r.get("status", "REVERTED"))
        hypotheses.append(r.get("hypothesis", ""))

    if not exp_nums:
        print("No valid rows to plot.")
        return

    # ── Running best line ─────────────────────────────────────────────────────
    running_best = []
    best = 0.0
    for acc, st in zip(accuracies, statuses):
        if st in ("COMMITTED", "BASELINE"):
            best = max(best, acc)
        running_best.append(best)

    # ── Split into kept vs discarded ──────────────────────────────────────────
    kept_x, kept_y, kept_labels = [], [], []
    disc_x, disc_y = [], []

    for x, y, st, hyp in zip(exp_nums, accuracies, statuses, hypotheses):
        if st in ("COMMITTED", "BASELINE"):
            kept_x.append(x)
            kept_y.append(y)
            # Shorten label: first 40 chars
            kept_labels.append(hyp[:42] if hyp else "baseline")
        else:
            disc_x.append(x)
            disc_y.append(y)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0d1117")   # GitHub dark background
    ax.set_facecolor("#0d1117")

    # Grid
    ax.grid(True, color="#21262d", linewidth=0.6, linestyle="--", zorder=0)

    # Running best line
    ax.plot(
        exp_nums, running_best,
        color="#3fb950", linewidth=2.0, zorder=2,
        label="Running best", alpha=0.9,
    )

    # Discarded experiments
    if disc_x:
        ax.scatter(
            disc_x, disc_y,
            color="#8b949e", s=30, marker="o", alpha=0.45,
            zorder=3, label="Discarded",
        )

    # Kept experiments
    if kept_x:
        ax.scatter(
            kept_x, kept_y,
            color="#3fb950", s=70, marker="o", zorder=4,
            label="Kept", edgecolors="#ffffff", linewidths=0.6,
        )

        # Diagonal labels on kept points (Karpathy style)
        for x, y, label in zip(kept_x, kept_y, kept_labels):
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=6.5,
                color="#e6edf3",
                rotation=35,
                rotation_mode="anchor",
                ha="left",
                va="bottom",
                zorder=5,
                path_effects=[
                    pe.withStroke(linewidth=2, foreground="#0d1117")
                ],
            )

    # ── Axes styling ──────────────────────────────────────────────────────────
    n_kept = len(kept_x)
    n_total = len(exp_nums)

    ax.set_title(
        f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements",
        color="#e6edf3", fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Experiment #", color="#8b949e", fontsize=11)
    ax.set_ylabel("Validation Accuracy (higher is better)", color="#8b949e", fontsize=11)

    ax.tick_params(colors="#8b949e", which="both")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    # Y-axis: leave a little headroom above/below the data range
    if accuracies:
        lo = min(accuracies) - 0.005
        hi = max(accuracies) + 0.010
        # Clamp to sensible range for SST-2
        lo = max(0.50, lo)
        hi = min(1.00, hi)
        ax.set_ylim(lo, hi)

    # Legend
    legend = ax.legend(
        facecolor="#161b22", edgecolor="#30363d",
        labelcolor="#e6edf3", fontsize=9,
    )

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {save_path}  ({n_total} experiments, {n_kept} kept)")


def main():
    watch = "--watch" in sys.argv

    rows = load_results()
    plot(rows)

    if watch:
        print("Watch mode — regenerating every 60 seconds. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(60)
                rows = load_results()
                plot(rows)
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()
