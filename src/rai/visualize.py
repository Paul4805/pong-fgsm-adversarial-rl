"""Visualization helpers for clean vs. adversarial Pong frames and results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_frame_comparison(
    records: Sequence[dict[str, Any]],
    path: str | Path,
    title: str,
    epsilon: float,
    amplification: float = 40.0,
) -> Path:
    """Save a clean/adv/perturbation grid for selected timesteps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return path

    n_rows = len(records)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(9.0, 2.75 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for row, record in enumerate(records):
        clean = np.asarray(record["clean_obs"], dtype=np.float32)[-1]
        adv = np.asarray(record["adv_obs"], dtype=np.float32)[-1]
        diff = adv - clean
        step = int(record["step"])
        clean_action = int(record["clean_action"])
        adv_action = int(record["adv_action"])

        axes[row, 0].imshow(clean, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 0].set_ylabel(
            f"step {step}\naction {clean_action} -> {adv_action}",
            fontsize=9,
        )
        axes[row, 1].imshow(adv, cmap="gray", vmin=0.0, vmax=1.0)
        image = axes[row, 2].imshow(
            diff * amplification,
            cmap="bwr",
            vmin=-1.0,
            vmax=1.0,
        )
        if row == 0:
            axes[row, 0].set_title("Clean frame", fontsize=10)
            axes[row, 1].set_title("Adversarial frame", fontsize=10)
            axes[row, 2].set_title(
                f"Perturbation x{amplification:g} (max |d|={np.abs(diff).max():.4f})",
                fontsize=10,
            )
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        fig.colorbar(image, ax=axes[row, 2], fraction=0.046, pad=0.02)

    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_summary_plots(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """Bar charts for episode return and action-change rate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return path

    labels = [f"{r['mode']}\neps={r['epsilon']:g}" for r in rows]
    clean = np.array([r["clean_return_mean"] for r in rows], dtype=np.float64)
    adv = np.array([r["adv_return_mean"] for r in rows], dtype=np.float64)
    change = np.array([r["action_change_rate"] for r in rows], dtype=np.float64) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    x = np.arange(len(rows))
    width = 0.36
    axes[0].bar(x - width / 2, clean, width, label="Clean", color="#4C78A8")
    axes[0].bar(x + width / 2, adv, width, label="Adversarial", color="#E45756")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean episode return")
    axes[0].set_title("Return: clean vs. FGSM-perturbed policy")
    axes[0].legend(frameon=False)

    bars = axes[1].bar(x, change, color="#F58518")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Action-change rate (%)")
    axes[1].set_title("Fraction of attacked steps with a different action")
    axes[1].set_ylim(0.0, max(100.0, float(change.max()) * 1.15 if len(change) else 100.0))
    for bar, value in zip(bars, change):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
