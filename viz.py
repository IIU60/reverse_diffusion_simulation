"""Visualization helpers for particle trajectories."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_frame(
    x: np.ndarray,
    gmm,
    step: int,
    total_steps: int,
    t_value: float,
    output_dir: str,
    sampler: str,
    shape: str,
    score_mean: Optional[float] = None,
    max_radius: Optional[float] = None,
) -> str:
    """Save a single scatter frame with an accompanying text panel."""

    ensure_dir(output_dir)
    fig, (ax_scatter, ax_text) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [3, 1]})

    ax_scatter.scatter(x[:, 0], x[:, 1], s=2, alpha=0.25, color="tab:blue")
    ax_scatter.scatter(gmm.mu[:, 0], gmm.mu[:, 1], s=12, c="gray", marker="x", alpha=0.4)
    ax_scatter.set_title(f"{shape} | {sampler.upper()} | step {step}/{total_steps}")
    ax_scatter.set_aspect("equal")
    ax_scatter.axis("off")

    text_lines = [
        f"t = {t_value:.4f}",
        f"step = {step}/{total_steps}",
    ]
    if score_mean is not None:
        text_lines.append(f"mean||score|| = {score_mean:.2f}")
    if max_radius is not None:
        text_lines.append(f"max||x|| = {max_radius:.2f}")
    ax_text.axis("off")
    ax_text.text(0.0, 0.95, "\n".join(text_lines), va="top", ha="left")

    fig.tight_layout()
    fname = os.path.join(output_dir, f"frame_{step:05d}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def save_trails(trajs: np.ndarray, output_dir: str, sampler: str, shape: str) -> str:
    """Plot trajectory trails for a subset of particles.

    Args:
        trajs: array shaped (T, M, 2)
    """

    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(5, 5))
    T, M, _ = trajs.shape
    colors = plt.cm.viridis(np.linspace(0, 1, M))
    for m in range(M):
        ax.plot(trajs[:, m, 0], trajs[:, m, 1], color=colors[m], alpha=0.6, linewidth=0.8)
    ax.scatter(trajs[-1, :, 0], trajs[-1, :, 1], s=10, color="black", alpha=0.6, label="final")
    ax.set_title(f"Trails | {shape} | {sampler.upper()} | {T} steps")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fname = os.path.join(output_dir, "trails.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname
