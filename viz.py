import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np


plt.switch_backend("Agg")


def save_scatter(x: np.ndarray, step: int, output_dir: str, shape: str, sampler: str, t_value: float):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x[:, 0], x[:, 1], s=10, alpha=0.6, color="#4c72b0")
    ax.set_title(f"{shape} – {sampler} step {step} (t={t_value:.3f})")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")

    mean = x.mean(axis=0)
    cov = np.cov(x.T)
    text = f"mean=({mean[0]:.2f},{mean[1]:.2f})\nvar=({cov[0,0]:.2f},{cov[1,1]:.2f})"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left")

    fname = os.path.join(output_dir, f"step_{step:04d}.png")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    return fname
