#!/usr/bin/env python3
"""
edge_prior_circle_vp.py
-----------------------
A compact, heavily commented demonstration of forward and reverse diffusion on a
64×64 circular edge image. The reverse process uses a hand-crafted edge-set
prior encoded as a Laplacian-based score that attracts samples toward a bright
ring. The reverse dynamic is *not* the exact reverse SDE, but a probability-flow
ODE that follows an energy-inspired score field, illustrating how priors can
nudge samples toward desired structures.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly backend
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio

# Reproducibility
RNG = np.random.default_rng(42)

# Diffusion configuration
SIZE = 64
T = 1.0
N_STEPS = 200
DT = T / N_STEPS
BETA0, BETA1 = 0.1, 2.0  # linear beta schedule bounds (kept modest for stability)
SIGMA_PRIOR = 0.2
N_FRAMES = 8  # number of frames to collect for GIFs


def linear_beta(t: float) -> float:
    """Linearly interpolate beta(t) between BETA0 and BETA1."""
    return BETA0 + (BETA1 - BETA0) * (t / T)


def make_circle_edge(size: int = SIZE, radius: float = 18.0, blur: float = 1.5):
    """Create a smooth circular edge map and its Laplacian for reuse.

    Steps
    -----
    1) Build a signed distance field to a circle centered in the image
       (negative inside, positive outside).
    2) Convert the distance into a narrow bright ring via a Gaussian.
    3) Smooth the ring to avoid aliasing and normalize to [0, 1].
    """
    yy, xx = np.indices((size, size))
    cx = cy = (size - 1) / 2.0
    sdf = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius

    # Bright ring centered on the zero level-set of the SDF
    ring = np.exp(-(sdf ** 2) / (2 * (1.0 ** 2)))
    ring = ndimage.gaussian_filter(ring, blur)
    ring = (ring - ring.min()) / (ring.max() - ring.min() + 1e-8)

    lap_ring = ndimage.laplace(ring)
    return ring.astype(np.float32), lap_ring.astype(np.float32), sdf.astype(np.float32)


def compute_score(x: np.ndarray, lap_gs: np.ndarray) -> np.ndarray:
    """Score from the edge prior: -(1/σ^2) (Δx - ΔgS)."""
    return -(1.0 / (SIGMA_PRIOR ** 2)) * (ndimage.laplace(x) - lap_gs)


def energy_proxy(x: np.ndarray, gS: np.ndarray) -> float:
    """Approximate E(x) = (1/(2σ^2))||∇x - ∇gS||^2 using finite differences."""
    gx, gy = np.gradient(x)
    gsx, gsy = np.gradient(gS)
    diff = (gx - gsx) ** 2 + (gy - gsy) ** 2
    return float(0.5 / (SIGMA_PRIOR ** 2) * diff.mean())


def simulate_forward(x0: np.ndarray):
    """Euler–Maruyama simulation of the VP forward SDE on x0."""
    frames = []
    x = x0.copy()
    capture_steps = np.linspace(0, N_STEPS, N_FRAMES, dtype=int)
    for n in range(N_STEPS):
        t = n * DT
        beta = linear_beta(t)
        noise = RNG.normal(size=x.shape)
        x += -0.5 * beta * DT * x + np.sqrt(beta * DT) * noise
        if n in capture_steps:
            frames.append(x.copy())
    frames.append(x.copy())  # ensure final frame
    return frames


def simulate_reverse(lap_gs: np.ndarray):
    """Probability-flow ODE integration with Heun (predictor-corrector).

    The ODE is dx/dt = -0.5 * beta(t) * score(x), where score uses the edge prior.
    """
    x = RNG.normal(size=(SIZE, SIZE)).astype(np.float32)
    frames = []
    energies = []
    # Precompute target gradients for the energy curve
    gS = target_edge
    capture_steps = np.linspace(0, N_STEPS, N_FRAMES, dtype=int)

    for i in range(N_STEPS):
        # We traverse t from T down to 0, so dt is negative.
        t = T - i * DT
        beta = linear_beta(t)
        score = compute_score(x, lap_gs)
        drift = -0.5 * beta * score
        x_pred = x + (-DT) * drift  # predictor step with dt = -DT

        # Corrector: recompute drift at the predicted point (using t - DT)
        beta_pred = linear_beta(t - DT)
        score_pred = compute_score(x_pred, lap_gs)
        drift_pred = -0.5 * beta_pred * score_pred
        x += 0.5 * (-DT) * (drift + drift_pred)
        # Keep values bounded to avoid blow-ups when visualizing.
        x = np.clip(x, -2.0, 2.0)

        if i in capture_steps:
            frames.append(x.copy())
        energies.append(energy_proxy(x, gS))

    frames.append(x.copy())
    return frames, energies


def to_uint8(frames):
    """Convert a list of float images to uint8 grayscale frames."""
    arr = [np.clip(f, 0, 1) for f in frames]
    return [np.uint8(f * 255) for f in arr]


def save_gif(frames, path: str, fps: int = 6):
    imageio.mimsave(path, frames, fps=fps)


if __name__ == "__main__":
    # Build target edge and its Laplacian once.
    target_edge, lap_target, sdf = make_circle_edge()

    # Forward diffusion of the clean target edge map.
    forward_frames = simulate_forward(target_edge)

    # Reverse probability-flow ODE guided by the edge prior.
    reverse_frames, energies = simulate_reverse(lap_target)

    # Save GIFs of both trajectories for visual inspection.
    save_gif(to_uint8(forward_frames), "edge_forward.gif")
    save_gif(to_uint8(reverse_frames), "edge_reverse_edgeprior.gif")

    # Quiver plot of the score vector field on a downsampled grid.
    score_map = compute_score(reverse_frames[len(reverse_frames) // 2], lap_target)
    sy, sx = np.gradient(score_map)
    step = 4
    Y, X = np.mgrid[0:SIZE:step, 0:SIZE:step]
    plt.figure(figsize=(5, 5))
    plt.imshow(target_edge, cmap="gray", origin="lower")
    plt.quiver(X, Y, sx[::step, ::step], sy[::step, ::step], color="cyan", angles="xy")
    plt.title("Score field (grad of score map) toward the circle")
    plt.tight_layout()
    plt.savefig("score_quiver.png", dpi=150)

    # Plot the energy proxy over reverse time.
    plt.figure(figsize=(6, 3))
    times = np.linspace(T, 0, len(energies))
    plt.plot(times, energies, label="E(x_t)")
    plt.xlabel("time t (reverse)")
    plt.ylabel("energy proxy")
    plt.title("Energy decreases as the sample follows the edge prior")
    plt.legend()
    plt.tight_layout()
    plt.savefig("energy_curve.png", dpi=150)

    # Show the target and SDF for reference.
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(target_edge, cmap="gray", origin="lower")
    plt.title("Target edge map gS")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(sdf, cmap="coolwarm", origin="lower")
    plt.title("Signed distance to the circle")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("target_and_sdf.png", dpi=150)

    print("Saved edge_forward.gif, edge_reverse_edgeprior.gif, score_quiver.png, energy_curve.png, target_and_sdf.png")
