"""
Demonstration of a toy variance-preserving diffusion model on a synthetic
low-dimensional manifold of stripe patterns. We show how forward noise
pushes samples away from the manifold and how simple priors (PCA/KDE)
can guide a reverse-style probability flow ODE back toward it.

The code is intentionally lightweight and heavily commented for
educational purposes. Only numpy/matplotlib/imageio are used.
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio

# Reproducibility -----------------------------------------------------------
np.random.seed(0)

# Global image configuration
IMG_SIZE = 32
DIM = IMG_SIZE * IMG_SIZE
DATASET_SIZE = 200
TOP_K = 8

# ----------------------------------------------------------------------------
# Synthetic dataset: horizontal/vertical stripes with small shifts.
# ----------------------------------------------------------------------------

def generate_stripe_dataset(n_samples: int) -> np.ndarray:
    """Create a small dataset of binary stripe patterns.

    Each image is either horizontal or vertical stripes with a random
    phase/shift. The images live in a low-dimensional manifold compared
    to the full 1024-D pixel space.
    """

    xs = np.linspace(0, 2 * np.pi, IMG_SIZE, endpoint=False)
    coords = np.stack(np.meshgrid(xs, xs), axis=-1)  # shape (H, W, 2)
    dataset = []
    for _ in range(n_samples):
        vertical = np.random.rand() < 0.5
        shift = np.random.uniform(0, 2 * np.pi)
        if vertical:
            pattern = np.sin(coords[..., 0] * 4 + shift)  # vary along x
        else:
            pattern = np.sin(coords[..., 1] * 4 + shift)  # vary along y
        # Normalize to [0, 1] for nicer visualization.
        img = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        dataset.append(img.reshape(-1))
    return np.stack(dataset, axis=0)  # (N, D)


def standardize(data: np.ndarray) -> np.ndarray:
    """Standardize to zero mean and unit std for each feature."""

    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-6
    return (data - mean) / std


# ----------------------------------------------------------------------------
# PCA utilities.
# ----------------------------------------------------------------------------

def compute_pca(data: np.ndarray, k: int):
    """Compute top-k principal components using SVD."""

    mu = data.mean(axis=0)
    centered = data - mu
    # SVD on normalized data for stability.
    u, s, vt = np.linalg.svd(centered / np.sqrt(len(data) - 1), full_matrices=False)
    components = vt[:k].T  # (D, k)
    return mu, components


def project_onto_pcs(x: np.ndarray, mu: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """Projection onto the PCA subspace: mu + U U^T (x - mu)."""

    centered = x - mu
    coeffs = comps.T @ centered  # (k,)
    return mu + comps @ coeffs


def explained_variance_fraction(x: np.ndarray, mu: np.ndarray, comps: np.ndarray) -> float:
    """Fraction of energy captured by the PCA subspace for a single sample."""

    centered = x - mu
    energy_total = np.sum(centered ** 2) + 1e-8
    coeffs = comps.T @ centered
    energy_pca = np.sum(coeffs ** 2)
    return float(energy_pca / energy_total)


# ----------------------------------------------------------------------------
# Simple priors (scores).
# ----------------------------------------------------------------------------

def pca_score(x: np.ndarray, mu: np.ndarray, comps: np.ndarray, sigma2: float = 0.05) -> np.ndarray:
    """Heuristic score for the PCA Gaussian prior.

    p(x) ∝ exp(-||x - Proj(x)||^2 / (2 σ^2))
    Score ≈ -∇ log p(x) ≈ -(x - Proj(x)) / σ^2 (points back to the subspace).
    """

    proj = project_onto_pcs(x, mu, comps)
    return -(x - proj) / sigma2


def kde_score(x: np.ndarray, data: np.ndarray, sigma2: float = 0.2) -> np.ndarray:
    """Score of an isotropic Gaussian KDE over the dataset.

    score(x) = Σ_i w_i (x_i - x) / σ^2, where w_i ∝ exp(-||x - x_i||^2 / (2σ^2)).
    This is the gradient of log density for an equally weighted Gaussian mixture.
    """

    diffs = data - x  # (N, D)
    sq_dist = np.sum(diffs ** 2, axis=1)
    # Numerical stability via max subtraction.
    max_term = np.max(-sq_dist / (2 * sigma2))
    weights = np.exp(-sq_dist / (2 * sigma2) - max_term)
    weights = weights / (weights.sum() + 1e-8)
    return (weights[:, None] * diffs).sum(axis=0) / sigma2


# ----------------------------------------------------------------------------
# Diffusion utilities.
# ----------------------------------------------------------------------------

def beta_schedule(t: float, beta0: float = 1e-4, beta1: float = 0.02) -> float:
    """Linear beta(t) on [0, 1]."""

    return beta0 + (beta1 - beta0) * t


def forward_vp_sample(x0: np.ndarray, n_steps: int = 200) -> tuple:
    """Simulate a VP forward SDE starting from a clean image.

    Returns the full trajectory and evenly spaced frames for visualization.
    """

    dt = 1.0 / n_steps
    x = x0.copy()
    trajectory = [x.copy()]
    frames = []
    save_indices = np.linspace(0, n_steps, 8, dtype=int)
    for step in range(1, n_steps + 1):
        t = step * dt
        beta_t = beta_schedule(t)
        noise = np.random.randn(*x.shape)
        # VP SDE: dx = -0.5 beta x dt + sqrt(beta) dW.
        x = x + (-0.5 * beta_t * x) * dt + np.sqrt(beta_t * dt) * noise
        trajectory.append(x.copy())
        if step in save_indices:
            frames.append(x.copy())
    return np.stack(trajectory), frames


def heun_reverse(score_fn, x_T: np.ndarray, n_steps: int = 200) -> tuple:
    """Integrate dx/dt = -0.5 beta(t) score(x, t) backward from t=1 to 0.

    The drift is a simplified probability-flow ODE using a heuristic score
    (PCA or KDE). Heun's method provides a simple predictor-corrector scheme.
    """

    dt = -1.0 / n_steps
    x = x_T.copy()
    trajectory = [x.copy()]
    frames = []
    save_indices = np.linspace(0, n_steps, 8, dtype=int)
    for step in range(1, n_steps + 1):
        t = 1.0 - (step - 1) / n_steps  # current time
        beta_t = beta_schedule(t)

        def drift(x_in):
            return -0.5 * beta_t * score_fn(x_in, t)

        k1 = drift(x)
        x_pred = x + dt * k1
        k2 = drift(x_pred)
        x = x + dt * 0.5 * (k1 + k2)
        trajectory.append(x.copy())
        if step in save_indices:
            frames.append(x.copy())
    return np.stack(trajectory), frames


# ----------------------------------------------------------------------------
# Visualization helpers.
# ----------------------------------------------------------------------------

def to_image_grid(flat: np.ndarray) -> np.ndarray:
    """Reshape a flat vector to a 2D image."""

    return flat.reshape(IMG_SIZE, IMG_SIZE)


def normalize_img(img: np.ndarray) -> np.ndarray:
    """Scale image to uint8 for GIF saving."""

    min_v, max_v = img.min(), img.max()
    if max_v - min_v < 1e-6:
        scaled = np.zeros_like(img)
    else:
        scaled = (img - min_v) / (max_v - min_v)
    return (scaled * 255).astype(np.uint8)


def save_gif(frames, filename: str):
    imgs = [normalize_img(to_image_grid(f)) for f in frames]
    imageio.mimsave(filename, imgs, duration=0.6)


def plot_ev_curves(forward_ev, reverse_ev, filename: str):
    ts = np.linspace(0, 1, len(forward_ev))
    plt.figure(figsize=(6, 4))
    plt.plot(ts, forward_ev, label="Forward EV", color="tab:blue")
    plt.plot(ts, reverse_ev, label="Reverse EV", color="tab:orange")
    plt.xlabel("Time t")
    plt.ylabel(f"Energy fraction in top {TOP_K} PCs")
    plt.title("Explained variance along diffusion trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# ----------------------------------------------------------------------------
# Main demo.
# ----------------------------------------------------------------------------

def main():
    # Build dataset and PCA basis (our proxy for the data manifold).
    raw_data = generate_stripe_dataset(DATASET_SIZE)
    data = standardize(raw_data)
    mu, comps = compute_pca(data, TOP_K)

    # Choose a clean example and simulate the forward VP process.
    x0 = data[0]
    forward_traj, forward_frames = forward_vp_sample(x0)
    forward_ev = [explained_variance_fraction(x, mu, comps) for x in forward_traj]
    save_gif(forward_frames, "pca_forward.gif")

    # Reverse probability-flow using the PCA score.
    def score_pca(x, _t):
        return pca_score(x, mu, comps, sigma2=0.05)

    x_T = np.random.randn(DIM)  # pure noise
    reverse_traj_pca, reverse_frames_pca = heun_reverse(score_pca, x_T)
    reverse_ev_pca = [explained_variance_fraction(x, mu, comps) for x in reverse_traj_pca]
    save_gif(reverse_frames_pca, "pca_reverse_pca.gif")

    # Optional: KDE-based score to demonstrate a different prior.
    def score_kde(x, _t):
        return kde_score(x, data, sigma2=0.4)

    reverse_traj_kde, reverse_frames_kde = heun_reverse(score_kde, x_T)
    save_gif(reverse_frames_kde, "pca_reverse_kde.gif")

    # For plotting, use the PCA-based reverse EV curve.
    plot_ev_curves(forward_ev, reverse_ev_pca, "pca_ev_curves.png")

    # Textual notes for the user.
    print(
        "Saved GIFs: pca_forward.gif (forward diffusion), "
        "pca_reverse_pca.gif (reverse with PCA score), "
        "pca_reverse_kde.gif (reverse with KDE score)."
    )
    print(
        "The explained variance plot shows how forward diffusion erodes structure, "
        "while the simple scores nudge samples back toward the low-dimensional stripe manifold."
    )


if __name__ == "__main__":
    main()
