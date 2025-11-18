"""
Demonstrate VP diffusion on a checkerboard using an analytic spectral score.
The script creates forward and reverse trajectories, saves GIFs, and plots
spectral energy outside the checkerboard's dominant frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio

# Reproducibility
np.random.seed(42)

# ---------------------------- Utility functions ---------------------------- #

def make_checkerboard(size: int = 64, tiles: int = 8) -> np.ndarray:
    """Create a clean checkerboard image in [0, 1]."""
    tile_size = size // tiles
    x = np.arange(size)
    y = np.arange(size)
    xv, yv = np.meshgrid(x, y, indexing="ij")
    pattern = ((xv // tile_size) + (yv // tile_size)) % 2
    return pattern.astype(np.float32)


def make_frequency_mask(base_image: np.ndarray, radius: int = 1, thresh: float = 0.5) -> np.ndarray:
    """Create a mask that zeros out the dominant checkerboard frequencies.

    Args:
        base_image: clean checkerboard to analyze.
        radius: neighborhood around each detected spike to keep as allowed.
        thresh: fraction of the maximum FFT magnitude to identify spikes.
    Returns:
        Mask M with 0 on allowed frequencies and 1 elsewhere.
    """
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(base_image)))
    peak_value = fft_mag.max()
    allowed = fft_mag > (thresh * peak_value)

    # Grow a small neighborhood around each detected peak.
    size = base_image.shape[0]
    expanded = np.zeros_like(allowed, dtype=bool)
    coords = np.argwhere(allowed)
    for u, v in coords:
        u_min, u_max = max(u - radius, 0), min(u + radius + 1, size)
        v_min, v_max = max(v - radius, 0), min(v + radius + 1, size)
        expanded[u_min:u_max, v_min:v_max] = True

    # Mask is 0 on allowed frequencies, 1 elsewhere.
    mask = (~expanded).astype(np.float32)
    return np.fft.ifftshift(mask)  # shift back to match fft2 indexing


def spectral_score(x: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Compute spectral prior score using the provided mask."""
    freq = np.fft.fft2(x)
    penalized = mask * freq
    score_freq = penalized / (sigma ** 2)
    score = np.fft.ifft2(score_freq).real
    # Negative sign because score = grad log p(x) = -(1/sigma^2) F^{-1}(M ∘ F(x))
    return -score


def linear_beta(t: float, beta_min: float = 0.1, beta_max: float = 20.0) -> float:
    return beta_min + (beta_max - beta_min) * t


# ---------------------------- Forward process ----------------------------- #

def forward_vp(x0: np.ndarray, mask: np.ndarray, steps: int = 200, T: float = 1.0,
                sigma: float = 0.1):
    """Simulate the forward VP SDE and collect frames and off-band energy."""
    dt = T / steps
    x = x0.copy()
    frames = []
    energies = []
    snapshot_times = np.linspace(0, steps, 8, dtype=int)

    for i in range(steps + 1):
        freq = np.fft.fft2(x)
        energies.append(np.linalg.norm(mask * freq))
        if i in snapshot_times:
            frames.append(np.clip(x, 0, 1))  # clip for visualization only

        if i == steps:
            break

        t = i * dt
        beta = linear_beta(t)
        noise = np.random.randn(*x.shape)
        # VP forward: dx = -0.5 * beta * x dt + sqrt(beta) dW
        x = x + (-0.5 * beta * x) * dt + np.sqrt(beta * dt) * noise

    return np.array(frames), np.array(energies)


# ---------------------------- Reverse process ----------------------------- #

def reverse_probability_flow(mask: np.ndarray, shape=(64, 64), steps: int = 200,
                              T: float = 1.0, sigma: float = 0.1):
    """Run the reverse probability-flow ODE using Heun's method."""
    dt = -T / steps  # integrate from T down to 0
    x = np.random.randn(*shape)
    frames = []
    energies = []
    snapshot_times = np.linspace(0, steps, 8, dtype=int)

    for i in range(steps + 1):
        t = T + i * dt  # decreasing time
        freq = np.fft.fft2(x)
        energies.append(np.linalg.norm(mask * freq))
        if i in snapshot_times:
            frames.append(np.clip(x, 0, 1))

        if i == steps:
            break

        beta = linear_beta(max(t, 0.0))
        # Optionally inflate sigma slightly at large t to mimic p_t smoothing.
        sigma_t = sigma * (1.0 + 0.5 * t)

        def drift(state):
            sc = spectral_score(state, mask, sigma_t)
            return -0.5 * beta * sc

        # Heun predictor-corrector
        k1 = drift(x)
        x_pred = x + dt * k1
        k2 = drift(x_pred)
        x = x + dt * 0.5 * (k1 + k2)

    return np.array(frames), np.array(energies), x


# ----------------------------- Visualization ------------------------------ #

def save_gif(frames: np.ndarray, filename: str, duration: float = 0.2):
    """Save a sequence of frames to a GIF."""
    images = (frames * 255).astype(np.uint8)
    imageio.mimsave(filename, images, duration=duration)


def plot_energy(forward_energy, reverse_energy, filename: str = "spectral_energy.png"):
    steps_f = len(forward_energy) - 1
    steps_r = len(reverse_energy) - 1
    t_forward = np.linspace(0, 1, steps_f + 1)
    t_reverse = np.linspace(1, 0, steps_r + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(t_forward, forward_energy, label="Forward off-band energy")
    plt.plot(t_reverse, reverse_energy, label="Reverse off-band energy")
    plt.xlabel("Time t")
    plt.ylabel(r"$\|M \circ F(x_t)\|_2$")
    plt.title("Spectral off-band energy over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ------------------------------- Main script ------------------------------ #

def main():
    size = 64
    tiles = 8
    steps = 200
    T = 1.0
    sigma = 0.1

    checker = make_checkerboard(size=size, tiles=tiles)
    mask = make_frequency_mask(checker, radius=1, thresh=0.5)

    # Forward diffusion
    f_frames, f_energy = forward_vp(checker, mask, steps=steps, T=T, sigma=sigma)
    save_gif(f_frames, "checker_forward.gif")

    # Reverse probability-flow ODE with spectral score
    r_frames, r_energy, x0 = reverse_probability_flow(mask, shape=checker.shape,
                                                      steps=steps, T=T, sigma=sigma)
    save_gif(r_frames, "checker_reverse_spectral.gif")

    # Plot energy curves for diagnostics
    plot_energy(f_energy, r_energy)

    # Show quick summary of the recovered sample energy at t=0
    print("Final reverse off-band energy:", r_energy[-1])
    print("Final sample min/max:", x0.min(), x0.max())


if __name__ == "__main__":
    main()
