"""
Demonstration of forward diffusion and reverse probability-flow ODE on
small grayscale circle images using a simple KDE score estimator.

The implementation is intentionally compact and heavily commented to
highlight the mechanics of the process without any deep-learning
frameworks (NumPy only).
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import imageio

# -------------------------- random seed --------------------------
# Fix the RNG for reproducibility so that the generated bank and
# trajectories are consistent between runs.
rng = np.random.default_rng(0)


# -------------------------- circle utils -------------------------
def make_circle(center_x: float, center_y: float, radius: float, thickness: float, H: int, W: int) -> np.ndarray:
    """Render a soft-edged circle ring on an HxW canvas.

    Pixels are anti-aliased by comparing distance to the desired radius
    with a tolerance proportional to the thickness.
    """
    # Build a coordinate grid in normalized pixel units.
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # Distance from each pixel center to the requested circle center.
    dist = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)

    # Compute a soft mask around the target radius using a linear ramp.
    inner = radius - thickness / 2.0
    outer = radius + thickness / 2.0
    # Negative values inside the inner radius, positive outside the outer.
    signed = np.maximum(dist - outer, inner - dist)
    # Map to [0, 1] using a smoothstep-like ramp.
    tol = thickness * 0.5 + 1.0  # add 1px slack to soften aliasing
    mask = 0.5 - 0.5 * np.clip(signed / tol, -1.0, 1.0)
    return mask.astype(np.float64)


def build_bank(N: int, H: int, W: int) -> np.ndarray:
    """Create a small bank of varied circle images flattened to vectors."""
    images = []
    for _ in range(N):
        # Sample center, radius, and thickness within modest ranges so
        # that shapes stay inside the 32x32 canvas.
        cx = rng.uniform(10, 22)
        cy = rng.uniform(10, 22)
        radius = rng.uniform(6, 10)
        thickness = rng.uniform(1.5, 3.5)
        img = make_circle(cx, cy, radius, thickness, H, W)
        images.append(img.reshape(-1))
    return np.stack(images, axis=0)


# ------------------------- beta schedule -------------------------
beta_min, beta_max, T = 0.1, 10.0, 1.0
N_steps = 200

dt = T / N_steps


def beta_of_t(t: float) -> float:
    """Linear schedule between beta_min and beta_max over [0, T]."""
    return beta_min + (beta_max - beta_min) * (t / T)


# Precompute discrete betas, alphas, and cumulative alpha_bar values for convenience.
times = np.linspace(0.0, T, N_steps, endpoint=False)
betas = beta_of_t(times)
alphas = np.exp(-0.5 * betas * dt)
# alpha_bar[t] = prod_{s<=t} alpha_s
alpha_bars = np.cumprod(alphas)


# --------------------------- KDE score ---------------------------
sigma0 = 0.05


def kde_score(x: np.ndarray, bank: np.ndarray, sigma_t: float):
    """Compute KDE log-density score and log normalizer.

    Args:
        x: Flattened image vector (D,).
        bank: Reference images (N, D).
        sigma_t: Bandwidth at current time.

    Returns:
        score: Gradient of log p(x) estimate.
        log_sum_w: Log of the weight sum (energy proxy, up to a constant).
    """
    sigma2 = float(sigma_t ** 2)
    diff = bank - x[None, :]
    dist2 = np.sum(diff ** 2, axis=1)
    # Unnormalized weights.
    w = np.exp(-dist2 / (2.0 * sigma2))
    denom = np.sum(w) + 1e-12  # guard against zero
    score = (w[:, None] * diff).sum(axis=0) / (sigma2 * denom)
    log_sum_w = np.log(denom)
    return score, log_sum_w


# --------------------------- forward SDE -------------------------
H = W = 32
bank_size = 100
bank = build_bank(bank_size, H, W)

# Pick a clean sample from the bank to diffuse forward.
x0 = bank[0].copy()

forward_frames = []
frame_steps = np.linspace(0, N_steps - 1, 8, dtype=int)

x_forward = x0.copy()
for step, alpha in enumerate(alphas):
    if step in frame_steps:
        forward_frames.append(x_forward.reshape(H, W))
    # Draw fresh noise and diffuse one step using the VP SDE discretization.
    eps = rng.normal(size=x_forward.shape)
    x_forward = alpha * x_forward + np.sqrt(1.0 - alpha ** 2) * eps
# Ensure final frame is captured.
forward_frames.append(x_forward.reshape(H, W))

# Save forward GIF.
forward_uint8 = [(np.clip(frame, 0, 1) * 255).astype(np.uint8) for frame in forward_frames]
imageio.mimsave("circle_forward.gif", forward_uint8, duration=0.5)


# --------------------------- reverse ODE -------------------------
# Start from pure Gaussian noise at time T.
x_rev = rng.normal(size=x0.shape)
reverse_frames = []
reverse_steps = set(np.linspace(N_steps, 0, 8, dtype=int))
energy_trace = []

start_time = time.time()
for idx in reversed(range(N_steps)):
    t = (idx + 1) * dt  # current time (descending)
    beta_t = beta_of_t(t)
    sigma_t = np.sqrt(sigma0 ** 2 + (1.0 / alpha_bars[idx] - 1.0))

    # Current score and energy.
    score_t, log_w = kde_score(x_rev, bank, sigma_t)
    energy_trace.append(-log_w)

    # Probability-flow ODE: dx/dt = -0.5 * beta(t) * score_t(x)
    dx_dt = -0.5 * beta_t * score_t

    # Predictor step (Euler) with negative dt to integrate backward.
    x_pred = x_rev + (-dt) * dx_dt

    # Evaluate at the predicted point and earlier time for Heun correction.
    t_prev = idx * dt
    beta_prev = beta_of_t(t_prev)
    if idx > 0:
        sigma_prev = np.sqrt(sigma0 ** 2 + (1.0 / alpha_bars[idx - 1] - 1.0))
    else:
        sigma_prev = sigma0
    score_pred, _ = kde_score(x_pred, bank, sigma_prev)
    dx_dt_pred = -0.5 * beta_prev * score_pred

    # Corrector: average the two slopes.
    x_rev = x_rev + (-dt) * 0.5 * (dx_dt + dx_dt_pred)

    if idx in reverse_steps:
        reverse_frames.append(x_rev.reshape(H, W))

elapsed = time.time() - start_time

# Reverse frames are collected from T->0; flip to chronological order.
reverse_frames = list(reversed(reverse_frames))
reverse_uint8 = [(np.clip(frame, 0, 1) * 255).astype(np.uint8) for frame in reverse_frames]
imageio.mimsave("circle_reverse_kde.gif", reverse_uint8, duration=0.5)


# --------------------------- plotting ----------------------------
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i, frame in enumerate(forward_frames[:8]):
    ax = axes[0, i]
    ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(f"t={frame_steps[i]*dt:.2f}")

for i, frame in enumerate(reverse_frames[:8]):
    ax = axes[1, i]
    ax.imshow(frame, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(f"t={(1 - i/7):.2f}")

axes[0, 0].set_ylabel("Forward", fontsize=10)
axes[1, 0].set_ylabel("Reverse", fontsize=10)
plt.tight_layout()

# Energy proxy plot.
plt.figure(figsize=(4, 3))
plt.plot(np.linspace(T, 0, len(energy_trace)), energy_trace)
plt.xlabel("Time t")
plt.ylabel("Energy proxy E(x)")
plt.title("KDE energy during reverse ODE")
plt.tight_layout()

print("Forward frames saved to circle_forward.gif")
print("Reverse frames saved to circle_reverse_kde.gif")
print(f"Reverse ODE integration time: {elapsed:.3f} s for {N_steps} steps")
plt.show()
