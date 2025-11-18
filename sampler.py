"""Reverse-time solvers for the VP diffusion process."""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional

from score import score_gmm_time_marginal
from shapes import GMM


def step_reverse_sde(
    x: np.ndarray,
    beta_t: float,
    alpha_t: float,
    sigma_t: float,
    gmm: GMM,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Single Euler–Maruyama step of the reverse VP SDE."""

    s = score_gmm_time_marginal(x, gmm, alpha_t, sigma_t)
    drift = -0.5 * beta_t * x - beta_t * s
    noise = np.sqrt(beta_t) * np.sqrt(abs(dt)) * rng.standard_normal(x.shape)
    return x + drift * dt + noise


def integrate_reverse_sde(
    xT: np.ndarray,
    schedule: dict,
    gmm: GMM,
    steps: int | None = None,
    seed: int = 0,
    callback: Optional[Callable[[np.ndarray, int], None]] = None,
) -> np.ndarray:
    """Integrate the reverse SDE from terminal noise to data space."""

    rng = np.random.default_rng(seed)
    x = xT.copy()

    betas = schedule["beta"]
    alphas = schedule["alpha"]
    sigmas = schedule["sigma"]
    dt = float(schedule["dt"])

    total = len(betas) if steps is None else min(steps, len(betas))
    for i in range(total):
        x = step_reverse_sde(
            x,
            beta_t=float(betas[i]),
            alpha_t=float(alphas[i]),
            sigma_t=float(sigmas[i]),
            gmm=gmm,
            dt=dt,
            rng=rng,
        )
        if callback:
            callback(x, i)
    return x


def drift_reverse_ode(
    x: np.ndarray,
    beta_t: float,
    alpha_t: float,
    sigma_t: float,
    gmm: GMM,
) -> np.ndarray:
    """Deterministic reverse-time drift (probability-flow ODE)."""

    s = score_gmm_time_marginal(x, gmm, alpha_t, sigma_t)
    return -0.5 * beta_t * x - 0.5 * beta_t * s


def integrate_reverse_ode(
    xT: np.ndarray,
    schedule: dict,
    gmm: GMM,
    steps: int | None = None,
    callback: Optional[Callable[[np.ndarray, int], None]] = None,
) -> np.ndarray:
    """Integrate the probability-flow ODE with RK4."""

    x = xT.copy()

    betas = schedule["beta"]
    alphas = schedule["alpha"]
    sigmas = schedule["sigma"]
    dt = float(schedule["dt"])

    total = len(betas) if steps is None else min(steps, len(betas))
    for i in range(total):
        beta_t = float(betas[i])
        alpha_t = float(alphas[i])
        sigma_t = float(sigmas[i])

        k1 = drift_reverse_ode(x, beta_t, alpha_t, sigma_t, gmm)
        k2 = drift_reverse_ode(x + 0.5 * dt * k1, beta_t, alpha_t, sigma_t, gmm)
        k3 = drift_reverse_ode(x + 0.5 * dt * k2, beta_t, alpha_t, sigma_t, gmm)
        k4 = drift_reverse_ode(x + dt * k3, beta_t, alpha_t, sigma_t, gmm)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if callback:
            callback(x, i)
    return x


if __name__ == "__main__":
    from shapes import circle
    from schedule import make_linear_schedule

    rng = np.random.default_rng(0)
    gmm = circle()
    schedule = make_linear_schedule(steps=100, beta_min=0.1, beta_max=20.0)
    xT = rng.standard_normal((1_000, 2))

    x_sde = integrate_reverse_sde(xT, schedule, gmm, seed=0)
    x_ode = integrate_reverse_ode(xT, schedule, gmm)
    assert np.isfinite(x_sde).all()
    assert np.isfinite(x_ode).all()
    print("mean |score|", np.mean(np.linalg.norm(score_gmm_time_marginal(x_ode, gmm, schedule["alpha"][0], schedule["sigma"][0]), axis=1)))
