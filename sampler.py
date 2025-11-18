import numpy as np
from typing import Callable, Optional


def reverse_sde_sampler(
    rng: np.random.Generator,
    score_fn: Callable[[np.ndarray, int], np.ndarray],
    schedule,
    n_particles: int,
    callback: Optional[Callable[[np.ndarray, int], None]] = None,
):
    x = rng.standard_normal((n_particles, 2))
    total_steps = schedule.t.shape[0] - 1
    for idx in range(total_steps, 0, -1):
        dt = schedule.t[idx] - schedule.t[idx - 1]
        beta_t = schedule.beta[idx]
        drift = -0.5 * beta_t * x - beta_t * score_fn(x, idx)
        noise = rng.standard_normal(x.shape)
        diffusion = np.sqrt(beta_t * dt) * noise
        x = x + drift * dt + diffusion
        if callback:
            callback(x, idx)
    if callback:
        callback(x, 0)
    return x


def probability_flow_ode(
    score_fn: Callable[[np.ndarray, int], np.ndarray],
    schedule,
    n_particles: int,
    rng: np.random.Generator,
    callback: Optional[Callable[[np.ndarray, int], None]] = None,
):
    x = rng.standard_normal((n_particles, 2))

    def ode_func(state, idx):
        beta_t = schedule.beta[idx]
        return -0.5 * beta_t * state - 0.5 * beta_t * score_fn(state, idx)

    total_steps = schedule.t.shape[0] - 1
    for idx in range(total_steps, 0, -1):
        dt = schedule.t[idx] - schedule.t[idx - 1]
        k1 = ode_func(x, idx)
        k2 = ode_func(x + 0.5 * dt * k1, idx)
        k3 = ode_func(x + 0.5 * dt * k2, idx)
        k4 = ode_func(x + dt * k3, idx - 1)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if callback:
            callback(x, idx)
    if callback:
        callback(x, 0)
    return x
