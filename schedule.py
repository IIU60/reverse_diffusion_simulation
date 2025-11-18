import numpy as np
from dataclasses import dataclass


@dataclass
class Schedule:
    t: np.ndarray
    beta: np.ndarray
    alpha: np.ndarray
    sigma: np.ndarray
    dt: float


def build_schedule(steps: int, beta_min: float, beta_max: float, total_time: float = 1.0) -> Schedule:
    t = np.linspace(0.0, total_time, steps + 1)
    beta = beta_min + (beta_max - beta_min) * (t / total_time)
    dt = t[1] - t[0]

    alpha = np.ones_like(t)
    for i in range(1, steps + 1):
        beta_mid = 0.5 * (beta[i] + beta[i - 1])
        alpha[i] = alpha[i - 1] * np.exp(-0.5 * beta_mid * dt)
    sigma = np.sqrt(1.0 - alpha**2)

    return Schedule(t=t, beta=beta, alpha=alpha, sigma=sigma, dt=dt)
