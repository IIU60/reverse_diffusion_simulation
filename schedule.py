import numpy as np


def validate(schedule: dict):
    alpha = schedule["alpha"]
    sigma = schedule["sigma"]
    identity_error = np.max(np.abs((1 - alpha**2) - sigma**2))
    if not np.all(np.diff(alpha) >= 0.0):
        raise ValueError("alpha must be non-decreasing over time")
    if not np.all(np.diff(sigma) <= 0.0):
        raise ValueError("sigma must be non-increasing over time")
    if identity_error >= 1e-6:
        raise ValueError("(1 - alpha^2) and sigma^2 deviate beyond tolerance")


def make_linear_schedule(steps: int, beta_min: float, beta_max: float):
    # forward time grid u in [0, 1]
    u = np.linspace(0.0, 1.0, steps + 1)
    beta_u = beta_min + u * (beta_max - beta_min)
    dt = 1.0 / steps

    alpha = np.ones_like(u)
    for i in range(1, steps + 1):
        alpha[i] = alpha[i - 1] * np.exp(-0.5 * beta_u[i - 1] * dt)
    sigma = np.sqrt(1.0 - alpha**2)

    # reverse for backward integration (t from 1 -> 0)
    schedule = {
        "t": u[::-1],
        "beta": beta_u[::-1],
        "alpha": alpha[::-1],
        "sigma": sigma[::-1],
        "dt": -dt,
    }

    validate(schedule)
    return schedule


if __name__ == "__main__":
    cfg = make_linear_schedule(steps=10, beta_min=0.1, beta_max=20.0)
    print("alpha[:3]", cfg["alpha"][:3])
    print("sigma[:3]", cfg["sigma"][:3])
    print("abs((1-alpha^2)-sigma^2).max()", np.abs((1 - cfg["alpha"] ** 2) - cfg["sigma"] ** 2).max())
