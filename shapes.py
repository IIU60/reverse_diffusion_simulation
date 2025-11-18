import numpy as np


def build_circle_gmm(components: int = 8, radius: float = 2.0, spread: float = 0.07):
    angles = np.linspace(0, 2 * np.pi, components, endpoint=False)
    means = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    cov = np.eye(2) * spread
    covs = np.repeat(cov[None, :, :], components, axis=0)
    weights = np.ones(components) / components
    return weights, means, covs


def build_s_gmm(components: int = 10, width: float = 2.0, spread: float = 0.05):
    x = np.linspace(-2.5, 2.5, components)
    y = width * np.sin(x)
    means = np.stack([x, y], axis=1)
    cov = np.eye(2) * spread
    covs = np.repeat(cov[None, :, :], components, axis=0)
    weights = np.ones(components) / components
    return weights, means, covs


def get_shape(name: str):
    name = name.lower()
    if name == "circle":
        return build_circle_gmm()
    if name in {"s", "curve"}:
        return build_s_gmm()
    raise ValueError(f"Unknown shape '{name}'")
