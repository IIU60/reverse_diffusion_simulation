from dataclasses import dataclass
import numpy as np


@dataclass
class GMM:
    pi: np.ndarray  # (K,)
    mu: np.ndarray  # (K, 2)
    Sig: np.ndarray  # (K, 2, 2)


def circle(K: int = 24, r: float = 2.0, sigma0: float = 0.12, jitter: float = 0.0) -> GMM:
    """Uniformly spaced Gaussian mixture components on a circle."""
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    base_mu = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    if jitter > 0:
        rng = np.random.default_rng()
        base_mu = base_mu + rng.normal(scale=jitter, size=base_mu.shape)

    pi = np.full(K, 1.0 / K)
    Sig = np.tile(np.eye(2) * (sigma0 ** 2), (K, 1, 1))
    return GMM(pi=pi, mu=base_mu, Sig=Sig)


def letter_S(K: int = 48, width: float = 4.0, sigma0: float = 0.12) -> GMM:
    """Gaussian mixture along a stylized letter 'S'."""
    if K < 6:
        raise ValueError("K should be at least 6 to form an 'S' shape")

    # Allocate components across segments
    n_top = max(2, K // 3)
    n_bridge = max(2, K // 4)
    n_bottom = K - n_top - n_bridge
    if n_bottom < 2:
        n_bottom = 2
        n_bridge = max(2, K - n_top - n_bottom)
    if n_top + n_bridge + n_bottom != K:
        n_top = K - n_bridge - n_bottom

    r = width / 2.0
    top_center = np.array([-r / 2.0, r / 2.0])
    bottom_center = np.array([r / 2.0, -r / 2.0])

    top_angles = np.linspace(3 * np.pi / 4, -np.pi / 4, n_top, endpoint=False)
    bottom_angles = np.linspace(5 * np.pi / 4, np.pi / 4, n_bottom)

    top_arc = top_center + r * np.stack([np.cos(top_angles), np.sin(top_angles)], axis=1)
    bottom_arc = bottom_center + r * np.stack([np.cos(bottom_angles), np.sin(bottom_angles)], axis=1)

    bridge_start = top_arc[-1]
    bridge_end = bottom_arc[0]
    bridge = np.linspace(bridge_start, bridge_end, n_bridge, endpoint=False)

    mu = np.concatenate([top_arc, bridge, bottom_arc], axis=0)
    pi = np.full(mu.shape[0], 1.0 / mu.shape[0])
    Sig = np.tile(np.eye(2) * (sigma0 ** 2), (mu.shape[0], 1, 1))
    return GMM(pi=pi, mu=mu, Sig=Sig)


def get_shape(name: str, **kwargs) -> GMM:
    name = name.lower()
    if name == "circle":
        return circle(**kwargs)
    if name in {"s", "letter_s"}:
        return letter_S(**kwargs)
    raise ValueError(f"Unknown shape '{name}'")


if __name__ == "__main__":
    g_circle = circle()
    print("circle K:", len(g_circle.pi))
    print("circle mu[:3]:", g_circle.mu[:3])
    assert g_circle.pi.shape == (len(g_circle.mu),)
    assert g_circle.Sig.shape == (len(g_circle.mu), 2, 2)

    g_s = letter_S()
    print("letter S K:", len(g_s.pi))
    print("letter S mu[:3]:", g_s.mu[:3])
    assert g_s.pi.shape == (len(g_s.mu),)
    assert g_s.Sig.shape == (len(g_s.mu), 2, 2)
