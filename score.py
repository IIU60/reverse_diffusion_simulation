import numpy as np
from scipy.special import logsumexp

from shapes import GMM


def score_gmm_time_marginal(x: np.ndarray, gmm: GMM, alpha_t: float, sigma_t: float, clip: float = 50.0) -> np.ndarray:
    """Compute the time-marginal score of a Gaussian mixture model.

    The forward VP dynamics yield a time-marginal mixture with component means
    ``alpha_t * mu_k`` and covariances ``Sigma_{k,t} = alpha_t**2 * Sigma_k + sigma_t**2 * I``.
    The score is the responsibility-weighted sum of per-component natural
    gradients ``Sigma_{k,t}^{-1} (alpha_t * mu_k - x)``.
    """

    if x.ndim != 2:
        raise ValueError("x must be (N, D)")

    N, D = x.shape
    K = gmm.mu.shape[0]

    Sig_t = alpha_t**2 * gmm.Sig + (sigma_t**2) * np.eye(D)[None, :, :]
    inv_Sig_t = np.linalg.inv(Sig_t)
    log_det = np.log(np.linalg.det(Sig_t))

    diffs = x[:, None, :] - (alpha_t * gmm.mu[None, :, :])
    maha = np.einsum("nkd,kde,nke->nk", diffs, inv_Sig_t, diffs)

    log_pi = np.log(gmm.pi + 1e-12)
    log_probs = log_pi - 0.5 * (log_det + maha + D * np.log(2 * np.pi))

    log_norm = logsumexp(log_probs, axis=1, keepdims=True)
    gamma = np.exp(log_probs - log_norm)

    grad = np.einsum("nk,kde,nke->nd", gamma, inv_Sig_t, (alpha_t * gmm.mu[None, :, :] - x[:, None, :]))
    grad = np.clip(grad, -clip, clip)
    return grad


def make_score_fn(gmm: GMM, schedule):
    def score_fn(x: np.ndarray, idx: int) -> np.ndarray:
        return score_gmm_time_marginal(
            x,
            gmm=gmm,
            alpha_t=float(schedule["alpha"][idx]),
            sigma_t=float(schedule["sigma"][idx]),
        )

    return score_fn


def _log_pdf(x: np.ndarray, gmm: GMM, alpha_t: float, sigma_t: float) -> float:
    """Log pdf of the time-marginal mixture at a single point x."""

    D = x.shape[-1]
    Sig_t = alpha_t**2 * gmm.Sig + (sigma_t**2) * np.eye(D)[None, :, :]
    inv_Sig_t = np.linalg.inv(Sig_t)
    log_det = np.log(np.linalg.det(Sig_t))

    diffs = x[None, :] - (alpha_t * gmm.mu)
    maha = np.einsum("kd,kde,ke->k", diffs, inv_Sig_t, diffs)

    log_pi = np.log(gmm.pi + 1e-12)
    log_probs = log_pi - 0.5 * (log_det + maha + D * np.log(2 * np.pi))
    return float(logsumexp(log_probs))


def _finite_difference_grad(x: np.ndarray, gmm: GMM, alpha_t: float, sigma_t: float, eps: float = 1e-4) -> np.ndarray:
    grad = np.zeros_like(x)
    for d in range(x.shape[-1]):
        e = np.zeros_like(x)
        e[d] = eps
        grad[d] = (_log_pdf(x + e, gmm, alpha_t, sigma_t) - _log_pdf(x - e, gmm, alpha_t, sigma_t)) / (2 * eps)
    return grad


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    from shapes import circle
    from schedule import make_linear_schedule

    gmm = circle(K=6, r=1.5, sigma0=0.2)
    schedule = make_linear_schedule(steps=10, beta_min=0.1, beta_max=20.0)
    x = rng.normal(size=(1, 2))
    idx = 3

    score = score_gmm_time_marginal(x, gmm, schedule["alpha"][idx], schedule["sigma"][idx])
    fd_grad = _finite_difference_grad(x[0], gmm, schedule["alpha"][idx], schedule["sigma"][idx])
    max_diff = np.max(np.abs(score[0] - fd_grad))
    print("score:", score)
    print("finite-difference grad:", fd_grad)
    print("max abs diff:", max_diff)
    assert max_diff < 1e-2, "Score and finite-difference gradient mismatch"
