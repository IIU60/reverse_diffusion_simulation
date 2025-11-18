import numpy as np


def _logsumexp(x: np.ndarray, axis: int = -1):
    m = np.max(x, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))


def gmm_time_marginal_score(
    x: np.ndarray,
    alpha: float,
    sigma: float,
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
) -> np.ndarray:
    # assumes isotropic covariances
    base_var = covs[:, 0, 0]
    mean_t = alpha * means  # (K, 2)
    var_t = alpha**2 * base_var + sigma**2  # (K,)

    diff = x[:, None, :] - mean_t[None, :, :]  # (N, K, 2)
    norm_sq = np.sum(diff**2, axis=2)

    log_weights = np.log(weights + 1e-12)
    log_norm = log_weights - np.log(2 * np.pi * var_t)
    log_probs = log_norm[None, :] - 0.5 * norm_sq / var_t

    log_denom = _logsumexp(log_probs, axis=1)  # (N, 1)
    resp = np.exp(log_probs - log_denom)  # (N, K)

    inv_var = 1.0 / var_t[None, :, None]
    score = np.sum(resp[:, :, None] * (-diff * inv_var), axis=1)
    return score


def make_score_fn(weights: np.ndarray, means: np.ndarray, covs: np.ndarray, schedule):
    def score_fn(x: np.ndarray, idx: int) -> np.ndarray:
        return gmm_time_marginal_score(
            x,
            alpha=float(schedule["alpha"][idx]),
            sigma=float(schedule["sigma"][idx]),
            weights=weights,
            means=means,
            covs=covs,
        )

    return score_fn
