"""
ar1_utils.py
============

Gaussian AR(1) benchmark utilities.

Implements the closed-form objects that govern the joint score
    S(x, t) = grad_x log P_t(x)
of a stationary Gaussian AR(1) chain corrupted by a coordinate-wise
Ornstein-Uhlenbeck (OU) channel:

    a_{u+1} = alpha * a_u + eta_u,   eta_u ~ N(0, sigma_eta^2)
    a_0     ~ N(mu_0, sigma_0^2)
    x_u     = exp(-t) a_u + sqrt(Delta_t) xi_u,    xi_u ~ N(0, 1)
    Delta_t = 1 - exp(-2 t)

For stationarity we set sigma_0^2 = sigma_eta^2 / (1 - alpha^2).

For Gaussian P_0 the joint score is exactly affine in x:
    S(x, t) = - Sigma_t^{-1} (x - exp(-t) mu_a)
where mu_a is the prior mean vector mu_a[k] = alpha^k mu_0.

References (Established):
    [G1] Gaussian_session_summary.pdf, sections 2.2-2.4
    [P1] precision_lifecycle_summary.pdf, sections 2.1-2.3, 3.1-3.3
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Forward-channel parameters
# ---------------------------------------------------------------------------

def ou_params(t: float) -> tuple[float, float]:
    """Return (mu_factor, Delta_t) for the OU channel at time t.

    mu_factor = exp(-t) is the signal-attenuation coefficient, so that
    x = mu_factor * a + sqrt(Delta_t) * xi.
    """
    if t < 0:
        raise ValueError("OU diffusion time must be non-negative.")
    mu = np.exp(-t)
    Delta = 1.0 - np.exp(-2.0 * t)
    return float(mu), float(Delta)


# ---------------------------------------------------------------------------
# Clean Gaussian AR(1) statistics
# ---------------------------------------------------------------------------

def stationary_sigma_eta(alpha: float) -> float:
    """Innovation std that yields unit stationary variance for AR(1).

    Returns sqrt(1 - alpha^2) so that Var(a_u) = 1 at stationarity.
    """
    if not -1.0 < alpha < 1.0:
        raise ValueError("AR(1) coefficient |alpha| must be < 1.")
    return float(np.sqrt(1.0 - alpha * alpha))


def ar1_covariance(K: int, alpha: float, sigma_eta: float | None = None,
                   sigma_0: float | None = None) -> np.ndarray:
    """Return the K x K covariance Sigma_0 of the AR(1) chain.

    If sigma_eta is None, it is set to sqrt(1 - alpha^2) (stationary).
    If sigma_0 is None it is set so that the chain starts at stationarity:
        sigma_0^2 = sigma_eta^2 / (1 - alpha^2).
    The resulting Sigma_0 is the Toeplitz matrix
        Sigma_0[i, j] = (sigma_eta^2 / (1 - alpha^2)) * alpha^|i-j|
    in the stationary case.
    """
    if K < 1:
        raise ValueError("K must be >= 1.")
    if sigma_eta is None:
        sigma_eta = stationary_sigma_eta(alpha)
    if sigma_0 is None:
        sigma_0 = sigma_eta / np.sqrt(1.0 - alpha * alpha)

    # Build Var(a_u) by unrolling the recursion. Stationary case gives a
    # constant column; non-stationary case uses the closed form
    #    Var(a_u) = alpha^{2u} sigma_0^2 + sigma_eta^2 (1 - alpha^{2u})/(1 - alpha^2).
    var_diag = np.empty(K)
    var_diag[0] = sigma_0 * sigma_0
    for u in range(1, K):
        var_diag[u] = (alpha * alpha) * var_diag[u - 1] + sigma_eta * sigma_eta

    Sigma_0 = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            Sigma_0[i, j] = (alpha ** abs(i - j)) * var_diag[min(i, j)]
    return Sigma_0


def ar1_precision_clean(K: int, alpha: float,
                        sigma_eta: float | None = None) -> np.ndarray:
    """Tridiagonal precision Q_0 of the stationary AR(1) chain.

    From the factorisation P_0(a) = p_0(a_0) prod_u p(a_{u+1} | a_u),
    the joint log-density is a sum of nearest-neighbour quadratics, so
    Q_0 is tridiagonal with
        diag interior = (1 + alpha^2) / sigma_eta^2
        diag boundary = 1 / sigma_eta^2
        off-diag      = - alpha / sigma_eta^2
    """
    if sigma_eta is None:
        sigma_eta = stationary_sigma_eta(alpha)
    s2 = sigma_eta * sigma_eta
    Q = np.zeros((K, K))
    for k in range(K):
        if k == 0 or k == K - 1:
            Q[k, k] = 1.0 / s2
        else:
            Q[k, k] = (1.0 + alpha * alpha) / s2
        if k + 1 < K:
            Q[k, k + 1] = -alpha / s2
            Q[k + 1, k] = -alpha / s2
    return Q


def prior_mean(K: int, alpha: float, mu_0: float = 0.0) -> np.ndarray:
    """Prior mean mu_a[k] = alpha^k * mu_0 for the AR(1) chain.

    For the stationary, zero-mean case this is identically zero.
    """
    return mu_0 * np.power(alpha, np.arange(K))


# ---------------------------------------------------------------------------
# Noisy covariance and precision under OU corruption
# ---------------------------------------------------------------------------

def sigma_t(Sigma_0: np.ndarray, t: float) -> np.ndarray:
    """Noisy covariance Sigma_t = exp(-2 t) Sigma_0 + (1 - exp(-2 t)) I."""
    _, Delta = ou_params(t)
    decay = 1.0 - Delta
    K = Sigma_0.shape[0]
    return decay * Sigma_0 + Delta * np.eye(K)


def precision_t(Sigma_0: np.ndarray, t: float) -> np.ndarray:
    """Noisy precision Q_t = Sigma_t^{-1} computed by direct inversion."""
    return np.linalg.inv(sigma_t(Sigma_0, t))


def precision_t_spectral(Sigma_0: np.ndarray, t: float
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Q_t via the eigenbasis of Sigma_0.

    Sigma_0 = U diag(lam_0) U^T, then
        Sigma_t = U diag(exp(-2 t) lam_0 + Delta_t) U^T,
        Q_t     = U diag(1 / (exp(-2 t) lam_0 + Delta_t)) U^T.
    Returns (Q_t, U, lam_t) where lam_t are the noisy eigenvalues.
    """
    lam_0, U = np.linalg.eigh(Sigma_0)
    mu, Delta = ou_params(t)
    lam_t = (mu * mu) * lam_0 + Delta
    Q_t = (U * (1.0 / lam_t)) @ U.T
    return Q_t, U, lam_t


# ---------------------------------------------------------------------------
# Joint score
# ---------------------------------------------------------------------------

def joint_score_matrix(x: np.ndarray, t: float, Sigma_0: np.ndarray,
                       alpha: float, mu_0: float = 0.0) -> np.ndarray:
    """Return S(x, t) = - Sigma_t^{-1} (x - exp(-t) mu_a).

    This is the exact joint score for the Gaussian AR(1) chain at OU
    diffusion time t. Linear in x.
    """
    K = Sigma_0.shape[0]
    if x.shape[-1] != K:
        raise ValueError("x has incompatible last dimension with Sigma_0.")
    mu, _ = ou_params(t)
    mu_a = prior_mean(K, alpha, mu_0)
    Q_t = precision_t(Sigma_0, t)
    return -(x - mu * mu_a) @ Q_t  # works for x of shape (...,K)


# ---------------------------------------------------------------------------
# Posterior mean E[a | x] via the matrix form (used by the audit)
# ---------------------------------------------------------------------------

def gaussian_posterior(x: np.ndarray, t: float, Sigma_0: np.ndarray,
                       alpha: float, mu_0: float = 0.0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu_post, Sigma_post) for p(a | x) under Gaussian AR(1) + OU.

    The posterior is Gaussian with
        Sigma_post^{-1} = (exp(-2 t) / Delta_t) I + Sigma_0^{-1}
        mu_post         = Sigma_post * ( (exp(-t) / Delta_t) x + Sigma_0^{-1} mu_a )
    Used to cross-check the BP recursion in numerical_audit.py.
    """
    K = Sigma_0.shape[0]
    mu, Delta = ou_params(t)
    Sigma_0_inv = np.linalg.inv(Sigma_0)
    prec_post = (mu * mu / Delta) * np.eye(K) + Sigma_0_inv
    Sigma_post = np.linalg.inv(prec_post)
    mu_a = prior_mean(K, alpha, mu_0)
    mu_post = Sigma_post @ ((mu / Delta) * x + Sigma_0_inv @ mu_a)
    return mu_post, Sigma_post


def joint_score_via_tweedie(x: np.ndarray, t: float, Sigma_0: np.ndarray,
                            alpha: float, mu_0: float = 0.0) -> np.ndarray:
    """Return S(x, t) computed through the Tweedie identity

        S_k(x, t) = (exp(-t) E[a_k | x] - x_k) / Delta_t

    using the exact Gaussian posterior mean. Equivalent to
    joint_score_matrix() up to floating-point arithmetic; checking the
    two forms agree is part of the numerical audit.
    """
    mu, Delta = ou_params(t)
    mu_post, _ = gaussian_posterior(x, t, Sigma_0, alpha, mu_0)
    return (mu * mu_post - x) / Delta


# ---------------------------------------------------------------------------
# Self-test when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    K = 8
    alpha = 0.7
    Sigma_0 = ar1_covariance(K, alpha)
    Q_0 = ar1_precision_clean(K, alpha)

    # Tridiagonality at t = 0
    err = np.linalg.norm(Q_0 - np.linalg.inv(Sigma_0))
    print(f"||Q_0 - Sigma_0^-1|| = {err:.2e}")

    # Spectral preservation
    t = 0.4
    Q_a = precision_t(Sigma_0, t)
    Q_b, _, _ = precision_t_spectral(Sigma_0, t)
    print(f"||Q_t direct - Q_t spectral|| = {np.linalg.norm(Q_a - Q_b):.2e}")

    # Score consistency
    x = rng.standard_normal(K)
    s1 = joint_score_matrix(x, t, Sigma_0, alpha)
    s2 = joint_score_via_tweedie(x, t, Sigma_0, alpha)
    print(f"||S_matrix - S_tweedie|| = {np.linalg.norm(s1 - s2):.2e}")
