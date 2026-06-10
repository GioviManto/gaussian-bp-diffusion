"""
bp_score.py
===========

Belief propagation on the Gaussian AR(1) chain with independent OU noise
on each frame.  Returns the posterior mean E[a_k | x] at every frame and
the joint score

    S_k(x, t) = (exp(-t) E[a_k | x] - x_k) / Delta_t

via Tweedie's identity. The construction is derived in main.pdf,
section 8 (Convention A: local evidence enters at combination time,
never inside a recursion); this implementation is validated by
numerical_audit.py against the matrix posterior. Cost is O(K).

The factor graph of the chain is a tree, the AR(1) prior is Gaussian
and the OU corruption is independent per frame, so all messages remain
Gaussian. Each message is therefore represented by two scalars (mean,
variance); a variance of +infty represents the uninformative initial
backward message and is handled exactly.

References:
    main.pdf sections 8-10 (Convention A, K=3 worked example, Kalman/RTS).
"""

from __future__ import annotations

import numpy as np

from ar1_utils import ou_params, stationary_sigma_eta


# ---------------------------------------------------------------------------
# Local evidence in 'a' coordinates
# ---------------------------------------------------------------------------

def _evidence_in_a(x_k: float, t: float) -> tuple[float, float]:
    """Return (mean, variance) of the evidence factor p_t(x_k | a_k)
    viewed as a Gaussian in a_k.

        p_t(x_k | a_k) propto N(a_k ; exp(t) x_k, exp(2 t) Delta_t).
    """
    mu, Delta = ou_params(t)
    inv_mu = 1.0 / mu  # exp(t)
    return inv_mu * x_k, inv_mu * inv_mu * Delta


# ---------------------------------------------------------------------------
# Gaussian product helper -- robust to v = +infty (uninformative)
# ---------------------------------------------------------------------------

def _gaussian_product(m1: float, v1: float, m2: float, v2: float
                      ) -> tuple[float, float]:
    """Return (m, v) for the Gaussian product N(m1,v1) * N(m2,v2)."""
    if np.isinf(v1):
        return m2, v2
    if np.isinf(v2):
        return m1, v1
    inv_v = 1.0 / v1 + 1.0 / v2
    v = 1.0 / inv_v
    m = v * (m1 / v1 + m2 / v2)
    return m, v


# ---------------------------------------------------------------------------
# Forward (Kalman filter) and backward (RTS smoother) recursions
# ---------------------------------------------------------------------------

def _forward_pass(x: np.ndarray, t: float, alpha: float, sigma_eta: float,
                  mu_0: float, sigma_0: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Convention A forward messages mu_->k(a_k) = N(a_k ; m_to[k], v_to[k]).

    The forward message at node k carries the evidence x_0, ..., x_{k-1}
    only; the local evidence at node k is inserted at combination time
    (so this code never double-counts).
    """
    K = len(x)
    m_to = np.empty(K)
    v_to = np.empty(K)
    m_to[0] = mu_0
    v_to[0] = sigma_0 * sigma_0

    for k in range(K - 1):
        # Combine forward message with local evidence at node k
        m_ev, v_ev = _evidence_in_a(x[k], t)
        m_c, v_c = _gaussian_product(m_to[k], v_to[k], m_ev, v_ev)
        # Predict to k+1 through the AR(1) transition
        m_to[k + 1] = alpha * m_c
        v_to[k + 1] = alpha * alpha * v_c + sigma_eta * sigma_eta
    return m_to, v_to


def _backward_pass(x: np.ndarray, t: float, alpha: float, sigma_eta: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Convention A backward messages mu_<-k(a_k) prop to N(m_lr[k], v_lr[k]).

    The backward message at node k carries x_{k+1}, ..., x_{K-1}; node K-1
    receives a flat (uninformative, v = +infty) initial message.
    """
    if np.isclose(alpha, 0.0):
        raise ValueError("Backward pass divides by alpha; choose |alpha| > 0.")
    K = len(x)
    m_lr = np.empty(K)
    v_lr = np.empty(K)
    m_lr[K - 1] = 0.0
    v_lr[K - 1] = np.inf

    for k in range(K - 1, 0, -1):
        # Combine backward message with local evidence at node k
        m_ev, v_ev = _evidence_in_a(x[k], t)
        m_c, v_c = _gaussian_product(m_lr[k], v_lr[k], m_ev, v_ev)
        # Back-propagate one AR(1) step: a_k = alpha a_{k-1} + eta_{k-1}
        m_lr[k - 1] = m_c / alpha
        v_lr[k - 1] = (sigma_eta * sigma_eta + v_c) / (alpha * alpha)
    return m_lr, v_lr


def bp_posterior(x: np.ndarray, t: float, alpha: float,
                 sigma_eta: float | None = None,
                 mu_0: float = 0.0, sigma_0: float | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu_post, var_post) for p(a_k | x), shape (K,).

    Combines forward, local evidence, and backward messages at every
    node. mu_post[k] equals E[a_k | x] (the Kalman-smoothed mean).
    """
    if sigma_eta is None:
        sigma_eta = stationary_sigma_eta(alpha)
    if sigma_0 is None:
        sigma_0 = sigma_eta / np.sqrt(1.0 - alpha * alpha)

    x = np.asarray(x, dtype=float)
    m_to, v_to = _forward_pass(x, t, alpha, sigma_eta, mu_0, sigma_0)
    m_lr, v_lr = _backward_pass(x, t, alpha, sigma_eta)

    K = len(x)
    mu_post = np.empty(K)
    var_post = np.empty(K)
    for k in range(K):
        m_ev, v_ev = _evidence_in_a(x[k], t)
        m_a, v_a = _gaussian_product(m_to[k], v_to[k], m_ev, v_ev)
        m_b, v_b = _gaussian_product(m_a, v_a, m_lr[k], v_lr[k])
        mu_post[k] = m_b
        var_post[k] = v_b
    return mu_post, var_post


def bp_score(x: np.ndarray, t: float, alpha: float,
             sigma_eta: float | None = None,
             mu_0: float = 0.0, sigma_0: float | None = None
             ) -> np.ndarray:
    """Return the joint score S(x, t) computed via belief propagation.

    Tweedie's identity:
        S_k(x, t) = (exp(-t) mu_post[k] - x_k) / Delta_t.
    """
    mu_post, _ = bp_posterior(x, t, alpha, sigma_eta, mu_0, sigma_0)
    mu, Delta = ou_params(t)
    return (mu * mu_post - np.asarray(x, dtype=float)) / Delta


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from ar1_utils import ar1_covariance, gaussian_posterior, joint_score_matrix

    rng = np.random.default_rng(1)
    K, alpha, t = 8, 0.7, 0.4
    Sigma_0 = ar1_covariance(K, alpha)
    x = rng.standard_normal(K)

    mu_bp, var_bp = bp_posterior(x, t, alpha)
    mu_mat, Sigma_mat = gaussian_posterior(x, t, Sigma_0, alpha)
    print(f"max |mu_bp - mu_matrix| = {np.max(np.abs(mu_bp - mu_mat)):.2e}")
    print(f"max |var_bp - diag(Sigma_matrix)| = "
          f"{np.max(np.abs(var_bp - np.diag(Sigma_mat))):.2e}")

    s_bp = bp_score(x, t, alpha)
    s_matrix = joint_score_matrix(x, t, Sigma_0, alpha)
    print(f"max |S_bp - S_matrix| = {np.max(np.abs(s_bp - s_matrix)):.2e}")
