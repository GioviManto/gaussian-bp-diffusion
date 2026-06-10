"""
local_bp.py
===========

Local (truncated-range) message passing on the AR(1) + OU chain, to compare
with the full forward+backward belief propagation sweeps.

Question (from the meeting): what is the difference between running only
*local* messages -- a node listens to its immediate neighbourhood and ignores
the rest of the chain -- versus the full BP that propagates information all the
way along the chain through forward and backward sweeps?

Formalisation.  The full BP posterior mean is

    E[a_k | x_0, ..., x_{K-1}]   (uses every frame).

The *local estimator of radius r* uses only the observations within distance r,

    E[a_k | x_{k-r}, ..., x_{k+r}]   (the rest of the chain marginalised out).

Because any contiguous block of a stationary AR(1) chain is itself a stationary
AR(1) chain, this conditional expectation is computed exactly on the windowed
sub-chain -- no extra approximation beyond the truncation of range.  Taking
r >= K-1 makes the window the whole chain and recovers the exact full BP.

In the Gaussian case the full BP is exact, so the gap

    || S_local^{(r)}(x, t) - S_full(x, t) ||

measures purely the error introduced by locality.  It shrinks as r grows and as
the diffusion time t weakens the inter-frame coupling.
"""

from __future__ import annotations

import numpy as np

from ar1_utils import ar1_covariance, ou_params


def local_posterior_mean(x: np.ndarray, t: float, alpha: float, radius: int,
                         sigma_eta: float | None = None) -> np.ndarray:
    """Return the radius-r local posterior mean E[a_k | x_{k-r..k+r}] for all k.

    radius >= K-1 reproduces the exact full-chain posterior mean.
    radius = 0 uses only the single local observation x_k.
    """
    x = np.asarray(x, dtype=float)
    K = x.shape[0]
    mu, Delta = ou_params(t)
    ev_prec = mu * mu / Delta          # evidence precision in 'a' coordinates
    ev_gain = mu / Delta               # maps x_j to the field h_j

    m = np.empty(K)
    for k in range(K):
        lo = max(0, k - radius)
        hi = min(K - 1, k + radius)
        idx = np.arange(lo, hi + 1)
        k0 = k - lo                    # position of k inside the window
        n = len(idx)

        Sigma_sub = ar1_covariance(n, alpha, sigma_eta)
        J = ev_prec * np.eye(n) + np.linalg.inv(Sigma_sub)
        h = ev_gain * x[idx]
        m_sub = np.linalg.solve(J, h)
        m[k] = m_sub[k0]
    return m


def local_score(x: np.ndarray, t: float, alpha: float, radius: int,
                sigma_eta: float | None = None) -> np.ndarray:
    """Joint score from the radius-r local posterior mean via Tweedie."""
    mu, Delta = ou_params(t)
    m = local_posterior_mean(x, t, alpha, radius, sigma_eta)
    return (mu * m - np.asarray(x, dtype=float)) / Delta


# ---------------------------------------------------------------------------
# Exact (deterministic) RMS truncation error
# ---------------------------------------------------------------------------

def _smoother_matrix(K: int, alpha: float, t: float,
                     sigma_eta: float | None = None) -> np.ndarray:
    """The full-chain smoother C with E[a | x] = C x  (linear, Gaussian)."""
    from ar1_utils import ar1_precision_clean
    mu, Delta = ou_params(t)
    J = (mu * mu / Delta) * np.eye(K) + ar1_precision_clean(K, alpha, sigma_eta)
    return np.linalg.solve(J, (mu / Delta) * np.eye(K))


def rms_truncation_error(K: int, alpha: float, t: float, k: int, radius: int,
                         sigma_eta: float | None = None) -> float:
    """Exact RMS error of the radius-r local estimator at frame k:

        RMS_r(k) = sqrt( E_{x ~ P_t} [ ( E[a_k|x] - E[a_k|x_window] )^2 ] ).

    Both estimators are LINEAR in x, so the error is e . x for a deterministic
    row vector e, and the RMS over x ~ N(0, Sigma_t) is sqrt(e' Sigma_t e):
    a closed-form deterministic quantity, no sampling involved.

    The audit verifies that log RMS_r decays in r with slope log q, where
    q = chain_formulas.bulk_correlation_decay(alpha, t): the locality error
    inherits exactly the posterior correlation decay rate.
    """
    mu, Delta = ou_params(t)
    Sigma_0 = ar1_covariance(K, alpha, sigma_eta)
    Sigma_t = mu * mu * Sigma_0 + Delta * np.eye(K)

    e = _smoother_matrix(K, alpha, t, sigma_eta)[k].copy()
    lo, hi = max(0, k - radius), min(K - 1, k + radius)
    Cw = _smoother_matrix(hi - lo + 1, alpha, t, sigma_eta)
    e[lo:hi + 1] -= Cw[k - lo]
    return float(np.sqrt(e @ Sigma_t @ e))


if __name__ == "__main__":
    from ar1_utils import ar1_covariance as _cov, joint_score_matrix

    rng = np.random.default_rng(3)
    K, alpha = 21, 0.8
    Sigma_0 = _cov(K, alpha)
    x = rng.standard_normal(K)

    print(f"AR(1) chain K={K}, alpha={alpha}: local radius-r score vs full BP")
    print("  (full BP == exact -Sigma_t^-1 x; error is purely from locality)\n")
    for t in (0.1, 0.5, 2.0):
        s_full = joint_score_matrix(x, t, Sigma_0, alpha)
        print(f"  t={t}")
        for r in (0, 1, 2, 4, 8, K - 1):
            s_loc = local_score(x, t, alpha, r)
            err = np.max(np.abs(s_loc - s_full))
            tag = "  (= full chain)" if r == K - 1 else ""
            print(f"     radius r={r:<2}  max|S_local - S_full| = {err:.2e}{tag}")
        print()
