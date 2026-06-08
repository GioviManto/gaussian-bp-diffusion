"""
amp.py
======

Gaussian message passing on the posterior precision of the AR(1)+OU model,
comparing three approximations of the marginals of a Gaussian graphical model

    p(a | x)  propto  exp( -1/2 a^T J a + h^T a ),

where (for our model, stationary zero-mean prior)

    J = (exp(-2 t) / Delta_t) I + Sigma_0^{-1}    (posterior precision, tridiagonal)
    h = (exp(-t)  / Delta_t) x                    (posterior field).

The exact marginals are mean = J^{-1} h and variances = diag(J^{-1}).

The point of this module is to make precise, and verify numerically, the
statement discussed with J. Garnier-Brun:

  * The posterior MEAN solves the linear system J m = h.  Mean-field, belief
    propagation, and AMP are all iterative solvers of that *same* system, so
    they share the *exact* mean -- hence the *same score*.  (Confirms the
    guess "BP in the Gaussian case recovers the right solution".)

  * They differ only in the per-node VARIANCE estimate:
        - mean field : V_i = 1 / J_ii                         (ignores neighbours)
        - AMP / TAP  : V_i = 1 / (J_ii - sum_k J_ik^2 V_k)     (cavity, no exclusion)
        - BP / exact : V_i = (J^{-1})_ii                       (exact on a tree)
    The AMP closure drops the "exclude one neighbour" correction.  That is
    exact when each node has *many* weak neighbours (a CLT over the incoming
    messages -- the dense-graph regime AMP was built for), and only
    approximate on the AR(1) chain, where every node has just two neighbours.

The dense-graph regime is used as a correctness gate in __main__: on a dense
random SPD J the AMP variance must match diag(J^{-1}); on the chain it must not.

References:
    Mezard & Montanari, Information, Physics, and Computation (2009), TAP/cavity.
    Zdeborova & Krzakala, Statistical Physics for Optimization and Learning,
        EPFL Doctoral Lectures (2021), AMP / TAP equations.
    Genovese & Piana, arXiv:2602.15191 (2026): BP means -> AMP asymptotically
        on the dense (complete bipartite) factor graph.
"""

from __future__ import annotations

import numpy as np

from ar1_utils import ar1_covariance, ar1_precision_clean, ou_params, prior_mean


# ---------------------------------------------------------------------------
# Posterior precision J and field h for the AR(1) + OU model
# ---------------------------------------------------------------------------

def posterior_precision_field(x: np.ndarray, t: float, alpha: float,
                              sigma_eta: float | None = None,
                              mu_0: float = 0.0
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Return (J, h) of the Gaussian posterior p(a | x).

        J = (exp(-2t)/Delta_t) I + Sigma_0^{-1},   h = (exp(-t)/Delta_t) x + Sigma_0^{-1} mu_a.

    For the stationary zero-mean prior mu_a = 0 and h = (exp(-t)/Delta_t) x.
    J is tridiagonal because Sigma_0^{-1} is tridiagonal and we add a multiple
    of the identity.
    """
    x = np.asarray(x, dtype=float)
    K = x.shape[0]
    mu, Delta = ou_params(t)
    Q0 = ar1_precision_clean(K, alpha, sigma_eta)
    J = (mu * mu / Delta) * np.eye(K) + Q0
    mu_a = prior_mean(K, alpha, mu_0)
    h = (mu / Delta) * x + Q0 @ mu_a
    return J, h


# ---------------------------------------------------------------------------
# Exact marginals
# ---------------------------------------------------------------------------

def exact_marginals(J: np.ndarray, h: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Exact (mean, variance) of p(a) propto exp(-1/2 a^T J a + h^T a)."""
    Jinv = np.linalg.inv(J)
    mean = Jinv @ h
    var = np.diag(Jinv).copy()
    return mean, var


# ---------------------------------------------------------------------------
# Shared mean iteration (the linear system J m = h)
# ---------------------------------------------------------------------------

def mean_iteration(J: np.ndarray, h: np.ndarray, damping: float = 0.5,
                   max_iter: int = 10_000, tol: float = 1e-13
                   ) -> tuple[np.ndarray, int, bool]:
    """Damped Jacobi iteration whose fixed point is the exact mean J m = h.

        m_i <- (1 - g) m_i + g (h_i - sum_{k != i} J_ik m_k) / J_ii.

    Mean field, BP and AMP all have this same mean fixed point; only the
    variance closure differs.  Returned to demonstrate that the message-passing
    mean converges to the exact posterior mean.
    """
    K = len(h)
    d = np.diag(J).copy()
    Off = J - np.diag(d)
    m = np.zeros(K)
    for it in range(1, max_iter + 1):
        m_new = (h - Off @ m) / d
        m_next = (1.0 - damping) * m + damping * m_new
        if np.max(np.abs(m_next - m)) < tol:
            return m_next, it, True
        m = m_next
    return m, max_iter, False


# ---------------------------------------------------------------------------
# Variance closures: mean field, AMP/TAP, exact
# ---------------------------------------------------------------------------

def mean_field_variance(J: np.ndarray) -> np.ndarray:
    """Naive mean-field variance V_i = 1 / J_ii (neighbours ignored)."""
    return 1.0 / np.diag(J)


def amp_variance(J: np.ndarray, max_iter: int = 10_000, tol: float = 1e-13,
                 damping: float = 0.5) -> tuple[np.ndarray, int, bool]:
    """AMP / TAP cavity variance, solved self-consistently:

        V_i = 1 / ( J_ii - sum_{k != i} J_ik^2 V_k ).

    This is the cavity variance with the "exclude one neighbour" correction
    dropped (V_{k -> i} approximated by V_k).  Exact when each node has many
    weak neighbours (dense regime); approximate on the chain, and -- when the
    coupling is strong enough -- it has no physical (positive) fixed point at
    all.  In that case we report failure honestly (NaN, converged=False)
    rather than a clipped meaningless number.
    """
    d = np.diag(J).copy()
    W2 = J * J
    np.fill_diagonal(W2, 0.0)
    V = 1.0 / d
    for it in range(1, max_iter + 1):
        denom = d - W2 @ V
        if np.any(denom <= 0):
            # The self-consistent closure left the physical (SPD) region:
            # no positive-variance fixed point exists for this coupling.
            return np.full_like(V, np.nan), it, False
        V_new = 1.0 / denom
        V_next = (1.0 - damping) * V + damping * V_new
        if np.max(np.abs(V_next - V)) < tol:
            return V_next, it, True
        V = V_next
    return V, max_iter, False


# ---------------------------------------------------------------------------
# Convenience: marginals from each scheme
# ---------------------------------------------------------------------------

def amp_marginals(J: np.ndarray, h: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """AMP marginals: exact mean (J m = h) and AMP cavity variance."""
    mean, _, _ = mean_iteration(J, h)
    var, _, _ = amp_variance(J)
    return mean, var


def amp_score(x: np.ndarray, t: float, alpha: float,
              sigma_eta: float | None = None, mu_0: float = 0.0) -> np.ndarray:
    """Joint score from the AMP posterior mean via Tweedie's identity."""
    J, h = posterior_precision_field(x, t, alpha, sigma_eta, mu_0)
    mean, _, _ = mean_iteration(J, h)
    mu, Delta = ou_params(t)
    return (mu * mean - np.asarray(x, dtype=float)) / Delta


# ---------------------------------------------------------------------------
# Correctness gate: dense vs chain
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- Gate 1: dense random SPD J -- AMP variance must match exact --------
    N = 400
    A = rng.standard_normal((N, N)) / np.sqrt(N)
    J_dense = np.eye(N) + 0.5 * (A + A.T) / np.sqrt(2.0)
    # shift to ensure SPD with a comfortable margin
    J_dense += (abs(min(np.linalg.eigvalsh(J_dense))) + 0.5) * np.eye(N)
    h_dense = rng.standard_normal(N)

    m_ex, v_ex = exact_marginals(J_dense, h_dense)
    m_it, n_it, ok_m = mean_iteration(J_dense, h_dense)
    v_amp, n_v, ok_v = amp_variance(J_dense)
    v_mf = mean_field_variance(J_dense)
    print("=== Gate 1: dense random SPD J (N=%d) ===" % N)
    print(f"  mean iteration converged={ok_m} in {n_it} steps; "
          f"max|m_iter - m_exact| = {np.max(np.abs(m_it - m_ex)):.2e}")
    print(f"  AMP variance converged={ok_v} in {n_v} steps; "
          f"max|V_amp - V_exact| = {np.max(np.abs(v_amp - v_ex)):.2e}  "
          f"(should be small: AMP exact on dense)")
    print(f"  mean-field variance error  = {np.max(np.abs(v_mf - v_ex)):.2e}  "
          f"(should be larger)")

    # --- Gate 2: AR(1) chain -- AMP mean exact, AMP variance NOT exact ------
    K, alpha = 9, 0.8
    x = rng.standard_normal(K)
    print(f"\n=== Gate 2: AR(1) chain (K={K}, alpha={alpha}) ===")
    print("   mean is exact for every t (same score as BP); the AMP variance")
    print("   closure is accurate at small t (evidence-dominated, weak coupling)")
    print("   and degrades / breaks down at larger t (prior-dominated):")
    for t in (0.05, 0.2, 0.5, 1.0, 2.0):
        J_chain, h_chain = posterior_precision_field(x, t, alpha)
        m_ex, v_ex = exact_marginals(J_chain, h_chain)
        m_amp = mean_iteration(J_chain, h_chain)[0]
        v_amp, _, ok = amp_variance(J_chain)
        err_m = np.max(np.abs(m_amp - m_ex))
        if ok:
            err_v = f"{np.max(np.abs(v_amp - v_ex)):.2e}"
        else:
            err_v = "no physical fixed point (breakdown)"
        print(f"   t={t:<4}  |m_amp-m_exact|={err_m:.1e}   AMP var: {err_v}")
