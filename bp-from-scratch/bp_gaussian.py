"""
bp_gaussian.py
==============

Belief propagation for the Gaussian AR(1) diffusion model, written from
scratch following the discussion with J. Garnier-Brun (call of 2026-06-05).

Everything in this file is self-contained: it implements only what the
companion note (main.pdf in this folder) derives, in the same notation,
and nothing else.  Run the file to execute the built-in checks.

The model
---------
Clean sequence (the data):    a_{k+1} = alpha * a_k + eta_k,
                              eta_k ~ N(0, 1 - alpha^2)  i.i.d.,
                              a_0 ~ N(0, 1)              (stationary start)
Diffusion channel at time t:  x_k = e^{-t} a_k + xi_k,
                              xi_k ~ N(0, D_t),  D_t = 1 - e^{-2t},
                              independent across k.

The object we want is the JOINT SCORE  S(x,t) = grad_x log P_t(x), where
P_t is the law of the noisy sequence x = (x_0 ... x_{K-1}).

Sections of this file (mirroring the note):
  1. model matrices and the exact score (the ground truth)
  2. belief propagation with Gaussian messages (two numbers per message)
  3. experiment 1a: truncating the exact score matrix to a band
  4. experiment 1b: truncating the message range ("cut the iteration")
  5. experiment 2:  AMP / tapification (per-node mean & variance updates)
  6. error metrics (deterministic RMS over x ~ P_t, no sampling)
  7. self-checks (run as a script)
"""

from __future__ import annotations

import numpy as np


# ===========================================================================
# 1. Model matrices and the exact score
# ===========================================================================

def channel(t: float) -> tuple[float, float]:
    """The diffusion channel constants (mu, D_t) at time t:

        x_k = mu * a_k + noise,   mu = e^{-t},   noise ~ N(0, D_t),
        D_t = 1 - e^{-2t}.

    Chosen variance-preserving: mu^2 * 1 + D_t = 1, so each x_k is N(0,1)
    marginally at every t.
    """
    mu = np.exp(-t)
    return mu, 1.0 - mu * mu


def clean_covariance(K: int, alpha: float) -> np.ndarray:
    """Sigma_0[i, j] = alpha^{|i-j|}: the covariance of the stationary
    unit-variance AR(1) chain (derived in the note, Step 2.1)."""
    idx = np.arange(K)
    return alpha ** np.abs(np.subtract.outer(idx, idx))


def clean_precision(K: int, alpha: float) -> np.ndarray:
    """Q_0 = Sigma_0^{-1}, written directly from the closed form
    (note, Step 2.2): tridiagonal, with

        diagonal  : 1/s2 at both ends, (1+alpha^2)/s2 inside,
        off-diag  : -alpha/s2,                 s2 = 1 - alpha^2.

    The chain couples only neighbours, so the *precision* (not the
    covariance) is the sparse object.
    """
    s2 = 1.0 - alpha * alpha
    Q = np.zeros((K, K))
    for k in range(K):
        Q[k, k] = (1.0 if k in (0, K - 1) else 1.0 + alpha * alpha) / s2
        if k + 1 < K:
            Q[k, k + 1] = Q[k + 1, k] = -alpha / s2
    return Q


def noisy_covariance(K: int, alpha: float, t: float) -> np.ndarray:
    """Sigma_t = e^{-2t} Sigma_0 + D_t I  (note, Step 3.1)."""
    mu, D = channel(t)
    return mu * mu * clean_covariance(K, alpha) + D * np.eye(K)


def exact_score(x: np.ndarray, alpha: float, t: float) -> np.ndarray:
    """Ground truth:  S(x,t) = -Sigma_t^{-1} x  (note, Step 3.2).

    This is the object every algorithm below must reproduce or
    approximate.  Cost: one K x K solve, O(K^3) done naively.
    """
    x = np.asarray(x, float)
    return -np.linalg.solve(noisy_covariance(len(x), alpha, t), x)


def posterior_mean_matrix(K: int, alpha: float, t: float) -> np.ndarray:
    """The matrix C with E[a | x] = C x  (note, Step 3.4).

    From completing the square in p(a|x) ∝ P_0(a) p(x|a):
        J = (mu^2 / D) I + Q_0      (posterior precision, tridiagonal)
        h = (mu  / D) x             (posterior field)
        E[a|x] = J^{-1} h  =  (mu/D) J^{-1} x  =: C x.
    """
    mu, D = channel(t)
    J = (mu * mu / D) * np.eye(K) + clean_precision(K, alpha)
    return (mu / D) * np.linalg.inv(J)


def score_from_denoiser(m: np.ndarray, x: np.ndarray, t: float) -> np.ndarray:
    """Tweedie's identity (note, Step 3.3):

        S_k(x,t) = ( e^{-t} * E[a_k | x]  -  x_k ) / D_t .

    Any algorithm that produces the posterior mean m = E[a|x] produces the
    score through this one line.
    """
    mu, D = channel(t)
    return (mu * np.asarray(m, float) - np.asarray(x, float)) / D


# ===========================================================================
# 2. Belief propagation with Gaussian messages
# ===========================================================================
#
# Every message is a Gaussian in its receiving variable and is therefore
# carried by TWO NUMBERS (mean, variance).  This is not an approximation
# for this model: the note (Step 5.3) proves each update maps Gaussians to
# Gaussians exactly.  A variance of +inf encodes the flat (uninformative)
# message.
#
# Bookkeeping convention (note, Step 5.2 -- stated explicitly because it is
# the one place where it is easy to silently double-count evidence):
#     * the forward message into node k summarises the prior and the
#       observations x_0 .. x_{k-1} ONLY;
#     * the backward message into node k summarises x_{k+1} .. x_{K-1} ONLY;
#     * the local observation x_k enters ONCE, at combination time.

def _evidence(x_k: float, t: float) -> tuple[float, float]:
    """The observation factor g_k(a_k) = N(x_k; mu a_k, D), read as a
    Gaussian in a_k (note, eq. (E)):

        g_k(a_k) ∝ N(a_k ; x_k/mu, D/mu^2).

    It is a 'pseudo-observation' of the clean frame at the de-attenuated
    location x_k/mu = e^{t} x_k with variance D e^{2t}.
    """
    mu, D = channel(t)
    return x_k / mu, D / (mu * mu)


def _product(m1: float, v1: float, m2: float, v2: float) -> tuple[float, float]:
    """Multiply two Gaussians in the same variable (note, Lemma 1):
    precisions add, precision-weighted means add.  Handles v = +inf
    (flat factor) exactly."""
    if np.isinf(v1):
        return m2, v2
    if np.isinf(v2):
        return m1, v1
    lam = 1.0 / v1 + 1.0 / v2
    return (m1 / v1 + m2 / v2) / lam, 1.0 / lam


def _through_transition(m: float, v: float, alpha: float, s2: float,
                        direction: str) -> tuple[float, float]:
    """Push a Gaussian belief through the transition factor
    f(a, a') = N(a'; alpha a, s2)  (note, Lemma 2 and eq. (B2)).

    direction='fwd' : integrate over a  (we know a ~ N(m,v), want a'):
                      a' = alpha a + eta  ->  N(alpha m, alpha^2 v + s2).
    direction='bwd' : integrate over a' (we know a' ~ N(m,v), want a):
                      read the same factor as a function of a; completing
                      the square gives N(m/alpha, (v + s2)/alpha^2).
    """
    if direction == "fwd":
        return alpha * m, alpha * alpha * v + s2
    return m / alpha, (v + s2) / (alpha * alpha)


def bp_messages(x: np.ndarray, alpha: float, t: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the two sweeps.  Returns (m_fwd, v_fwd, m_bwd, v_bwd): the
    Gaussian parameters of the forward and backward messages at each node.

    Forward sweep  (note, eqs. (F0)-(F2)):  prior -> combine evidence ->
        push through transition, left to right.
    Backward sweep (note, eqs. (B0)-(B2)):  flat -> combine evidence ->
        push through transition, right to left.
    """
    x = np.asarray(x, float)
    K = len(x)
    s2 = 1.0 - alpha * alpha

    m_fwd = np.zeros(K)
    v_fwd = np.zeros(K)
    m_fwd[0], v_fwd[0] = 0.0, 1.0                      # (F0): the prior N(0,1)
    for k in range(K - 1):
        me, ve = _evidence(x[k], t)                    # (F1): absorb x_k
        mc, vc = _product(m_fwd[k], v_fwd[k], me, ve)
        m_fwd[k + 1], v_fwd[k + 1] = _through_transition(mc, vc, alpha, s2,
                                                         "fwd")   # (F2)

    m_bwd = np.zeros(K)
    v_bwd = np.full(K, np.inf)                         # (B0): flat at the end
    for k in range(K - 1, 0, -1):
        me, ve = _evidence(x[k], t)                    # (B1): absorb x_k
        mc, vc = _product(m_bwd[k], v_bwd[k], me, ve)
        m_bwd[k - 1], v_bwd[k - 1] = _through_transition(mc, vc, alpha, s2,
                                                         "bwd")   # (B2)
    return m_fwd, v_fwd, m_bwd, v_bwd


def bp_posterior(x: np.ndarray, alpha: float, t: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Combine, at every node, forward message x local evidence x backward
    message (note, eq. (C)).  Returns the K posterior (means, variances)."""
    x = np.asarray(x, float)
    K = len(x)
    m_fwd, v_fwd, m_bwd, v_bwd = bp_messages(x, alpha, t)
    means = np.zeros(K)
    variances = np.zeros(K)
    for k in range(K):
        me, ve = _evidence(x[k], t)
        mc, vc = _product(m_fwd[k], v_fwd[k], me, ve)
        means[k], variances[k] = _product(mc, vc, m_bwd[k], v_bwd[k])
    return means, variances


def bp_score(x: np.ndarray, alpha: float, t: float) -> np.ndarray:
    """The full pipeline: two sweeps + combination + Tweedie.  O(K)."""
    m, _ = bp_posterior(x, alpha, t)
    return score_from_denoiser(m, x, t)


# ===========================================================================
# 3. Experiment 1a: truncate the exact score MATRIX to a band
# ===========================================================================
#
# Jerome (call, 31:27): "imagine you take the exact score and instead of
# using the full matrix you truncate it to the tridiagonal thing.  How bad
# is the approximation?"

def banded_score_matrix(K: int, alpha: float, t: float, band: int
                        ) -> np.ndarray:
    """Q_t with every entry at distance > band set to zero.
    band = 1 is the tridiagonal truncation; band = K-1 is exact."""
    Q = np.linalg.inv(noisy_covariance(K, alpha, t))
    mask = np.abs(np.subtract.outer(np.arange(K), np.arange(K))) <= band
    return Q * mask


# ===========================================================================
# 4. Experiment 1b: truncate the MESSAGE RANGE ("cut the iteration")
# ===========================================================================
#
# Jerome (call, 31:48): "you will restrain your factor graph in a way, and
# you cut the iteration at some point ... the nodes are updated in such a
# way that we don't actually run through the entire graph, but we do local
# approximations every time."
#
# Concretely: to estimate frame k we let messages travel at most r hops, and
# beyond the cut we replace the incoming message by what it is BEFORE any
# evidence arrives -- the stationary prior marginal N(0,1).  Because a
# contiguous block of the stationary chain is itself a stationary AR(1)
# chain, this truncated message passing is identical to exact inference on
# the window  x_{k-r} .. x_{k+r}  alone.

def local_mean_matrix(K: int, alpha: float, t: float, r: int) -> np.ndarray:
    """The matrix C_r with  m_r(x) = C_r x, where m_r(x)[k] is the
    radius-r truncated-BP estimate of E[a_k | window of x around k]."""
    mu, D = channel(t)
    C = np.zeros((K, K))
    for k in range(K):
        lo, hi = max(0, k - r), min(K - 1, k + r)
        n = hi - lo + 1
        J = (mu * mu / D) * np.eye(n) + clean_precision(n, alpha)
        Cw = (mu / D) * np.linalg.inv(J)          # window smoother
        C[k, lo:hi + 1] = Cw[k - lo]
    return C


def local_bp_score(x: np.ndarray, alpha: float, t: float, r: int
                   ) -> np.ndarray:
    """Score from the radius-r truncated message passing, via Tweedie."""
    x = np.asarray(x, float)
    m = local_mean_matrix(len(x), alpha, t, r) @ x
    return score_from_denoiser(m, x, t)


# ===========================================================================
# 5. Experiment 2: AMP / tapification (per-node mean & variance updates)
# ===========================================================================
#
# The call's prescription: instead of propagating per-EDGE messages, keep a
# single (mean, variance) per NODE and update them self-consistently --
# "you reduce everything to a mean and variance update" (call, 21:49).
# We work on the posterior in natural parameters,
#     p(a|x) ∝ exp( -1/2 a^T J a + h^T a ),
#     J = (mu^2/D) I + Q_0  (tridiagonal),   h = (mu/D) x.

def posterior_natural_params(x: np.ndarray, alpha: float, t: float
                             ) -> tuple[np.ndarray, np.ndarray]:
    mu, D = channel(t)
    K = len(x)
    J = (mu * mu / D) * np.eye(K) + clean_precision(K, alpha)
    return J, (mu / D) * np.asarray(x, float)


def amp_mean(J: np.ndarray, h: np.ndarray, damping: float = 0.5,
             tol: float = 1e-13, max_iter: int = 20000
             ) -> tuple[np.ndarray, bool]:
    """Per-node mean update,
        m_i <- ( h_i - sum_{j != i} J_ij m_j ) / J_ii ,
    damped for stability.  Its fixed point satisfies J m = h EXACTLY --
    the mean update closes on the true posterior mean regardless of what
    the variances do (note, Step 7.2)."""
    d = np.diag(J).copy()
    Off = J - np.diag(d)
    m = np.zeros_like(h)
    for _ in range(max_iter):
        m_new = (1 - damping) * m + damping * (h - Off @ m) / d
        if np.max(np.abs(m_new - m)) < tol:
            return m_new, True
        m = m_new
    return m, False


def amp_variance(J: np.ndarray, damping: float = 0.5,
                 tol: float = 1e-13, max_iter: int = 20000
                 ) -> tuple[np.ndarray, bool]:
    """Per-node variance update,
        V_i <- 1 / ( J_ii - sum_{j != i} J_ij^2 V_j ).
    This is the Gaussian/AMP closure of the call: each node sees the
    aggregate of its neighbours' current uncertainties, with NO per-edge
    exclusion.  When the bracket turns non-positive there is no physical
    fixed point and we report failure honestly (NaN, False) instead of a
    clipped number."""
    d = np.diag(J).copy()
    W2 = J * J
    np.fill_diagonal(W2, 0.0)
    V = 1.0 / d
    for _ in range(max_iter):
        bracket = d - W2 @ V
        if np.any(bracket <= 0.0):
            return np.full_like(V, np.nan), False
        V_new = (1 - damping) * V + damping / bracket
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, True
        V = V_new
    return V, False


def amp_score(x: np.ndarray, alpha: float, t: float) -> np.ndarray:
    """AMP's score: per-node mean iteration + Tweedie."""
    J, h = posterior_natural_params(np.asarray(x, float), alpha, t)
    m, _ = amp_mean(J, h)
    return score_from_denoiser(m, x, t)


# ===========================================================================
# 6. Deterministic error metrics (no sampling)
# ===========================================================================
#
# Every estimator above is LINEAR in x:  S_hat(x) = -M x for some matrix M
# (M = banded Q_t, or built from C_r, ...), while the truth is S = -Q_t x.
# Over the data distribution x ~ P_t = N(0, Sigma_t), the mean squared
# error has the closed form
#     E || S_hat - S ||^2  =  tr( (M - Q_t) Sigma_t (M - Q_t)^T ),
# and the score's own size is  E||S||^2 = tr(Q_t Sigma_t Q_t) = tr(Q_t).
# We report the relative root error -- a number with no seeds attached.

def relative_score_error(M: np.ndarray, K: int, alpha: float, t: float
                         ) -> float:
    """sqrt( E||(M - Q_t) x||^2 / E||Q_t x||^2 )  for x ~ N(0, Sigma_t)."""
    St = noisy_covariance(K, alpha, t)
    Qt = np.linalg.inv(St)
    Dlt = M - Qt
    num = np.trace(Dlt @ St @ Dlt.T)
    den = np.trace(Qt)                      # = tr(Qt St Qt)
    return float(np.sqrt(max(num, 0.0) / den))


def score_matrix_of_local_bp(K: int, alpha: float, t: float, r: int
                             ) -> np.ndarray:
    """The matrix M_r with S_local(x) = -M_r x, from Tweedie applied to the
    linear estimator m_r(x) = C_r x:
        S_local = (mu C_r - I) x / D   =>   M_r = (I - mu C_r)/D."""
    mu, D = channel(t)
    return (np.eye(K) - mu * local_mean_matrix(K, alpha, t, r)) / D


# ===========================================================================
# 7. Self-checks
# ===========================================================================

def _run_checks() -> None:
    rng = np.random.default_rng(7)
    print("=" * 72)
    print("bp_gaussian.py self-checks")
    print("=" * 72)

    # --- check 1: closed-form Q_0 really inverts Sigma_0 ------------------
    worst = 0.0
    for K in (4, 9, 17):
        for a in (0.25, 0.6, 0.9, -0.45):
            worst = max(worst, float(np.max(np.abs(
                clean_precision(K, a) @ clean_covariance(K, a) - np.eye(K)))))
    print(f"[1] Q_0 Sigma_0 = I ............................ {worst:.2e}")
    assert worst < 1e-10

    # --- check 2: BP posterior == matrix posterior ------------------------
    worst_m = worst_v = worst_s = 0.0
    for K in (2, 3, 6, 12):
        for a in (0.3, 0.7, 0.9, -0.5):
            for t in (0.05, 0.4, 1.2, 3.0):
                x = rng.standard_normal(K)
                m_bp, v_bp = bp_posterior(x, a, t)
                C = posterior_mean_matrix(K, a, t)
                mu, D = channel(t)
                J = (mu * mu / D) * np.eye(K) + clean_precision(K, a)
                Sig_post = np.linalg.inv(J)
                worst_m = max(worst_m, float(np.max(np.abs(m_bp - C @ x))))
                worst_v = max(worst_v, float(np.max(np.abs(
                    v_bp - np.diag(Sig_post)))))
                worst_s = max(worst_s, float(np.max(np.abs(
                    bp_score(x, a, t) - exact_score(x, a, t)))))
    print(f"[2] BP mean  == matrix mean .................... {worst_m:.2e}")
    print(f"    BP var   == matrix var ..................... {worst_v:.2e}")
    print(f"    BP score == exact score  (CHECKPOINT) ...... {worst_s:.2e}")
    assert max(worst_m, worst_v, worst_s) < 1e-10

    # --- check 3: local BP with r = K-1 is exact ---------------------------
    K, a = 15, 0.8
    x = rng.standard_normal(K)
    worst = max(float(np.max(np.abs(
        local_bp_score(x, a, t, K - 1) - exact_score(x, a, t))))
        for t in (0.1, 0.6, 2.0))
    print(f"[3] local BP, r = K-1, == exact ................ {worst:.2e}")
    assert worst < 1e-10

    # --- check 4: banded matrix with band = K-1 is exact -------------------
    M = banded_score_matrix(K, a, 0.6, K - 1)
    err = relative_score_error(M, K, a, 0.6)
    print(f"[4] banded score, band = K-1, rel. error ....... {err:.2e}")
    assert err < 1e-12

    # --- check 5: AMP mean == exact mean (=> same score) -------------------
    worst = 0.0
    for t in (0.05, 0.3, 1.0, 3.0):
        x = rng.standard_normal(9)
        worst = max(worst, float(np.max(np.abs(
            amp_score(x, 0.8, t) - exact_score(x, 0.8, t)))))
    print(f"[5] AMP score == exact score (all t) ........... {worst:.2e}")
    assert worst < 1e-9

    # --- check 6: AMP variance accurate small t, breaks at larger t --------
    x = rng.standard_normal(9)
    J, _ = posterior_natural_params(x, 0.8, 0.05)
    V, ok = amp_variance(J)
    v_exact = np.diag(np.linalg.inv(J))
    err_small_t = float(np.max(np.abs(V - v_exact)))
    J, _ = posterior_natural_params(x, 0.8, 1.0)
    _, ok_large = amp_variance(J)
    print(f"[6] AMP variance err at t=0.05 ................. {err_small_t:.2e}")
    print(f"    AMP variance at t=1.0 has fixed point? ..... {ok_large}"
          f"   (expected: False)")
    assert ok and err_small_t < 1e-2 and not ok_large

    print("-" * 72)
    print("ALL SELF-CHECKS PASS")


if __name__ == "__main__":
    _run_checks()
