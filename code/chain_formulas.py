"""
chain_formulas.py
=================

Closed-form results for the HOMOGENEOUS INTERIOR (bulk) of the posterior
precision of the AR(1)+OU model, and the fully explicit K = 2 case.

Everything in this module is an explicit formula -- no iteration, no matrix
inversion.  Each formula is verified against brute force in
``numerical_audit.py`` (sections 10 and 11); the derivations are in the
companion note ``main.pdf``:

Bulk of the chain (interior of a long chain, far from both boundaries).
The posterior precision J = (e^{-2t}/Delta_t) I + Q_0 is tridiagonal
Toeplitz in the interior, with

    J_d  = e^{-2t}/Delta_t + (1 + alpha^2) / sigma_eta^2   (diagonal)
    beta = -alpha / sigma_eta^2                            (off-diagonal),

and sigma_eta^2 = 1 - alpha^2 in the stationary unit-variance normalisation.

  * Exact bulk posterior variance  (= what BP returns in the bulk):

        V_exact = 1 / sqrt(J_d^2 - 4 beta^2).

  * Gaussian-BP cavity (message) precision fixed point:

        lambda* = ( J_d + sqrt(J_d^2 - 4 beta^2) ) / 2 ,

    which exists for EVERY (alpha, t) because J_d >= 2 |beta| always
    (AM-GM: 1 + alpha^2 >= 2|alpha|, plus the positive evidence term).
    Recombining,  J_d - 2 beta^2 / lambda* = sqrt(J_d^2 - 4 beta^2),
    i.e. BP reproduces V_exact -- BP never breaks down on the chain.

  * AMP/TAP variance fixed point (cavity WITHOUT the exclusion):
    the closure V = 1 / (J_d - 2 beta^2 V) is the quadratic
    2 beta^2 V^2 - J_d V + 1 = 0, whose physical root is

        V_amp = ( J_d - sqrt(J_d^2 - 8 beta^2) ) / (4 beta^2),

    and which has a real solution  iff  J_d >= 2 sqrt(2) |beta|.
    Below that line in the (alpha, t) plane the AMP variance has no
    physical fixed point at all (breakdown).  Note the factor 8 = 2*4:
    AMP needs a strictly stronger condition than BP.

K = 2 explicitly.  With unit stationary variance, Sigma_0 = [[1, a],[a, 1]],

    Sigma_t = [[1, alpha e^{-2t}], [alpha e^{-2t}, 1]],
    Q_t     = 1/(1 - alpha^2 e^{-4t}) [[1, -alpha e^{-2t}],
                                       [-alpha e^{-2t}, 1]],
    S_0(x)  = -(x_0 - alpha e^{-2t} x_1) / (1 - alpha^2 e^{-4t}),

and symmetrically for S_1.  The cross term -alpha e^{-2t} x_1 in S_0 is the
inter-frame coupling that the per-frame marginal score misses entirely.
"""

from __future__ import annotations

import numpy as np

from ar1_utils import ou_params


# ---------------------------------------------------------------------------
# Bulk (homogeneous interior) parameters of the posterior precision
# ---------------------------------------------------------------------------

def bulk_params(alpha: float, t: float) -> tuple[float, float]:
    """(J_d, beta): diagonal and off-diagonal of the interior of J.

    J = (e^{-2t}/Delta_t) I + Q_0 with Q_0 the clean tridiagonal precision;
    in the interior Q_0 has diagonal (1+alpha^2)/sigma_eta^2 and off-diagonal
    -alpha/sigma_eta^2, with sigma_eta^2 = 1 - alpha^2.
    """
    mu, Delta = ou_params(t)
    s2 = 1.0 - alpha * alpha
    J_d = mu * mu / Delta + (1.0 + alpha * alpha) / s2
    beta = -alpha / s2
    return J_d, beta


def bulk_variance_exact(alpha: float, t: float) -> float:
    """Exact posterior variance in the bulk:  1 / sqrt(J_d^2 - 4 beta^2).

    This is also what Gaussian BP returns in the bulk (BP is exact on the
    chain); the identity J_d - 2 beta^2/lambda* = sqrt(J_d^2 - 4 beta^2)
    is checked in the audit.
    """
    J_d, beta = bulk_params(alpha, t)
    return 1.0 / np.sqrt(J_d * J_d - 4.0 * beta * beta)


def bp_cavity_precision(alpha: float, t: float) -> float:
    """BP message-precision fixed point  lambda* = (J_d + sqrt(J_d^2-4b^2))/2.

    The discriminant J_d^2 - 4 beta^2 is positive for every (alpha, t):
    BP always has a (unique attractive) fixed point on the chain.
    """
    J_d, beta = bulk_params(alpha, t)
    return 0.5 * (J_d + np.sqrt(J_d * J_d - 4.0 * beta * beta))


def amp_bulk_variance(alpha: float, t: float) -> float:
    """AMP/TAP bulk variance fixed point, or NaN when none exists.

        V_amp = (J_d - sqrt(J_d^2 - 8 beta^2)) / (4 beta^2),  J_d >= 2*sqrt(2)|beta|;
        NaN otherwise (no physical fixed point: AMP breakdown).

    The physical root is the '-' branch: it is the one continuously
    connected to the decoupled limit V = 1/J_d as beta -> 0.
    """
    J_d, beta = bulk_params(alpha, t)
    disc = J_d * J_d - 8.0 * beta * beta
    if disc < 0.0:
        return float("nan")
    return (J_d - np.sqrt(disc)) / (4.0 * beta * beta)


def amp_fixed_point_exists(alpha: float, t: float) -> bool:
    """Existence condition of the AMP variance closure:  J_d >= 2 sqrt(2) |beta|."""
    J_d, beta = bulk_params(alpha, t)
    return J_d * J_d - 8.0 * beta * beta >= 0.0


def amp_critical_time(alpha: float) -> float:
    """Exact diffusion time at which the AMP variance fixed point disappears.

    Solving J_d(alpha, t) = 2 sqrt(2) |beta(alpha)| for t, with
    u := e^{-2t} and  e^{-2t}/Delta_t = u/(1-u):

        g(alpha) = (2 sqrt(2)|alpha| - 1 - alpha^2) / (1 - alpha^2),
        t_c      = -(1/2) log( g / (1 + g) )       if g > 0,
        t_c      = +inf                             if g <= 0.

    g > 0  iff  alpha^2 - 2 sqrt(2)|alpha| + 1 < 0  iff  |alpha| > sqrt(2)-1.
    Hence the CRITICAL COUPLING

        alpha_c = sqrt(2) - 1 = 0.41421...:

    for |alpha| <= alpha_c the AMP variance has a fixed point at EVERY t;
    for |alpha| > alpha_c it breaks down for all t > t_c(alpha).
    Verified against the flip point of the self-consistent iteration
    (bisection, K=300) to 4-5 digits in the audit.
    """
    a = abs(alpha)
    g = (2.0 * np.sqrt(2.0) * a - 1.0 - a * a) / (1.0 - a * a)
    if g <= 0.0:
        return float("inf")
    return -0.5 * np.log(g / (1.0 + g))


def bulk_correlation_decay(alpha: float, t: float) -> float:
    """Posterior correlation decay rate q in (0, 1):

        q = ( J_d - sqrt(J_d^2 - 4 beta^2) ) / ( 2 |beta| ).

    The bulk posterior covariance decays geometrically,
        (J^{-1})_{i, i+d} = q^d / sqrt(J_d^2 - 4 beta^2),
    so 1/log(1/q) is the correlation length of the posterior in frames.
    The same q is the decay rate of the locality (truncation) error of the
    radius-r local estimator -- see local_bp.rms_truncation_error and the
    audit.  q -> 0 at small t (evidence pins each frame: posterior almost
    local) and q -> alpha-dependent limit at large t (prior-dominated).
    """
    J_d, beta = bulk_params(alpha, t)
    return (J_d - np.sqrt(J_d * J_d - 4.0 * beta * beta)) / (2.0 * abs(beta))


def bulk_covariance(alpha: float, t: float, d: int) -> float:
    """Closed-form bulk posterior covariance  Cov(a_i, a_{i+d} | x):

        (J^{-1})_{i, i+d} = q^d / sqrt(J_d^2 - 4 beta^2).

    Verified against brute-force inversion at the centre of a long chain
    in the audit (machine precision).
    """
    return bulk_variance_exact(alpha, t) * bulk_correlation_decay(alpha, t) ** d


def amp_weak_coupling_error(alpha: float, t: float) -> float:
    """Leading weak-coupling AMP variance error:

        V_amp - V_exact = 2 beta^4 / J_d^5 + O(beta^6 / J_d^7).

    Both closures agree to O(beta^2/J_d^2); AMP first deviates -- always
    OVERestimating the variance -- at fourth order in the coupling.
    """
    J_d, beta = bulk_params(alpha, t)
    return 2.0 * beta ** 4 / J_d ** 5


# ---------------------------------------------------------------------------
# K = 2: everything fully explicit
# ---------------------------------------------------------------------------

def k2_sigma_t(alpha: float, t: float) -> np.ndarray:
    """Noised covariance for K=2:  [[1, alpha e^{-2t}], [alpha e^{-2t}, 1]].

    Diagonal: e^{-2t} * 1 + Delta_t = 1 (variance-preserving channel on a
    unit-variance frame).  Off-diagonal: e^{-2t} * alpha (the noise is
    independent across frames, so only the signal part correlates).
    """
    r = alpha * np.exp(-2.0 * t)
    return np.array([[1.0, r], [r, 1.0]])


def k2_precision_t(alpha: float, t: float) -> np.ndarray:
    """Closed-form inverse of k2_sigma_t (2x2 inversion)."""
    r = alpha * np.exp(-2.0 * t)
    return np.array([[1.0, -r], [-r, 1.0]]) / (1.0 - r * r)


def k2_score(x: np.ndarray, alpha: float, t: float) -> np.ndarray:
    """Joint score for K=2 in closed form:

        S_0 = -(x_0 - r x_1) / (1 - r^2),   r = alpha e^{-2t},

    and symmetrically for S_1.  The r x_1 term in S_0 is the inter-frame
    coupling; the per-frame marginal score is just -x_0 (unit variance)
    and misses it.
    """
    x = np.asarray(x, dtype=float)
    r = alpha * np.exp(-2.0 * t)
    return np.array([
        -(x[0] - r * x[1]) / (1.0 - r * r),
        -(x[1] - r * x[0]) / (1.0 - r * r),
    ])
