"""
numerical_audit.py
==================

Independent numerical verification of every closed-form identity quoted in the
companion note "Gaussian Belief Propagation for the Diffusion Score".  The
audit is the gate: a statement is allowed in the text only if the corresponding
check passes here.  Each check fixes a small set of parameters, evaluates the
closed form and an independent reference (matrix product, exact inverse,
spectral identity, finite differences), and reports the maximum error against
an explicit tolerance.

Sections
    1. Clean precision Q_0 is tridiagonal and equals Sigma_0^{-1}.
    2. Sigma_t Q_t = I and the spectral preservation Q_t = U f(lambda) U^T.
    3. Band-fill leading order, |Q_t[i,i+d]| ~ (2t)^{d-1} (NO 1/(d-1)! factor).
    4. Large-t return Q_t -> I.
    5. BP (Convention A) reproduces the matrix posterior and the exact score.
    6. Tweedie matrix score equals -Q_t x.
    7. AMP mean == exact mean == BP score (the score is closure-independent).
    8. AMP variance: accurate on a dense graph, breakdown on the chain.
    9. Local (radius-r) BP -> full BP as r -> K-1; error monotone in r.

Tolerance bands:
    matrix algebra / inverses        : 1e-10 .. 1e-12
    band-fill leading coefficient    : 0.05  (relative)
    large-t exponential return       : structure-dependent, see check
"""

from __future__ import annotations

from dataclasses import dataclass
import sys

import numpy as np

from ar1_utils import (
    ar1_covariance,
    ar1_precision_clean,
    gaussian_posterior,
    joint_score_matrix,
    joint_score_via_tweedie,
    precision_t,
    precision_t_spectral,
    sigma_t,
)
from bp_score import bp_posterior, bp_score
from amp import (
    amp_score,
    amp_variance,
    exact_marginals,
    mean_field_variance,
    mean_iteration,
    posterior_precision_field,
)
from local_bp import local_score


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    error: float
    tol: float

    @property
    def passed(self) -> bool:
        return self.error <= self.tol


def _fmt(c: Check) -> str:
    flag = "PASS" if c.passed else "FAIL"
    return f"[{flag}] {c.name:<60s} err={c.error:.2e}  tol={c.tol:.0e}"


# ---------------------------------------------------------------------------
# 1. Clean precision is tridiagonal and equals Sigma_0^{-1}
# ---------------------------------------------------------------------------

def check_clean_precision() -> list[Check]:
    out: list[Check] = []
    for K in (5, 9, 20):
        for alpha in (0.3, 0.7, 0.9, -0.5):
            Sigma_0 = ar1_covariance(K, alpha)
            Q_formula = ar1_precision_clean(K, alpha)
            err = float(np.max(np.abs(Q_formula - np.linalg.inv(Sigma_0))))
            out.append(Check(f"clean precision Q_0 (K={K}, a={alpha:+.2f})",
                             err, 1e-12))
            mask = np.abs(np.subtract.outer(np.arange(K), np.arange(K))) > 1
            off_band = float(np.max(np.abs(Q_formula[mask])))
            out.append(Check(f"Q_0 strictly tridiagonal (K={K}, a={alpha:+.2f})",
                             off_band, 0.0))
    return out


# ---------------------------------------------------------------------------
# 2. Sigma_t Q_t = I and spectral preservation
# ---------------------------------------------------------------------------

def check_inverse_consistency() -> list[Check]:
    out: list[Check] = []
    K = 12
    Sigma_0 = ar1_covariance(K, 0.8)
    for t in (0.001, 0.05, 0.4, 1.5, 5.0):
        Q_direct = precision_t(Sigma_0, t)
        prod = sigma_t(Sigma_0, t) @ Q_direct
        out.append(Check(f"Sigma_t Q_t = I (t={t})",
                         float(np.max(np.abs(prod - np.eye(K)))), 1e-10))
        Q_spec, U, _ = precision_t_spectral(Sigma_0, t)
        out.append(Check(f"Q_t direct vs spectral (t={t})",
                         float(np.max(np.abs(Q_spec - Q_direct))), 1e-10))
        diag_check = U.T @ Sigma_0 @ U
        off = diag_check - np.diag(np.diag(diag_check))
        out.append(Check(f"U diagonalises Sigma_0 (t={t})",
                         float(np.max(np.abs(off))), 1e-10))
    return out


# ---------------------------------------------------------------------------
# 3. Band-fill theorem: |Q_t[i,i+d]| ~ (2t)^{d-1} (NO 1/(d-1)! factor)
# ---------------------------------------------------------------------------

def check_band_fill() -> list[Check]:
    """Leading distance-d coefficient of Q_t at small t, from the resolvent
    Q_t = e^{2t} sum_m (-c_t)^m Q_0^{m+1}, c_t = e^{2t} - 1 = 2t + O(t^2):

        (Q_t)_{i,i+d} = (-1)^{d-1} (2t)^{d-1} (Q_0^d)_{i,i+d} + O(t^d).

    The published precision_lifecycle_summary.pdf eq. (12) carries a spurious
    1/(d-1)! factor that does NOT follow from this expansion; the version below
    (no factorial) is what the audit confirms.
    """
    K, alpha = 25, 0.9
    Sigma_0 = ar1_covariance(K, alpha)
    Q_0 = ar1_precision_clean(K, alpha)
    i = K // 2
    t_for_d = {1: 1e-5, 2: 1e-4, 3: 1e-4, 4: 1e-4, 5: 1e-4}
    Q_pow = {1: Q_0}
    for k in range(2, 6):
        Q_pow[k] = Q_pow[k - 1] @ Q_0
    out: list[Check] = []
    for d in range(1, 6):
        t = t_for_d[d]
        c_pred = ((-1) ** (d - 1)) * (2.0 ** (d - 1)) * Q_pow[d][i, i + d]
        c_emp = precision_t(Sigma_0, t)[i, i + d] / (t ** (d - 1))
        out.append(Check(f"band-fill leading coeff d={d} (no factorial)",
                         float(abs(c_emp - c_pred) / abs(c_pred)), 0.05))
    return out


# ---------------------------------------------------------------------------
# 4. Large-t return Q_t -> I
# ---------------------------------------------------------------------------

def check_large_t_return() -> list[Check]:
    K = 20
    Sigma_0 = ar1_covariance(K, 0.9)
    base = np.linalg.norm(Sigma_0 - np.eye(K), ord="fro")
    radius = float(np.max(np.abs(np.linalg.eigvalsh(Sigma_0) - 1.0)))
    out: list[Check] = []
    for t in (3.0, 6.0, 10.0):
        actual = np.linalg.norm(precision_t(Sigma_0, t) - np.eye(K), ord="fro")
        predicted = np.exp(-2.0 * t) * base
        tol = max(1e-7, 1.5 * radius * np.exp(-2.0 * t))
        out.append(Check(f"||Q_t - I||_F ~ e^-2t ||Sigma_0 - I||_F (t={t})",
                         float(abs(actual - predicted) / predicted), tol))
    return out


# ---------------------------------------------------------------------------
# 5. BP (Convention A) reproduces the matrix posterior and the exact score
# ---------------------------------------------------------------------------

def check_bp_vs_matrix() -> list[Check]:
    rng = np.random.default_rng(11)
    err_mu = err_var = err_score = 0.0
    n = 0
    for K in (2, 3, 5, 8, 16):
        for alpha in (0.2, 0.5, 0.9, -0.4):
            for t in (0.05, 0.3, 1.0, 3.0):
                Sigma_0 = ar1_covariance(K, alpha)
                x = rng.standard_normal(K)
                mu_bp, var_bp = bp_posterior(x, t, alpha)
                mu_m, Sigma_m = gaussian_posterior(x, t, Sigma_0, alpha)
                err_mu = max(err_mu, float(np.max(np.abs(mu_bp - mu_m))))
                err_var = max(err_var,
                              float(np.max(np.abs(var_bp - np.diag(Sigma_m)))))
                err_score = max(err_score, float(np.max(np.abs(
                    bp_score(x, t, alpha)
                    - joint_score_matrix(x, t, Sigma_0, alpha)))))
                n += 1
    return [
        Check(f"BP vs matrix posterior mean ({n} cases)", err_mu, 1e-10),
        Check(f"BP vs matrix posterior variance ({n} cases)", err_var, 1e-10),
        Check(f"BP vs matrix joint score ({n} cases)", err_score, 1e-10),
    ]


# ---------------------------------------------------------------------------
# 6. Tweedie matrix score equals -Q_t x
# ---------------------------------------------------------------------------

def check_tweedie_matrix_consistency() -> list[Check]:
    rng = np.random.default_rng(12)
    K = 8
    Sigma_0 = ar1_covariance(K, 0.6)
    err = 0.0
    for t in (0.1, 0.5, 2.0):
        for _ in range(10):
            x = rng.standard_normal(K)
            err = max(err, float(np.max(np.abs(
                joint_score_matrix(x, t, Sigma_0, 0.6)
                - joint_score_via_tweedie(x, t, Sigma_0, 0.6)))))
    return [Check("Tweedie matrix score consistency", err, 1e-12)]


# ---------------------------------------------------------------------------
# 7. AMP mean == exact mean == BP score (closure-independent score)
# ---------------------------------------------------------------------------

def check_amp_mean_equals_bp() -> list[Check]:
    rng = np.random.default_rng(20)
    err_mean = err_score = 0.0
    n = 0
    for K in (3, 5, 9):
        for alpha in (0.3, 0.6, 0.8):
            for t in (0.05, 0.3, 1.0, 3.0):
                x = rng.standard_normal(K)
                J, h = posterior_precision_field(x, t, alpha)
                m_amp = mean_iteration(J, h)[0]
                m_ex, _ = exact_marginals(J, h)
                err_mean = max(err_mean, float(np.max(np.abs(m_amp - m_ex))))
                err_score = max(err_score, float(np.max(np.abs(
                    amp_score(x, t, alpha) - bp_score(x, t, alpha)))))
                n += 1
    return [
        Check(f"AMP mean == exact mean ({n} cases)", err_mean, 1e-9),
        Check(f"AMP score == BP score ({n} cases)", err_score, 1e-9),
    ]


# ---------------------------------------------------------------------------
# 8. AMP variance: accurate on a dense graph, breakdown on the chain
# ---------------------------------------------------------------------------

def check_amp_variance_regimes() -> list[Check]:
    out: list[Check] = []
    rng = np.random.default_rng(21)

    # (a) Dense random SPD J: AMP variance accurate and beats mean field.
    N = 600
    A = rng.standard_normal((N, N)) / np.sqrt(N)
    J = np.eye(N) + 0.4 * (A + A.T) / np.sqrt(2.0)
    J += (abs(min(np.linalg.eigvalsh(J))) + 0.5) * np.eye(N)
    _, v_ex = exact_marginals(J, np.zeros(N))
    v_amp, _, ok = amp_variance(J)
    err_amp = float(np.max(np.abs(v_amp - v_ex))) if ok else np.inf
    err_mf = float(np.max(np.abs(mean_field_variance(J) - v_ex)))
    out.append(Check(f"AMP variance accurate on dense graph (N={N})",
                     err_amp, 0.05))
    out.append(Check("AMP variance beats mean field on dense graph",
                     err_amp, err_mf))   # pass iff err_amp <= err_mf

    # (b) Weak-coupling chain (small t): AMP variance still accurate.
    K, alpha = 9, 0.8
    x = rng.standard_normal(K)
    J, _ = posterior_precision_field(x, 0.05, alpha)
    _, v_ex = exact_marginals(J, np.zeros(K))
    v_amp, _, ok = amp_variance(J)
    err = float(np.max(np.abs(v_amp - v_ex))) if ok else np.inf
    out.append(Check("AMP variance accurate on chain at small t (t=0.05)",
                     err, 1e-2))

    # (c) Strong-coupling chain (larger t): AMP variance has NO fixed point.
    J, _ = posterior_precision_field(x, 1.0, alpha)
    _, _, ok = amp_variance(J)
    # Pass iff the closure correctly fails to converge (ok is False).
    out.append(Check("AMP variance breakdown on chain at t=1.0 (expected)",
                     0.0 if not ok else 1.0, 0.0))
    return out


# ---------------------------------------------------------------------------
# 9. Local (radius-r) BP converges to full BP; error monotone in r
# ---------------------------------------------------------------------------

def check_local_vs_full() -> list[Check]:
    rng = np.random.default_rng(22)
    out: list[Check] = []
    K, alpha = 21, 0.8
    Sigma_0 = ar1_covariance(K, alpha)
    x = rng.standard_normal(K)
    worst_full = 0.0
    worst_monotone = 0.0
    for t in (0.1, 0.5, 2.0):
        s_full = joint_score_matrix(x, t, Sigma_0, alpha)
        worst_full = max(worst_full, float(np.max(np.abs(
            local_score(x, t, alpha, K - 1) - s_full))))
        errs = [float(np.max(np.abs(local_score(x, t, alpha, r) - s_full)))
                for r in range(0, K)]
        # monotone non-increasing: max positive jump should be ~0
        jumps = [errs[r + 1] - errs[r] for r in range(len(errs) - 1)]
        worst_monotone = max(worst_monotone, max(jumps))
    out.append(Check("local BP radius=K-1 equals full score", worst_full, 1e-9))
    out.append(Check("local BP error monotone non-increasing in radius",
                     max(0.0, worst_monotone), 1e-9))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_all() -> int:
    checks: list[Check] = []
    checks += check_clean_precision()
    checks += check_inverse_consistency()
    checks += check_band_fill()
    checks += check_large_t_return()
    checks += check_bp_vs_matrix()
    checks += check_tweedie_matrix_consistency()
    checks += check_amp_mean_equals_bp()
    checks += check_amp_variance_regimes()
    checks += check_local_vs_full()

    n_total = len(checks)
    n_fail = sum(1 for c in checks if not c.passed)
    print("=" * 80)
    print(f"Gaussian BP / diffusion-score numerical audit  --  {n_total} checks")
    print("=" * 80)
    for c in checks:
        print(_fmt(c))
    print("-" * 80)
    print(f"PASSED {n_total - n_fail} / {n_total}  FAILED {n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
