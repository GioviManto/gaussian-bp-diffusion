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
   10. K = 2 fully explicit closed forms (Sigma_t, Q_t, score).
   11. Homogeneous-chain (bulk) closed forms: exact bulk variance
       1/sqrt(J_d^2-4beta^2); BP cavity fixed point exists for all (alpha,t)
       and recombines to the exact variance; AMP bulk variance formula and
       the existence boundary J_d >= 2*sqrt(2)*|beta|.

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
from local_bp import local_score, rms_truncation_error
from chain_formulas import (
    amp_bulk_variance,
    amp_critical_time,
    amp_fixed_point_exists,
    amp_weak_coupling_error,
    bp_cavity_precision,
    bulk_correlation_decay,
    bulk_covariance,
    bulk_params,
    bulk_variance_exact,
    k2_precision_t,
    k2_score,
    k2_sigma_t,
)


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
# 10. K = 2 fully explicit closed forms
# ---------------------------------------------------------------------------

def check_k2_closed_forms() -> list[Check]:
    rng = np.random.default_rng(30)
    err_sig = err_prec = err_score = 0.0
    for alpha in (0.3, 0.7, 0.95, -0.5):
        Sigma_0 = ar1_covariance(2, alpha)
        for t in (0.01, 0.2, 1.0, 4.0):
            err_sig = max(err_sig, float(np.max(np.abs(
                k2_sigma_t(alpha, t) - sigma_t(Sigma_0, t)))))
            err_prec = max(err_prec, float(np.max(np.abs(
                k2_precision_t(alpha, t) - precision_t(Sigma_0, t)))))
            for _ in range(5):
                x = rng.standard_normal(2)
                err_score = max(err_score, float(np.max(np.abs(
                    k2_score(x, alpha, t)
                    - joint_score_matrix(x, t, Sigma_0, alpha)))))
    return [
        Check("K=2 closed-form Sigma_t (16 configs)", err_sig, 1e-12),
        Check("K=2 closed-form Q_t (16 configs)", err_prec, 1e-12),
        Check("K=2 closed-form score (80 cases)", err_score, 1e-12),
    ]


# ---------------------------------------------------------------------------
# 11. Homogeneous-chain (bulk) closed forms: exact variance, BP cavity,
#     AMP fixed point and existence boundary
# ---------------------------------------------------------------------------

def check_bulk_closed_forms() -> list[Check]:
    rng = np.random.default_rng(31)
    out: list[Check] = []

    # (a) Exact bulk variance 1/sqrt(J_d^2-4beta^2) vs brute-force inverse
    #     at the centre of a long chain (boundary effects decay geometrically).
    K = 400
    err = 0.0
    for alpha in (0.2, 0.5, 0.8, 0.95, -0.6):
        for t in (0.05, 0.3, 1.0, 3.0):
            J, _ = posterior_precision_field(rng.standard_normal(K), t, alpha)
            v_bf = float(np.linalg.inv(J)[K // 2, K // 2])
            v_cf = bulk_variance_exact(alpha, t)
            err = max(err, abs(v_bf - v_cf) / v_cf)
    out.append(Check("bulk variance = 1/sqrt(J_d^2-4b^2) (20 configs, K=400)",
                     err, 1e-9))

    # (b) BP cavity fixed point: existence margin J_d - 2|beta| > 0 on a grid,
    #     and the recombination identity J_d - 2 b^2/lambda* = sqrt(J_d^2-4b^2).
    margin = np.inf
    err_recomb = 0.0
    for alpha in np.linspace(-0.99, 0.99, 67):
        for t in np.logspace(-3, 1.5, 30):
            J_d, beta = bulk_params(alpha, t)
            margin = min(margin, J_d - 2.0 * abs(beta))
            lam = bp_cavity_precision(alpha, t)
            lhs = J_d - 2.0 * beta * beta / lam
            rhs = np.sqrt(J_d * J_d - 4.0 * beta * beta)
            err_recomb = max(err_recomb, abs(lhs - rhs) / rhs)
    out.append(Check("BP cavity exists for all (alpha,t): min margin > 0",
                     0.0 if margin > 0 else 1.0, 0.0))
    out.append(Check("BP recombination identity (2010-point grid)",
                     err_recomb, 1e-12))

    # (c) AMP bulk variance formula vs the self-consistent iteration,
    #     at the centre of a long chain, where the fixed point exists.
    K = 400
    err = 0.0
    n_cmp = 0
    for alpha in (0.3, 0.6, 0.8):
        for t in (0.02, 0.05, 0.1, 0.2):
            v_cf = amp_bulk_variance(alpha, t)
            if np.isnan(v_cf):
                continue
            J, _ = posterior_precision_field(rng.standard_normal(K), t, alpha)
            v_it, _, ok = amp_variance(J)
            if not ok:
                continue
            err = max(err, abs(float(v_it[K // 2]) - v_cf) / v_cf)
            n_cmp += 1
    out.append(Check(f"AMP bulk variance closed form ({n_cmp} configs, K=400)",
                     err, 1e-8))

    # (d) AMP existence boundary J_d >= 2*sqrt(2)*|beta| predicts whether the
    #     iteration on a long finite chain converges (180-point scan).
    K = 200
    n_disagree = 0
    n_total = 0
    for alpha in (0.3, 0.5, 0.7, 0.85, 0.95):
        for t in np.logspace(-2.5, 1.0, 36):
            predicted = amp_fixed_point_exists(alpha, t)
            J, _ = posterior_precision_field(rng.standard_normal(K), t, alpha)
            _, _, ok = amp_variance(J)
            n_disagree += (predicted != ok)
            n_total += 1
    out.append(Check(f"AMP existence boundary vs iteration ({n_total} pts)",
                     float(n_disagree), 0.0))

    # (e) Full bulk posterior covariance (J^{-1})_{i,i+d} = q^d V_exact,
    #     q = bulk_correlation_decay, vs brute-force inversion.
    K = 400
    err = 0.0
    for alpha in (0.5, 0.8, 0.95):
        for t in (0.05, 0.5, 2.0):
            J, _ = posterior_precision_field(rng.standard_normal(K), t, alpha)
            Jinv = np.linalg.inv(J)
            i = K // 2
            for d in range(0, 6):
                pred = bulk_covariance(alpha, t, d)
                err = max(err, abs(float(Jinv[i, i + d]) - pred) / pred)
    out.append(Check("bulk covariance (J^-1)_{i,i+d} = q^d V (9 configs, d<=5)",
                     err, 1e-8))

    # (f) Weak-coupling AMP error: (V_amp - V_exact) / (2 b^4 / J_d^5) -> 1.
    worst = 0.0
    for alpha in (0.2, 0.3, 0.4):
        for t in (0.01, 0.02):
            ratio = ((amp_bulk_variance(alpha, t)
                      - bulk_variance_exact(alpha, t))
                     / amp_weak_coupling_error(alpha, t))
            worst = max(worst, abs(ratio - 1.0))
    out.append(Check("AMP weak-coupling error = 2b^4/J_d^5 (ratio->1)",
                     worst, 0.02))

    # (g) Locality error decay rate: the exact RMS truncation error of the
    #     radius-r estimator decays in r with slope log q (within 1%).
    K, alpha = 121, 0.8
    worst = 0.0
    for t in (0.1, 0.5, 2.0):
        rms = [rms_truncation_error(K, alpha, t, K // 2, r)
               for r in range(0, 15)]
        rs = np.arange(2, 14)
        slope = float(np.polyfit(rs, np.log([rms[r] for r in rs]), 1)[0])
        worst = max(worst, abs(slope / np.log(bulk_correlation_decay(alpha, t))
                               - 1.0))
    out.append(Check("locality RMS error decay slope = log q (3 t's, 1%)",
                     worst, 0.01))

    # (h) Exact AMP breakdown time t_c(alpha) and the critical coupling
    #     alpha_c = sqrt(2)-1: bisection of the iteration's flip point on a
    #     long chain must match the closed form; below alpha_c the iteration
    #     must converge even at very large t.
    K = 300
    worst = 0.0
    for alpha in (0.5, 0.7):
        tc = amp_critical_time(alpha)
        lo, hi = 1e-3, 20.0
        for _ in range(40):
            mid = float(np.sqrt(lo * hi))
            J, _ = posterior_precision_field(rng.standard_normal(K), mid, alpha)
            _, _, ok = amp_variance(J)
            lo, hi = (mid, hi) if ok else (lo, mid)
        worst = max(worst, abs(float(np.sqrt(lo * hi)) - tc) / tc)
    out.append(Check("AMP breakdown time t_c closed form (bisection, 0.5%)",
                     worst, 5e-3))
    ok_below = True
    for alpha in (0.2, 0.41):
        J, _ = posterior_precision_field(rng.standard_normal(K), 50.0, alpha)
        _, _, ok = amp_variance(J)
        ok_below = ok_below and ok and np.isinf(amp_critical_time(alpha))
    out.append(Check("below alpha_c = sqrt(2)-1 AMP never breaks down",
                     0.0 if ok_below else 1.0, 0.0))
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
    checks += check_k2_closed_forms()
    checks += check_bulk_closed_forms()

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
