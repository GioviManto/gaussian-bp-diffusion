"""
experiments.py
==============

Generates every figure used in main.tex and the companion notebook.  All
quantities are the audited closed forms from ar1_utils / bp_score / amp /
local_bp; nothing here is approximate beyond what each experiment studies.

Figures written to ../figures:
    fig_precision_lifecycle.png  -- heatmaps of Q_t = Sigma_t^{-1} vs t
    fig_band_fill.png            -- |Q_t[i,i+d]| ~ (2t)^{d-1} (log-log)
    fig_tridiag_loss.png         -- ||Sigma_t - Sigma_0|| and off-band mass vs t
    fig_spectral.png             -- eigenvalues / eigenvectors of Sigma_0, Sigma_t
    fig_local_vs_full.png        -- locality error vs radius and vs t
    fig_bp_vs_amp.png            -- AMP variance error vs t and existence map
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ar1_utils import (
    ar1_covariance, ar1_precision_clean, ou_params, precision_t, sigma_t,
    joint_score_matrix,
)
from amp import (
    amp_variance, exact_marginals, mean_field_variance,
    posterior_precision_field,
)
from local_bp import local_score, rms_truncation_error
from chain_formulas import (
    amp_bulk_variance, amp_critical_time, bulk_correlation_decay,
    bulk_params, bulk_variance_exact,
)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")

# --- house style -----------------------------------------------------------
BG = "#f7f4ef"; INK = "#1a1410"; RED = "#b5341a"; NAVY = "#1a3a5c"
GREEN = "#2f6b34"; GOLD = "#8a6d2e"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "serif", "font.size": 11, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.titlesize": 12, "axes.grid": True,
    "grid.alpha": 0.25, "grid.color": INK,
})


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------------------
# Exp B.1 -- precision lifecycle heatmaps
# ---------------------------------------------------------------------------

def fig_precision_lifecycle(K=15, alpha=0.8):
    Sigma_0 = ar1_covariance(K, alpha)
    ts = [0.0, 0.1, 0.5, 1.5, 5.0]
    fig, axes = plt.subplots(1, len(ts), figsize=(3.0 * len(ts), 3.1))
    Q0 = ar1_precision_clean(K, alpha)
    vmax = np.max(np.abs(Q0))
    for ax, t in zip(axes, ts):
        Q = Q0 if t == 0.0 else precision_t(Sigma_0, t)
        im = ax.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(rf"$t={t:g}$"); ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
    axes[0].set_ylabel(r"$Q_t=\Sigma_t^{-1}$")
    fig.suptitle(r"Lifecycle of the precision matrix $Q_t=\Sigma_t^{-1}$: "
                 r"tridiagonal at $t{=}0$, band fills for $t{>}0$, "
                 r"$\to I$ as $t\to\infty$", y=1.04)
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    _save(fig, "fig_precision_lifecycle.png")


# ---------------------------------------------------------------------------
# Exp B.2 -- band-fill scaling (no 1/(d-1)! factor)
# ---------------------------------------------------------------------------

def fig_band_fill(K=25, alpha=0.9):
    Sigma_0 = ar1_covariance(K, alpha)
    Q0 = ar1_precision_clean(K, alpha)
    i = K // 2
    ts = np.logspace(-5, -1, 40)
    Qpow = {1: Q0}
    for d in range(2, 6):
        Qpow[d] = Qpow[d - 1] @ Q0
    colors = [RED, NAVY, GREEN, GOLD, "#6b2356"]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for d, c in zip(range(1, 6), colors):
        vals = [abs(precision_t(Sigma_0, t)[i, i + d]) for t in ts]
        ax.loglog(ts, vals, "o", ms=3, color=c, label=f"$d={d}$ (measured)")
        pred = [abs(((-1) ** (d - 1)) * (2.0 * t) ** (d - 1)
                    * Qpow[d][i, i + d]) for t in ts]
        ax.loglog(ts, pred, "-", lw=1.2, color=c, alpha=0.7)
    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel(r"$|Q_t[i,\,i+d]|$")
    ax.set_title(r"Band fill: $|Q_t[i,i+d]|\sim(2t)^{\,d-1}|Q_0^{\,d}[i,i+d]|$"
                 "\n(solid = leading order, no $1/(d-1)!$ factor)")
    ax.legend(fontsize=9, framealpha=0.3)
    _save(fig, "fig_band_fill.png")


# ---------------------------------------------------------------------------
# Exp B.3 -- loss of tridiagonal structure vs diffusion time
# ---------------------------------------------------------------------------

def _offband_mass(M, bandwidth=1):
    K = M.shape[0]
    mask = np.abs(np.subtract.outer(np.arange(K), np.arange(K))) > bandwidth
    return np.linalg.norm(M[mask])


def fig_tridiag_loss(K=20, alpha=0.8):
    Sigma_0 = ar1_covariance(K, alpha)
    ts = np.logspace(-3, 1.3, 120)
    d_sigma, offband_Q = [], []
    for t in ts:
        St = sigma_t(Sigma_0, t)
        Qt = precision_t(Sigma_0, t)
        d_sigma.append(np.linalg.norm(St - Sigma_0, ord="fro"))
        offband_Q.append(_offband_mass(Qt) / np.linalg.norm(Qt, ord="fro"))
    fig, ax1 = plt.subplots(figsize=(6.4, 4.6))
    ax1.semilogx(ts, d_sigma, color=NAVY, lw=2,
                 label=r"$\|\Sigma_t-\Sigma_0\|_F$")
    ax1.set_xlabel(r"diffusion time $t$")
    ax1.set_ylabel(r"$\|\Sigma_t-\Sigma_0\|_F$", color=NAVY)
    ax1.tick_params(axis="y", labelcolor=NAVY)
    ax2 = ax1.twinx()
    ax2.semilogx(ts, offband_Q, color=RED, lw=2,
                 label=r"off-tridiagonal mass of $Q_t$")
    ax2.set_ylabel(r"$\|Q_t^{\mathrm{off\text{-}band}}\|_F/\|Q_t\|_F$",
                   color=RED)
    ax2.tick_params(axis="y", labelcolor=RED); ax2.grid(False)
    ax1.set_title(r"Loss then recovery of structure: $\Sigma_t$ departs from "
                  r"$\Sigma_0$;"
                  "\n"
                  r"$Q_t$ fills out of tridiagonal, then collapses back to "
                  r"diagonal ($\to I$)")
    _save(fig, "fig_tridiag_loss.png")


# ---------------------------------------------------------------------------
# Exp B.4 -- spectral decomposition (shared eigenbasis, eigenvalue flow)
# ---------------------------------------------------------------------------

def fig_spectral(K=24, alpha=0.85):
    Sigma_0 = ar1_covariance(K, alpha)
    lam0, U = np.linalg.eigh(Sigma_0)
    order = np.argsort(lam0)[::-1]
    lam0 = lam0[order]; U = U[:, order]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.2))
    for t, c in [(0.0, INK), (0.2, NAVY), (1.0, RED), (3.0, GOLD)]:
        mu, Delta = ou_params(t) if t > 0 else (1.0, 0.0)
        lam_t = mu * mu * lam0 + Delta
        axL.plot(range(1, K + 1), lam_t, "o-", ms=3, color=c, label=f"$t={t:g}$")
    axL.set_xlabel("mode index"); axL.set_ylabel(r"eigenvalue $\lambda_i(\Sigma_t)$")
    axL.axhline(1.0, ls="--", color=INK, alpha=0.5)
    axL.set_title(r"Eigenvalues flow $\lambda_i(\Sigma_t)=e^{-2t}\lambda_i(\Sigma_0)"
                  r"+\Delta_t\to1$"); axL.legend(fontsize=9, framealpha=0.3)
    for j, c in zip([0, 1, 2], [RED, NAVY, GREEN]):
        axR.plot(range(1, K + 1), U[:, j], "o-", ms=3, color=c,
                 label=f"mode {j + 1}")
    axR.set_xlabel("frame index $k$"); axR.set_ylabel("eigenvector component")
    axR.set_title(r"Eigenvectors of $\Sigma_0$ (shared by $\Sigma_t,Q_t$):"
                  "\nnear-Fourier modes (Toeplitz)")
    axR.legend(fontsize=9, framealpha=0.3)
    _save(fig, "fig_spectral.png")


# ---------------------------------------------------------------------------
# Exp A -- local (radius-r) vs full BP
# ---------------------------------------------------------------------------

def fig_local_vs_full(K=121, alpha=0.8):
    """Exact (deterministic) RMS truncation error of the radius-r local
    estimator at the centre frame, against the predicted geometric decay
    q^r with q = bulk_correlation_decay(alpha, t)."""
    radii = list(range(0, 13))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.2))
    for t, c in [(0.1, RED), (0.5, NAVY), (2.0, GREEN)]:
        rms = [rms_truncation_error(K, alpha, t, K // 2, r) for r in radii]
        axL.semilogy(radii, rms, "o", ms=5, color=c, label=f"$t={t:g}$")
        q = bulk_correlation_decay(alpha, t)
        # anchor the predicted slope at r=2
        pred = [rms[2] * q ** (r - 2) for r in radii]
        axL.semilogy(radii, pred, "--", lw=1.3, color=c, alpha=0.8)
    axL.set_xlabel(r"local radius $r$ (window $x_{k-r},\dots,x_{k+r}$)")
    axL.set_ylabel(r"exact RMS truncation error RMS$_r$")
    axL.set_title("Locality error decays exactly as $q^{\\,r}$\n"
                  "(dashed: predicted slope)")
    axL.legend(fontsize=9, framealpha=0.3)

    ts = np.logspace(-1.6, 0.9, 50)
    for r, c in [(1, RED), (2, NAVY), (4, GREEN)]:
        rms = [rms_truncation_error(K, alpha, t, K // 2, r) for t in ts]
        axR.semilogy(ts, rms, "-", lw=2, color=c, label=f"$r={r}$")
    axR.set_xscale("log")
    axR.set_xlabel(r"diffusion time $t$")
    axR.set_ylabel(r"RMS$_r$ at centre frame")
    axR.set_title("Truncation error vs diffusion time:\n"
                  "worst at intermediate $t$")
    axR.legend(fontsize=9, framealpha=0.3)
    _save(fig, "fig_local_vs_full.png")


# ---------------------------------------------------------------------------
# Bulk variance closed forms: exact / BP vs AMP vs mean field
# ---------------------------------------------------------------------------

def fig_bulk_variance():
    """Left: bulk posterior variance vs t at alpha = 0.8 -- exact (= BP),
    AMP closed form (ends at t_c), mean field.  Right: same at alpha = 0.3
    (below alpha_c = sqrt(2)-1: AMP never breaks down)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, alpha in zip(axes, (0.8, 0.3)):
        ts = np.logspace(-2, 1.0, 300)
        v_ex = [bulk_variance_exact(alpha, t) for t in ts]
        v_amp = [amp_bulk_variance(alpha, t) for t in ts]
        v_mf = [1.0 / bulk_params(alpha, t)[0] for t in ts]
        ax.semilogx(ts, v_ex, "-", lw=2.4, color=NAVY,
                    label=r"exact $=$ BP: $1/\sqrt{J_d^2-4\beta^2}$")
        ax.semilogx(ts, v_amp, "-", lw=2.0, color=RED,
                    label=r"AMP: $(J_d-\sqrt{J_d^2-8\beta^2})/4\beta^2$")
        ax.semilogx(ts, v_mf, ":", lw=1.8, color=GOLD,
                    label=r"mean field: $1/J_d$")
        tc = amp_critical_time(alpha)
        if np.isfinite(tc):
            ax.axvline(tc, color=RED, lw=1.0, ls="--", alpha=0.8)
            ax.axvspan(tc, ts[-1], color=RED, alpha=0.07)
            ax.text(tc * 1.1, 0.05, r"$t_c$: AMP fixed point"
                    "\nceases to exist", color=RED, fontsize=9)
            ax.set_title(rf"$\alpha={alpha} > \alpha_c$: AMP breaks down "
                         rf"at $t_c={tc:.3f}$")
        else:
            ax.set_title(rf"$\alpha={alpha} < \alpha_c=\sqrt{{2}}-1$: "
                         "AMP exists at every $t$")
        ax.set_xlabel(r"diffusion time $t$")
        ax.set_ylabel("bulk posterior variance")
        ax.legend(fontsize=8.5, framealpha=0.3, loc="upper left")
    _save(fig, "fig_bulk_variance.png")


# ---------------------------------------------------------------------------
# BP vs AMP -- variance accuracy and existence of the AMP fixed point
# ---------------------------------------------------------------------------

def fig_bp_vs_amp(K=9, alpha=0.8, seed=21):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(K)
    ts = np.logspace(-2, 0.6, 60)
    var_err, mean_err, exists = [], [], []
    for t in ts:
        J, h = posterior_precision_field(x, t, alpha)
        m_ex, v_ex = exact_marginals(J, h)
        v_amp, _, ok = amp_variance(J)
        from amp import mean_iteration
        m_amp = mean_iteration(J, h)[0]
        mean_err.append(np.max(np.abs(m_amp - m_ex)))
        exists.append(ok)
        var_err.append(np.max(np.abs(v_amp - v_ex)) if ok else np.nan)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 4.2))
    axL.semilogx(ts, var_err, "o-", ms=3, color=RED,
                 label="AMP variance error")
    axL.semilogx(ts, mean_err, "s-", ms=3, color=NAVY,
                 label="AMP mean error (= BP score)")
    # shade the breakdown region
    brk = [not e for e in exists]
    if any(brk):
        t_brk = ts[np.argmax(brk)]
        axL.axvspan(t_brk, ts[-1], color=RED, alpha=0.08)
        axL.text(t_brk * 1.05, 1e-9, "AMP variance\nbreakdown",
                 color=RED, fontsize=9, va="bottom")
    axL.set_xlabel(r"diffusion time $t$")
    axL.set_ylabel("max error vs exact")
    axL.set_yscale("log")
    axL.set_title(f"BP vs AMP on the chain ($K={K},\\ \\alpha={alpha}$):\n"
                  "means coincide (same score); AMP variance fails")
    axL.legend(fontsize=9, framealpha=0.3)

    # existence map over (alpha, t)
    alphas = np.linspace(0.0, 0.95, 48)
    tgrid = np.logspace(-2, 0.8, 48)
    Z = np.zeros((len(alphas), len(tgrid)))
    xb = rng.standard_normal(K)
    for ia, a in enumerate(alphas):
        for it, t in enumerate(tgrid):
            J, _ = posterior_precision_field(xb, t, a)
            _, _, ok = amp_variance(J)
            Z[ia, it] = 1.0 if ok else 0.0
    im = axR.pcolormesh(tgrid, alphas, Z, cmap="RdYlGn", shading="auto",
                        vmin=0, vmax=1)
    # overlay the EXACT analytic phase boundary t_c(alpha)
    a_bdry = np.linspace(np.sqrt(2.0) - 1.0 + 1e-4, 0.95, 200)
    t_bdry = [amp_critical_time(a) for a in a_bdry]
    axR.plot(t_bdry, a_bdry, "-", color=INK, lw=2.2,
             label=r"exact boundary $J_d=2\sqrt{2}\,|\beta|$")
    axR.axhline(np.sqrt(2.0) - 1.0, color=INK, lw=1.2, ls="--")
    axR.text(1.4e-2, np.sqrt(2.0) - 1.0 + 0.015,
             r"$\alpha_c=\sqrt{2}-1$", fontsize=10, color=INK)
    axR.set_xscale("log")
    axR.set_xlabel(r"diffusion time $t$"); axR.set_ylabel(r"AR(1) coupling $\alpha$")
    axR.set_title("Where the AMP variance fixed point exists\n"
                  "(green = iteration converges; black = exact boundary)")
    axR.legend(fontsize=9, framealpha=0.6, loc="lower left")
    axR.grid(False)
    _save(fig, "fig_bp_vs_amp.png")


def main():
    print("Generating figures into", os.path.abspath(FIGDIR))
    fig_precision_lifecycle()
    fig_band_fill()
    fig_tridiag_loss()
    fig_spectral()
    fig_local_vs_full()
    fig_bp_vs_amp()
    fig_bulk_variance()
    print("done.")


if __name__ == "__main__":
    main()
