"""
experiments.py -- the two experiments of the note, with deterministic errors.

Experiment 1 (truncation, two faces of the same question):
  1a. truncate the exact score MATRIX Q_t to a band  (Jerome, 31:27)
  1b. truncate the MESSAGE RANGE to r hops           (Jerome, 31:48)
  Both errors are relative root-mean-square over the data law x ~ P_t,
  computed in closed form (no sampling).

Experiment 2 (AMP vs the exact score):
  the per-node mean update reproduces the exact score at every t;
  the per-node variance closure is accurate at small t and loses its
  fixed point at larger t.

Figures -> ./figures/, key numbers printed for the note.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bp_gaussian import (
    banded_score_matrix, relative_score_error, score_matrix_of_local_bp,
    posterior_natural_params, amp_mean, amp_variance, channel,
    clean_precision, noisy_covariance, exact_score, amp_score,
)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGDIR, exist_ok=True)

BG = "#fbfaf7"; INK = "#22201c"; RED = "#b5341a"; NAVY = "#1a3a5c"
GREEN = "#2f6b34"; GOLD = "#8a6d2e"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "serif", "font.size": 11, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.grid": True, "grid.alpha": 0.25,
})


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/" + name)


# ---------------------------------------------------------------------------
# Experiment 1: truncation (matrix band / message range), error vs t
# ---------------------------------------------------------------------------

K, alpha = 40, 0.8
ts = np.logspace(-2, 1.0, 60)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.3))

for b, c in [(1, RED), (2, NAVY), (4, GREEN)]:
    errs = [relative_score_error(banded_score_matrix(K, alpha, t, b),
                                 K, alpha, t) for t in ts]
    axA.loglog(ts, errs, "-", lw=2, color=c, label=f"band $b={b}$")
axA.set_xlabel("diffusion time $t$")
axA.set_ylabel("relative RMS score error")
axA.set_title("Exp. 1a — truncate the score matrix\n"
              "$Q_t \\to$ banded ($b{=}1$: tridiagonal)")
axA.legend(framealpha=0.3, fontsize=9)

for r, c in [(1, RED), (2, NAVY), (4, GREEN)]:
    errs = [relative_score_error(score_matrix_of_local_bp(K, alpha, t, r),
                                 K, alpha, t) for t in ts]
    axB.loglog(ts, errs, "-", lw=2, color=c, label=f"range $r={r}$")
axB.set_xlabel("diffusion time $t$")
axB.set_ylabel("relative RMS score error")
axB.set_title("Exp. 1b — truncate the message range\n"
              "(messages travel at most $r$ hops)")
axB.legend(framealpha=0.3, fontsize=9)
save(fig, "exp1_truncation.png")

# numbers for the note
print("\nExp.1 numbers (K=40, alpha=0.8), relative RMS errors:")
for t in (0.05, 0.3, 1.0, 3.0):
    e1a = relative_score_error(banded_score_matrix(K, alpha, t, 1), K, alpha, t)
    e1b = relative_score_error(score_matrix_of_local_bp(K, alpha, t, 1),
                               K, alpha, t)
    print(f"  t={t:<5}  banded b=1: {e1a:.4f}    message range r=1: {e1b:.4f}")

# error vs band/range at fixed t
fig, ax = plt.subplots(figsize=(6.4, 4.3))
bands = list(range(0, 13))
for t, c in [(0.1, RED), (0.5, NAVY), (2.0, GREEN)]:
    errs_b = [relative_score_error(banded_score_matrix(K, alpha, t, b),
                                   K, alpha, t) for b in bands]
    errs_r = [relative_score_error(score_matrix_of_local_bp(K, alpha, t, b),
                                   K, alpha, t) for b in bands]
    ax.semilogy(bands, errs_b, "o-", ms=4, lw=1.6, color=c,
                label=f"banded matrix, $t={t}$")
    ax.semilogy(bands, errs_r, "s--", ms=4, lw=1.4, color=c, alpha=0.75,
                label=f"message range, $t={t}$")
ax.set_xlabel("band $b$  /  message range $r$")
ax.set_ylabel("relative RMS score error")
ax.set_title("Exp. 1 — both truncations decay geometrically;\n"
             "the rate worsens with diffusion time")
ax.legend(framealpha=0.3, fontsize=8.5)
save(fig, "exp1_vs_radius.png")

# ---------------------------------------------------------------------------
# Experiment 2: AMP -- score exact, variance degrades then breaks
# ---------------------------------------------------------------------------

K2, alpha2 = 12, 0.8
rng = np.random.default_rng(11)
x2 = rng.standard_normal(K2)
ts2 = np.logspace(-2, 0.7, 50)

score_err, var_err, has_fp = [], [], []
for t in ts2:
    J, h = posterior_natural_params(x2, alpha2, t)
    m, _ = amp_mean(J, h)
    Jinv = np.linalg.inv(J)
    score_err.append(np.max(np.abs(m - Jinv @ h)))
    V, ok = amp_variance(J)
    has_fp.append(ok)
    var_err.append(np.max(np.abs(V - np.diag(Jinv))) if ok else np.nan)

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.loglog(ts2, score_err, "s-", ms=3, lw=1.6, color=NAVY,
          label="mean error  (= score error, all $t$)")
ax.loglog(ts2, var_err, "o-", ms=3, lw=1.6, color=RED,
          label="variance error (while a fixed point exists)")
if not all(has_fp):
    t_break = ts2[has_fp.index(False)]
    ax.axvspan(t_break, ts2[-1], color=RED, alpha=0.08)
    ax.axvline(t_break, color=RED, lw=1, ls="--")
    ax.text(t_break * 1.07, 1e-10, "variance update loses\nits fixed point",
            color=RED, fontsize=9)
ax.set_xlabel("diffusion time $t$")
ax.set_ylabel("max abs error vs exact")
ax.set_title(f"Exp. 2 — AMP on the chain ($K={K2}$, $\\alpha={alpha2}$):\n"
             "the mean (hence the score) is exact; the variance closure is not")
ax.legend(framealpha=0.3, fontsize=9)
save(fig, "exp2_amp.png")

print("\nExp.2 numbers (K=12, alpha=0.8):")
for t in (0.05, 0.2, 0.5, 1.0):
    J, h = posterior_natural_params(x2, alpha2, t)
    m, _ = amp_mean(J, h)
    Jinv = np.linalg.inv(J)
    V, ok = amp_variance(J)
    vtxt = (f"var err {np.max(np.abs(V - np.diag(Jinv))):.2e}" if ok
            else "variance: NO fixed point")
    print(f"  t={t:<5} mean err {np.max(np.abs(m - Jinv @ h)):.1e}   {vtxt}")

# breakdown time as a function of alpha (bisection on the iteration)
print("\nAMP variance breakdown time (bisection, K=200):")
for a in (0.3, 0.5, 0.7, 0.9):
    lo, hi = 1e-3, 30.0
    x_l = rng.standard_normal(200)
    J_hi, _ = posterior_natural_params(x_l, a, hi)
    _, ok_hi = amp_variance(J_hi)
    if ok_hi:
        print(f"  alpha={a}: no breakdown up to t={hi}")
        continue
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        J, _ = posterior_natural_params(x_l, a, mid)
        _, ok = amp_variance(J)
        lo, hi = (mid, hi) if ok else (lo, mid)
    print(f"  alpha={a}: t_break ~ {np.sqrt(lo*hi):.4f}")

# ---------------------------------------------------------------------------
# Worked example for the note: K = 4, every message printed
# ---------------------------------------------------------------------------

print("\nWorked example for the note: K=4, alpha=0.6, t=0.3,"
      " x=(0.9,-0.2,0.5,-1.1)")
from bp_gaussian import bp_messages, bp_posterior, bp_score, _evidence
K4, a4, t4 = 4, 0.6, 0.3
x4 = np.array([0.9, -0.2, 0.5, -1.1])
mu4, D4 = channel(t4)
print(f"  mu=e^-t={mu4:.6f}  D_t={D4:.6f}  s2=1-a^2={1-a4*a4:.2f}"
      f"  evid.prec mu^2/D={mu4*mu4/D4:.6f}")
mf, vf, mb, vb = bp_messages(x4, a4, t4)
for k in range(K4):
    me, ve = _evidence(x4[k], t4)
    print(f"  k={k}: fwd=({mf[k]:+.6f},{vf[k]:.6f})  evid=({me:+.6f},{ve:.6f})"
          f"  bwd=({mb[k]:+.6f},{'inf' if np.isinf(vb[k]) else f'{vb[k]:.6f}'})")
m4, v4 = bp_posterior(x4, a4, t4)
print("  posterior means:", np.array2string(m4, precision=6))
print("  posterior vars :", np.array2string(v4, precision=6))
print("  BP score       :", np.array2string(bp_score(x4, a4, t4), precision=6))
print("  exact score    :", np.array2string(exact_score(x4, a4, t4), precision=6))
