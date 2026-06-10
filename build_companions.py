"""Build and execute the three companion notebooks.

    01_model_and_score.ipynb       -- mirrors main.pdf sections 2-6
    02_belief_propagation.ipynb    -- mirrors main.pdf sections 7-10
    03_amp_and_experiments.ipynb   -- mirrors main.pdf sections 11-14 + audit

Each notebook is self-contained (it only needs `code/` on the path and the
figures in `figures/`), heavily commented, and EXECUTED here so that the
committed .ipynb files always show real output.  If any cell raises, this
script fails -- the notebooks share the audit's "no unverified claims" rule.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient
import os

HERE = os.path.dirname(os.path.abspath(__file__))
md, co = new_markdown_cell, new_code_cell

PREAMBLE = """import sys
sys.path.insert(0, 'code')
import numpy as np
np.set_printoptions(precision=6, suppress=True)"""


def build(name: str, cells: list) -> None:
    nb = new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'name': 'python3',
                                 'display_name': 'Python 3',
                                 'language': 'python'}
    print(f'executing {name} ...')
    NotebookClient(nb, timeout=600,
                   resources={'metadata': {'path': HERE}}).execute()
    out = os.path.join(HERE, name)
    with open(out, 'w') as f:
        nbf.write(nb, f)
    print(f'  wrote {name}')


# ===========================================================================
# 01 -- model and score
# ===========================================================================

c1 = []
c1.append(md(r"""# 01 — The model and the joint score

Companion to **sections 2–6 of `main.pdf`** (Gaussian AR(1) + OU diffusion).

A sequence of $K$ frames follows a stationary Gaussian AR(1) chain
$a_{k+1}=\alpha a_k+\eta_k$, $\eta_k\sim\mathcal N(0,1-\alpha^2)$, so that
$\mathrm{Var}(a_k)=1$ for every $k$.  Each frame is independently corrupted by
the variance-preserving OU channel
$x_k = e^{-t}a_k+\xi_k$, $\xi_k\sim\mathcal N(0,\Delta_t)$, $\Delta_t=1-e^{-2t}$.

We verify, numerically and at machine precision:
1. the clean covariance $\Sigma_0[i,j]=\alpha^{|i-j|}$ and its **tridiagonal** inverse $Q_0$;
2. the noised law $\Sigma_t=e^{-2t}\Sigma_0+\Delta_t I$;
3. the joint score **two ways** — direct $S=-\Sigma_t^{-1}x$ vs Tweedie
   $S_k=(e^{-t}\langle a_k\rangle_{a|x}-x_k)/\Delta_t$ — and that they agree;
4. the fully explicit $K=2$ case, where the inter-frame coupling is the single
   number $r=\alpha e^{-2t}$.

*Rule of the project: nothing is claimed without a numerical check. The full
gate is `code/numerical_audit.py` (72/72 passing); this notebook re-derives
the model-level facts interactively.*"""))
c1.append(co(PREAMBLE + """
from ar1_utils import (ar1_covariance, ar1_precision_clean, ou_params, sigma_t,
                       precision_t, joint_score_matrix, joint_score_via_tweedie,
                       gaussian_posterior)
print('model code loaded')"""))

c1.append(md(r"""## 1. Clean joint law: dense covariance, tridiagonal precision

The covariance is dense Toeplitz ($\alpha^{|i-j|}$: correlations never vanish),
but its inverse is **exactly tridiagonal** — the algebraic shadow of the Markov
property.  Interior diagonal $(1+\alpha^2)/\sigma_\eta^2$, boundaries
$1/\sigma_\eta^2$, off-diagonal $-\alpha/\sigma_\eta^2$
(`main.pdf` Prop. 5)."""))
c1.append(co("""K, alpha = 6, 0.8
Sigma0 = ar1_covariance(K, alpha)
Q0 = ar1_precision_clean(K, alpha)            # the closed-form tridiagonal
print('Sigma_0 (dense Toeplitz):\\n', Sigma0)
print('\\nQ0 from the closed form:\\n', Q0)
print('\\nmax |Q0 - inv(Sigma_0)|      =', np.max(np.abs(Q0 - np.linalg.inv(Sigma0))))
mask = np.abs(np.subtract.outer(np.arange(K), np.arange(K))) > 1
print('max |off-tridiagonal entries| =', np.max(np.abs(Q0[mask])), ' (exactly zero)')"""))

c1.append(md(r"""## 2. Noised joint law

$x=e^{-t}a+\xi$ with independent noise, so
$\Sigma_t=e^{-2t}\Sigma_0+\Delta_t I$.  The diagonal stays $1$ for all $t$
(variance preservation): the channel trades signal for noise at constant total
variance."""))
c1.append(co("""t = 0.4
St = sigma_t(Sigma0, t)
print('Sigma_t diagonal (should be all 1):', np.diag(St))
print('Sigma_t[0,1] =', St[0,1], ' = alpha e^{-2t} =', alpha*np.exp(-2*t))
Qt = precision_t(Sigma0, t)
print('\\nmax |Sigma_t @ Q_t - I| =', np.max(np.abs(St @ Qt - np.eye(K))))"""))

c1.append(md(r"""## 3. The score, two ways

**Direct**: $S(x,t)=-\Sigma_t^{-1}x$ (one matrix inversion).

**Bayesian (Tweedie, `main.pdf` Th. 7)**:
$S_k=(e^{-t}\,\mathbb E[a_k\,|\,x]-x_k)/\Delta_t$ — compute the posterior mean
of the clean frame given the **whole** noisy sequence, then rescale.

Two completely different computational routes; they must give the same vector.
This is a real end-to-end check of every formula above."""))
c1.append(co("""rng = np.random.default_rng(0)
x = rng.standard_normal(K)
S_direct  = joint_score_matrix(x, t, Sigma0, alpha)
S_tweedie = joint_score_via_tweedie(x, t, Sigma0, alpha)
print('S (direct)  =', S_direct)
print('S (Tweedie) =', S_tweedie)
print('max abs difference =', np.max(np.abs(S_direct - S_tweedie)))"""))

c1.append(md(r"""## 4. $K=2$: everything explicit

With unit variances ('main.pdf' section 6):

$$\Sigma_t=\begin{pmatrix}1&r\\ r&1\end{pmatrix},\qquad
S_0=-\frac{x_0-r\,x_1}{1-r^2},\qquad r=\alpha e^{-2t}.$$

The cross term $-r\,x_1$ in $S_0$ is the **inter-frame coupling**: the denoiser
of frame 0 borrows evidence from frame 1.  The per-frame *marginal* score is
just $-x_0$ at every $t$ (each $x_k$ is $\mathcal N(0,1)$ by variance
preservation) — it knows nothing about the neighbour.  This tiny example is
why the *joint* score is the right object (Mézard's correction, `main.pdf`
Remark 1)."""))
c1.append(co("""from chain_formulas import k2_sigma_t, k2_precision_t, k2_score

alpha2 = 0.7
for t2 in (0.05, 0.5, 2.0):
    r = alpha2*np.exp(-2*t2)
    x2 = np.array([1.0, -0.5])
    S_closed  = k2_score(x2, alpha2, t2)
    S_generic = joint_score_matrix(x2, t2, ar1_covariance(2, alpha2), alpha2)
    print(f't={t2:<4}  r=alpha e^-2t={r:.4f}   S_closed={S_closed}'
          f'   max|closed-generic|={np.max(np.abs(S_closed-S_generic)):.2e}')
print('\\nmarginal score of x_0 would be -x_0 = -1.0 at EVERY t: '
      'all the t-dependence above is joint structure.')"""))

c1.append(md(r"""## 5. The posterior behind Tweedie

$p(a|x)=\mathcal N(m, J^{-1})$ with $J=(e^{-2t}/\Delta_t)I+Q_0$ (tridiagonal!)
and $m=J^{-1}h$, $h=(e^{-t}/\Delta_t)x$ (`main.pdf` Prop. 9).  We verify the
posterior mean is what Tweedie needs, and that the posterior precision is
exactly 'evidence + prior'."""))
c1.append(co("""mu, Delta = ou_params(t)
mu_post, Sig_post = gaussian_posterior(x, t, Sigma0, alpha)
J = (mu*mu/Delta)*np.eye(K) + Q0
h = (mu/Delta)*x
print('max |J^-1 h - posterior mean| =', np.max(np.abs(np.linalg.solve(J, h) - mu_post)))
print('max |inv(J) - posterior cov | =', np.max(np.abs(np.linalg.inv(J) - Sig_post)))
S_from_post = (mu*mu_post - x)/Delta
print('max |Tweedie(posterior mean) - S_direct| =', np.max(np.abs(S_from_post - S_direct)))"""))

c1.append(md(r"""## Summary

* $Q_0$ tridiagonal at machine precision — Markov structure made algebra.
* $\Sigma_t = e^{-2t}\Sigma_0+\Delta_t I$, unit diagonal for all $t$.
* Direct score $=$ Tweedie score to $\sim10^{-15}$.
* $K=2$ closed form: coupling $r=\alpha e^{-2t}$, invisible to the marginal score.
* Posterior: $J = \text{evidence}\cdot I + Q_0$, mean linear in $x$.

Continue with **`02_belief_propagation.ipynb`** for the factor graph and the
message-passing computation of the same score in $O(K)$."""))


# ===========================================================================
# 02 -- belief propagation
# ===========================================================================

c2 = []
c2.append(md(r"""# 02 — Belief propagation on the chain

Companion to **sections 7–10 of `main.pdf`**.

The posterior factorises as prior $\times$ transitions $\times$ local evidence;
its factor graph is a **caterpillar tree**, so sum–product BP is *exact* and
costs $O(K)$ instead of $O(K^3)$.

We use **Convention A** (`main.pdf` Prop. 11): the forward message into $a_k$
carries the evidence $x_0..x_{k-1}$ *only*; the backward message carries
$x_{k+1}..x_{K-1}$ *only*; the local evidence enters once, at combination time.
This is the bookkeeping under which
$$p(a_k|x)\propto \mu_{\to k}(a_k)\;\psi^{(k)}_{\rm ob}(a_k)\;\mu_{\leftarrow k}(a_k)$$
is correct *as written* — the audit caught an earlier version that absorbed the
evidence into the forward message and silently double-counted every $x_k$."""))
c2.append(co(PREAMBLE + """
from ar1_utils import (ar1_covariance, ou_params, gaussian_posterior,
                       joint_score_matrix)
from bp_score import (bp_posterior, bp_score, _forward_pass, _backward_pass,
                      _evidence_in_a, _gaussian_product)
print('BP code loaded')"""))

c2.append(md(r"""## 1. The $K=3$ worked example — every message, numerically

Same instance as `main.pdf` section 9.3: $\alpha=0.7$, $t=0.5$,
$x=(1.2,\,-0.4,\,0.8)$.  The observation factor, read as a function of $a_k$,
is the Gaussian *pseudo-observation*
$\mathcal N(a_k;\, e^{t}x_k,\, \Delta_t e^{2t})$ — precision $e^{-2t}/\Delta_t$
(the 'pinning strength' of the tilted-measure picture)."""))
c2.append(co("""K3, alpha3, t3 = 3, 0.7, 0.5
x3 = np.array([1.2, -0.4, 0.8])
sig_eta = np.sqrt(1 - alpha3**2)
mu3, Delta3 = ou_params(t3)
print(f'e^-t = {mu3:.6f}   Delta_t = {Delta3:.6f}   sigma_eta^2 = {1-alpha3**2:.2f}')
print(f'evidence precision e^-2t/Delta = {mu3*mu3/Delta3:.6f}')

m_to, v_to = _forward_pass(x3, t3, alpha3, sig_eta, 0.0, 1.0)
m_lr, v_lr = _backward_pass(x3, t3, alpha3, sig_eta)

print('\\nFORWARD  messages mu_->k  (carry x_0..x_{k-1}, Convention A):')
for k in range(K3):
    print(f'   k={k}:  mean={m_to[k]:+.6f}   var={v_to[k]:.6f}')
print('BACKWARD messages mu_<-k  (carry x_{k+1}..x_{K-1}):')
for k in range(K3):
    vv = 'flat (lambda=0)' if np.isinf(v_lr[k]) else f'var={v_lr[k]:.6f}'
    print(f'   k={k}:  mean={m_lr[k]:+.6f}   {vv}')"""))

c2.append(co("""print('COMBINATION  forward x evidence x backward at each node:')
for k in range(K3):
    m_ev, v_ev = _evidence_in_a(x3[k], t3)
    m_a, v_a = _gaussian_product(m_to[k], v_to[k], m_ev, v_ev)
    m_b, v_b = _gaussian_product(m_a, v_a, m_lr[k], v_lr[k])
    print(f'   posterior a_{k}:  mean={m_b:+.6f}   var={v_b:.6f}')

mu_post, Sig_post = gaussian_posterior(x3, t3, ar1_covariance(K3, alpha3), alpha3)
print('\\nmatrix posterior mean      =', mu_post)
print('matrix posterior variances =', np.diag(Sig_post))
print('\\n-> identical: BP recombination = exact matrix posterior.')
S3 = (mu3*mu_post - x3)/Delta3
print('score via Tweedie =', S3)
print('note S_1 > 0 although x_1 = -0.4 < 0: the neighbours pull the posterior')
print('mean of a_1 positive -- a purely JOINT effect (main.pdf, end of sec. 9).')"""))

c2.append(md(r"""## 2. Hand-check of one forward step (transparency)

Forward step $0\to1$, exactly as in the paper:
1. **update**: combine prior $(0,1)$ with pseudo-observation at node 0 —
   precisions add;
2. **predict**: push through the transition — mean $\times\alpha$, variance
   $\alpha^2 v + \sigma_\eta^2$.

This is one predict–update cycle of the **Kalman filter**; the backward sweep
plus combination is the **RTS smoother**."""))
c2.append(co("""lam_c = 1.0 + mu3*mu3/Delta3                      # combined precision at node 0
v_c   = 1.0/lam_c
m_c   = v_c*(0.0 + (mu3/Delta3)*x3[0])
print(f'update : combined (m, v) at node 0 = ({m_c:.6f}, {v_c:.6f})')
m_pred = alpha3*m_c
v_pred = alpha3**2*v_c + (1-alpha3**2)
print(f'predict: message into a_1        = ({m_pred:.6f}, {v_pred:.6f})')
print(f'code   : mu_->1                  = ({m_to[1]:.6f}, {v_to[1]:.6f})   match')"""))

c2.append(md(r"""## 3. BP $=$ exact score, across the parameter space

The same comparison the audit runs as its section 5: 80 configurations
($K\in\{2,3,5,8,16\}$, $\alpha\in\{0.2,0.5,0.9,-0.4\}$,
$t\in\{0.05,0.3,1,3\}$), random $x$.  Gaussian messages are **exact closure**
(product of Gaussians is Gaussian, Gaussian through a linear transition is
Gaussian) — no central limit theorem anywhere, so exactness holds for any node
degree, on this degree-2 chain included (`main.pdf` Th. 13)."""))
c2.append(co("""err = 0.0
for Kk in (2, 3, 5, 8, 16):
    for aa in (0.2, 0.5, 0.9, -0.4):
        for tt in (0.05, 0.3, 1.0, 3.0):
            xx = np.random.default_rng(Kk).standard_normal(Kk)
            err = max(err, np.max(np.abs(
                bp_score(xx, tt, aa)
                - joint_score_matrix(xx, tt, ar1_covariance(Kk, aa), aa))))
print('max |S_BP - S_exact| over 80 configurations =', err)
print('-> BP IS the score, computed through the tridiagonal posterior in O(K).')"""))

c2.append(md(r"""## Summary

* Convention A avoids the double-counting trap (the audit is what caught it).
* Forward sweep $=$ Kalman filter; combination $=$ RTS smoother.
* All messages exactly Gaussian by algebraic closure — *not* a CLT.
* BP score $=$ matrix score to $\sim 10^{-14}$ over the full parameter sweep.

Continue with **`03_amp_and_experiments.ipynb`** for the closed-form bulk
analysis, the AMP comparison and the two experiments."""))


# ===========================================================================
# 03 -- bulk analysis, AMP, experiments
# ===========================================================================

c3 = []
c3.append(md(r"""# 03 — Bulk closed forms, AMP, and the two experiments

Companion to **sections 11–14 of `main.pdf`**.

In the homogeneous interior (bulk) of a long chain, the posterior precision is
tridiagonal Toeplitz with
$$J_d=\frac{e^{-2t}}{\Delta_t}+\frac{1+\alpha^2}{1-\alpha^2},\qquad
\beta=-\frac{\alpha}{1-\alpha^2},$$
and *everything* has a closed form:

| quantity | formula | exists |
|---|---|---|
| exact/BP bulk variance | $1/\sqrt{J_d^2-4\beta^2}$ | always |
| BP cavity precision | $\lambda^*=(J_d+\sqrt{J_d^2-4\beta^2})/2$ | always ($J_d\ge2|\beta|$ by AM–GM) |
| posterior covariance | $(J^{-1})_{i,i+d}=q^d/\sqrt{J_d^2-4\beta^2}$, $q=\frac{J_d-\sqrt{J_d^2-4\beta^2}}{2|\beta|}$ | always |
| AMP/TAP bulk variance | $(J_d-\sqrt{J_d^2-8\beta^2})/4\beta^2$ | **iff** $J_d\ge2\sqrt2|\beta|$ |

The AMP existence line gives an exact breakdown time $t_c(\alpha)$ and the
critical coupling $\alpha_c=\sqrt2-1$."""))
c3.append(co(PREAMBLE + """
from IPython.display import Image, display
from amp import (posterior_precision_field, exact_marginals, mean_iteration,
                 amp_variance, mean_field_variance)
from chain_formulas import (bulk_params, bulk_variance_exact, bp_cavity_precision,
                            bulk_correlation_decay, bulk_covariance,
                            amp_bulk_variance, amp_critical_time,
                            amp_fixed_point_exists, amp_weak_coupling_error)
from local_bp import local_score, rms_truncation_error
from ar1_utils import ar1_covariance, joint_score_matrix
print('all modules loaded')"""))

c3.append(md(r"""## 1. The bulk formulas vs brute force

Centre of a $K=400$ chain (boundary effects decay geometrically, so the centre
is 'infinite-chain' to machine precision)."""))
c3.append(co("""K = 400
rng = np.random.default_rng(0)
alpha, t = 0.8, 0.5
J, _ = posterior_precision_field(rng.standard_normal(K), t, alpha)
Jinv = np.linalg.inv(J)
i = K//2

Jd, beta = bulk_params(alpha, t)
print(f'J_d = {Jd:.6f}   beta = {beta:.6f}   |beta|/J_d = {abs(beta)/Jd:.4f}')
print(f'\\nbulk variance: brute force {Jinv[i,i]:.12f}  closed form {bulk_variance_exact(alpha,t):.12f}')
q = bulk_correlation_decay(alpha, t)
print(f'\\nposterior covariance is geometric with q = {q:.6f}:')
for d in range(5):
    print(f'   d={d}:  (J^-1)[i,i+d] = {Jinv[i,i+d]:+.8f}   q^d V = {bulk_covariance(alpha,t,d):+.8f}')
lam = bp_cavity_precision(alpha, t)
print(f'\\nBP cavity fixed point lambda* = {lam:.6f};  '
      f'recombined precision J_d - 2 beta^2/lambda* = {Jd - 2*beta**2/lam:.6f}'
      f' = sqrt(J_d^2-4beta^2) = {np.sqrt(Jd*Jd-4*beta*beta):.6f}')"""))

c3.append(md(r"""## 2. BP vs AMP: same score, different variance

The posterior **mean** solves $Jm=h$ under mean field, BP *and* AMP — the
closures differ only in the **variance**.  So all three give the *same score*
(`main.pdf` Th. 17).  The AMP variance closure
$V_i=1/(J_{ii}-\sum_k J_{ik}^2V_k)$ keeps both neighbours where the BP cavity
excludes one: on the chain that single factor 2 moves the discriminant from
$J_d^2-4\beta^2$ (never negative) to $J_d^2-8\beta^2$ (negative at strong
coupling) — J. Garnier-Brun's "CLT on two points", made exact."""))
c3.append(co("""Kc, ac = 9, 0.8
xc = np.random.default_rng(21).standard_normal(Kc)
print(f'chain K={Kc}, alpha={ac}  (t_c({ac}) = {amp_critical_time(ac):.4f}):\\n')
for tt in (0.05, 0.1, 0.2, 0.5, 1.0):
    J, h = posterior_precision_field(xc, tt, ac)
    m_ex, v_ex = exact_marginals(J, h)
    m_amp = mean_iteration(J, h)[0]
    v_amp, _, ok = amp_variance(J)
    vtxt = (f'AMP var err = {np.max(np.abs(v_amp-v_ex)):.2e}' if ok
            else 'AMP variance: NO physical fixed point (breakdown)')
    print(f'  t={tt:<5} |m_amp - m_exact| = {np.max(np.abs(m_amp-m_ex)):.1e}   {vtxt}')
print('\\n-> the mean (hence the score) is exact at every t; the variance closure')
print('   degrades and then ceases to exist, exactly at t_c from the closed form.')"""))

c3.append(co("""# The exact breakdown boundary: t_c(alpha) and alpha_c = sqrt(2)-1
print('alpha_c = sqrt(2) - 1 =', np.sqrt(2)-1)
for a in (0.3, 0.41, 0.42, 0.5, 0.7, 0.8, 0.95):
    tc = amp_critical_time(a)
    print(f'  alpha={a:<5} t_c = {"inf (never breaks down)" if np.isinf(tc) else f"{tc:.5f}"}')
display(Image(filename='figures/fig_bulk_variance.png'))
display(Image(filename='figures/fig_bp_vs_amp.png'))"""))

c3.append(md(r"""## 3. Weak coupling: where AMP's variance error first appears

Expanding both closures in $\epsilon=\beta^2/J_d^2$:
they agree to $O(\epsilon)$, and
$$V_{\rm AMP}-V_{\rm exact}=\frac{2\beta^4}{J_d^5}+O(\beta^6)$$
— AMP always **over**estimates, starting at fourth order (`main.pdf` Th. 19)."""))
c3.append(co("""print('ratio (V_amp - V_exact) / (2 beta^4/J_d^5)  ->  1 at weak coupling:')
for a in (0.2, 0.3, 0.4):
    for tt in (0.01, 0.02):
        ratio = ((amp_bulk_variance(a, tt) - bulk_variance_exact(a, tt))
                 / amp_weak_coupling_error(a, tt))
        print(f'   alpha={a} t={tt}:  ratio = {ratio:.4f}')"""))

c3.append(md(r"""## 4. Experiment A — local messages vs full sweeps

The radius-$r$ local estimator uses only $x_{k-r}..x_{k+r}$ (computed
*exactly* on the window: a block of a stationary AR(1) is a stationary AR(1)).
Both estimators are linear in $x$, so the RMS truncation error over
$x\sim P_t$ is a **deterministic closed-form number** — and its decay rate in
$r$ is exactly the posterior correlation decay $q(\alpha,t)$
(`main.pdf` Th. 20)."""))
c3.append(co("""KA, aA = 121, 0.8
print('log-slope of the exact RMS truncation error vs the predicted log q:')
for tt in (0.1, 0.5, 2.0):
    rms = [rms_truncation_error(KA, aA, tt, KA//2, r) for r in range(0, 15)]
    rs = np.arange(2, 14)
    slope = np.polyfit(rs, np.log([rms[r] for r in rs]), 1)[0]
    print(f'   t={tt:<4} fitted slope = {slope:+.5f}   log q = {np.log(bulk_correlation_decay(aA, tt)):+.5f}')
print('\\nradius = K-1 recovers the exact score:')
xA = np.random.default_rng(3).standard_normal(21)
S_full = joint_score_matrix(xA, 0.5, ar1_covariance(21, aA), aA)
print('   max |S_local(r=K-1) - S_full| =',
      np.max(np.abs(local_score(xA, 0.5, aA, 20) - S_full)))
display(Image(filename='figures/fig_local_vs_full.png'))"""))

c3.append(md(r"""## 5. Experiment B — lifecycle of the precision matrix

$Q_t=\Sigma_t^{-1}$ is tridiagonal at $t=0$, the band fills for $t>0$ with the
**corrected** law
$(Q_t)_{i,i+d}=(-1)^{d-1}(2t)^{d-1}(Q_0^d)_{i,i+d}+O(t^d)$
(no $1/(d-1)!$ factor — `main.pdf` Remark 4 documents the correction), and
$Q_t\to I$ as $t\to\infty$ at rate $e^{-2t}$.  Eigenvectors are shared with
$\Sigma_0$ for every $t$ (near-Fourier modes); only eigenvalues flow."""))
c3.append(co("""display(Image(filename='figures/fig_precision_lifecycle.png'))
display(Image(filename='figures/fig_band_fill.png'))
display(Image(filename='figures/fig_spectral.png'))
display(Image(filename='figures/fig_tridiag_loss.png'))"""))

c3.append(md(r"""## 6. The full numerical audit — the gate

Every closed form quoted in `main.pdf` and in these notebooks, re-verified
end-to-end.  **A claim enters the text only if its check passes here.**"""))
c3.append(co("""import numerical_audit
rc = numerical_audit.run_all()
assert rc == 0, 'AUDIT FAILED - do not trust the documents until fixed'"""))

c3.append(md(r"""## Summary

* Bulk posterior: variance $1/\sqrt{J_d^2-4\beta^2}$, covariance $q^d\,V$ — verified to $10^{-11}$.
* BP fixed point exists for **every** $(\alpha,t)$; AMP's only for $J_d\ge2\sqrt2|\beta|$,
  i.e. below the exact breakdown time $t_c(\alpha)$; $\alpha_c=\sqrt2-1$.
* BP, AMP and mean field share the exact mean ⇒ the **same score**; the variance
  is where they differ ($+2\beta^4/J_d^5$ at weak coupling, no solution beyond $t_c$).
* Locality error decays exactly as $q^r$; the lifecycle of $Q_t$ is the same
  $q$-story seen in matrix form.
* Audit: **72/72**."""))


build('01_model_and_score.ipynb', c1)
build('02_belief_propagation.ipynb', c2)
build('03_amp_and_experiments.ipynb', c3)
print('all three companions built and executed.')
