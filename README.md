# Gaussian Belief Propagation for the Diffusion Score

A fully solvable, closed-form study of the **joint diffusion score** of a
stationary Gaussian **AR(1)** Markov chain whose frames are independently
corrupted by an **Ornstein–Uhlenbeck** channel — derived, then attacked with
**belief propagation (BP)** and **approximate message passing (AMP)**.

No neural networks. Everything is analytical, and every quantitative claim is
backed by an independent numerical audit (**59/59 checks pass**).

- 📄 **[`main.pdf`](main.pdf)** — the note (derivations, factor graph, sweeps, AMP, experiments)
- 📓 **[`companion.ipynb`](companion.ipynb)** — executed notebook reproducing the results & figures
- 🧮 **[`code/`](code/)** — the audited code

## What's inside

Following the order of the handwritten project notebook and the notation of
Mézard:

1. **Model** — Gaussian AR(1), stationarity, tridiagonal clean precision `Q₀ = Σ₀⁻¹`.
2. **Clean joint** `P₀(a)` (chain rule + Markov).
3. **Noisy joint** — OU channel, `Pₜ(x) = N(0, Σₜ)`, `Σₜ = e⁻²ᵗΣ₀ + Δₜ I`.
4. **Score** — directly `S = −Σₜ⁻¹x`, and the Bayesian/Tweedie denoiser
   `Sₖ = (e⁻ᵗ⟨aₖ⟩ − xₖ)/Δₜ`; the two agree.
5. **Posterior & factor graph** — a caterpillar tree ⇒ BP is exact.
6. **BP, K=3 explicit** — every message by hand; forward (Kalman filter) and
   backward (RTS smoother) sweeps drawn on the factor nodes.
7. **General K + Gaussian closure** — messages stay Gaussian *by closure, not by
   a CLT* (answers the "are messages/posterior Gaussian?" question).
8. **AMP** — the general idea, then our case: BP and AMP give the **same score**
   (the mean solves the same linear system); they differ only in the variance,
   where AMP's dense-graph closure is accurate at weak coupling but **breaks
   down** on the degree-2 chain.
9. **Experiment A** — local (truncated-range) messages vs full sweeps.
10. **Experiment B** — lifecycle of `Σₜ⁻¹`: tridiagonal at `t=0` → band fills →
    `→ I` as `t→∞`; spectral decomposition and the band-fill scaling
    `|Qₜ[i,i+d]| ~ (2t)^{d−1}` (**no `1/(d−1)!` factor**).

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib jupyter nbclient nbformat

# 1. verify every identity (machine precision)
python code/numerical_audit.py          # -> PASSED 59 / 59

# 2. regenerate the figures
python code/experiments.py               # -> figures/*.png

# 3. rebuild the executed notebook
python build_companion.py                # -> companion.ipynb

# 4. compile the PDF (needs Tectonic: `brew install tectonic`)
tectonic main.tex                        # -> main.pdf
```

## Code map

| file | contents |
|------|----------|
| `code/ar1_utils.py` | AR(1) covariance/precision, `Σₜ`, exact score (matrix & Tweedie), spectral form |
| `code/bp_score.py`  | BP on the chain (Convention A): forward = Kalman filter, backward = RTS smoother |
| `code/amp.py`       | Gaussian message passing: exact, mean iteration, AMP/TAP variance, mean field |
| `code/local_bp.py`  | local (radius-r) truncated estimator |
| `code/experiments.py` | all figures |
| `code/numerical_audit.py` | 59 independent checks (the gate) |

## Key references

- M. Mézard & A. Montanari, *Information, Physics, and Computation* (Oxford, 2009), ch. 14.
- L. Zdeborová & F. Krzakala, *Statistical Physics for Optimization and Learning*,
  [EPFL Doctoral Lectures 2021](https://sphinxteam.github.io/EPFLDoctoralLecture2021/).
- G. Genovese & A. Piana, *Derivation of the AMP equations from belief propagation
  for the ℓ₂ minimisation problem*, [arXiv:2602.15191](https://arxiv.org/abs/2602.15191) (2026).
- A. Jones, [*Belief propagation*](https://andrewcharlesjones.github.io/journal/belief-propagation.html);
  J. Stringham, [*Sum-product message passing*](https://jessicastringham.net/2019/01/09/sum-product-message-passing/);
  [krashkov/Belief-Propagation](https://github.com/krashkov/Belief-Propagation).
