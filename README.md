# Gaussian Belief Propagation for the Diffusion Score

A fully solvable, closed-form study of the **joint diffusion score** of a
stationary Gaussian **AR(1)** Markov chain whose frames are independently
corrupted by an **Ornstein–Uhlenbeck** channel — derived from first
principles, attacked with **belief propagation (BP)** and **approximate
message passing (AMP)**, and solved in closed form in the bulk.

No neural networks. Everything is analytical, and every quantitative claim is
backed by an independent numerical audit (**72/72 checks pass**). The
documents are written to be read standalone: every derivation is in the PDF,
every number is reproduced by the notebooks.

- 📄 **[`main.pdf`](main.pdf)** — the note (31 pages: model, Tweedie, factor
  graph, BP by hand, closed-form bulk analysis, AMP breakdown, experiments)
- 📓 **Three executed companion notebooks**, mirroring the PDF:
  - [`01_model_and_score.ipynb`](01_model_and_score.ipynb) — model, clean/noisy
    joint, score two ways, the fully explicit `K=2` case (PDF §2–6)
  - [`02_belief_propagation.ipynb`](02_belief_propagation.ipynb) — Convention A,
    the `K=3` message tables by hand, BP = exact score (PDF §7–10)
  - [`03_amp_and_experiments.ipynb`](03_amp_and_experiments.ipynb) — bulk
    closed forms, BP vs AMP, breakdown boundary, both experiments, full audit
    (PDF §11–14)
- 🧮 **[`code/`](code/)** — the audited code

## Main results

1. **Joint score, two ways.** `S(x,t) = −Σ_t⁻¹x` directly, and
   `S_k = (e⁻ᵗ⟨a_k⟩_{a|x} − x_k)/Δ_t` via Tweedie — proved and verified
   identical. The `K=2` case is fully explicit (coupling `r = α e⁻²ᵗ`).
2. **BP is exact and is Kalman.** The posterior is a caterpillar tree;
   Convention-A sum–product (every message Gaussian by *algebraic closure*,
   not a CLT) reproduces the score at machine precision; the sweeps are the
   Kalman filter and RTS smoother.
3. **The chain in closed form (bulk).** With
   `J_d = e⁻²ᵗ/Δ_t + (1+α²)/(1−α²)`, `β = −α/(1−α²)`:
   - exact bulk variance `1/√(J_d²−4β²)`; BP cavity fixed point
     `λ* = (J_d+√(J_d²−4β²))/2` exists for **every** (α,t);
   - full posterior covariance is geometric: `(J⁻¹)_{i,i+d} = qᵈ·V` with
     `q = (J_d−√(J_d²−4β²))/(2|β|)` — `q→0` at `t→0`, `q→α` at `t→∞`.
4. **AMP: same score, explicit breakdown.** BP, AMP and mean field share the
   exact posterior mean ⇒ the **same score**. The AMP/TAP variance closure has
   the closed-form fixed point `(J_d−√(J_d²−8β²))/4β²`, existing **iff**
   `J_d ≥ 2√2·|β|` — an exact breakdown time `t_c(α)` and a critical coupling
   **`α_c = √2−1 ≈ 0.4142`** below which AMP never breaks down. At weak
   coupling `V_AMP − V_exact = 2β⁴/J_d⁵ + …` (AMP overestimates, at fourth
   order).
5. **Experiment A (locality).** The RMS truncation error of a radius-`r`
   local estimator decays **exactly** as `qʳ` — local messages vs full sweeps,
   quantified.
6. **Experiment B (lifecycle).** `Q_t = Σ_t⁻¹` is tridiagonal at `t=0`, fills
   as `(Q_t)[i,i+d] = (−1)^{d−1}(2t)^{d−1}(Q_0^d)[i,i+d] + O(t^d)`
   (**no `1/(d−1)!` factor** — a corrected formula, see PDF Remark 4), and
   returns to `I` at rate `e⁻²ᵗ`; eigenvectors are shared with `Σ_0` at all
   times.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib jupyter nbclient nbformat

# 1. verify every identity (the gate)
python code/numerical_audit.py           # -> PASSED 72 / 72

# 2. regenerate the figures
python code/experiments.py               # -> figures/*.png

# 3. rebuild the three executed notebooks
python build_companions.py               # -> 01_*.ipynb 02_*.ipynb 03_*.ipynb

# 4. compile the PDF (needs Tectonic: `brew install tectonic`)
tectonic main.tex                        # -> main.pdf (31 pages)
```

## Code map

| file | contents |
|------|----------|
| `code/ar1_utils.py` | model: `Σ_0`, `Q_0`, `Σ_t`, `Q_t` (direct & spectral), exact score (matrix & Tweedie), posterior |
| `code/bp_score.py`  | Convention-A BP: forward = Kalman filter, backward = RTS smoother |
| `code/chain_formulas.py` | every closed form of the bulk analysis: `J_d, β, λ*, V_exact, q`, bulk covariance, `V_AMP`, existence, `t_c`, `α_c`, weak-coupling error; `K=2` forms |
| `code/amp.py`       | the three marginal schemes (mean field, AMP/TAP, exact) on any `(J,h)`; honest breakdown reporting |
| `code/local_bp.py`  | radius-`r` local estimator + exact RMS truncation error |
| `code/experiments.py` | all figures |
| `code/numerical_audit.py` | **72 independent checks** — the gate: a claim enters the documents only if its check passes |

## Audit discipline

Every numbered claim in `main.pdf` maps to a named check in
`code/numerical_audit.py` (see the PDF's Appendix A for the claim ↔ check
table). Checks compare each closed form against an *independent* route:
brute-force inversion, spectral identities, bisection of iterations, fitted
slopes. Fewer results, certainly right, over many merely plausible.

## Key references

- M. Mézard & A. Montanari, *Information, Physics, and Computation* (Oxford, 2009), ch. 14.
- M. Mézard, *Generative diffusion models: lecture notes* (2025).
- L. Zdeborová & F. Krzakala, *Statistical Physics for Optimization and Learning*,
  [EPFL Doctoral Lectures 2021](https://sphinxteam.github.io/EPFLDoctoralLecture2021/).
- G. Genovese & A. Piana, *Derivation of the AMP equations from belief propagation
  for the ℓ₂ minimisation problem*, [arXiv:2602.15191](https://arxiv.org/abs/2602.15191) (2026).
- B. Efron, *Tweedie's formula and selection bias*, JASA (2011);
  P. Vincent, *A connection between score matching and denoising autoencoders*, Neural Comp. (2011).
- A. Jones, [*Belief propagation*](https://andrewcharlesjones.github.io/journal/belief-propagation.html);
  J. Stringham, [*Sum-product message passing*](https://jessicastringham.net/2019/01/09/sum-product-message-passing/);
  [krashkov/Belief-Propagation](https://github.com/krashkov/Belief-Propagation).
