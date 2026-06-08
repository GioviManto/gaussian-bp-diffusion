# Design — Gaussian Belief Propagation for the Diffusion Score

**Goal.** A clean, self-contained note (LaTeX → PDF + companion Jupyter notebook) that
derives, for the **Gaussian AR(1) Markov chain corrupted by independent
Ornstein–Uhlenbeck (OU) noise**, the joint diffusion score, then applies
**belief propagation (BP)** and **approximate message passing (AMP)** to it, and
verifies numerically that BP reproduces the exact score through the precision
matrix. Audience: thesis meeting with J. Garnier-Brun / M. Mézard.

**Principles (from the user):** simple → correct → understandable → clear.
Nothing unnecessary. Joint score throughout (never per-frame marginal).
Notation follows Mézard / the handwritten notebook.

## Model and conventions

- Clean frames `a = (a_0,…,a_{K-1})`, AR(1): `a_{k+1} = α a_k + η_k`, `η_k ~ N(0,σ_η²)`, `|α|<1`.
- Stationary normalization: `σ_η² = 1−α²` ⇒ stationary variance `σ_∞² = 1`.
- Clean covariance `(Σ_0)_{ij} = α^{|i-j|}`; clean precision `Σ_0^{-1}` **tridiagonal** (Markov).
- Forward OU at diffusion time `t`: `x_k = e^{-t} a_k + ξ_k`, `ξ_k ~ N(0, Δ_t)`, `Δ_t = 1−e^{-2t}` (variance-preserving, `Var(x_k)=1`).
- Noisy joint `P_t(x) = N(0, Σ_t)`, `Σ_t = e^{-2t} Σ_0 + Δ_t I`.
- Joint score `S(x,t) = −Σ_t^{-1} x` = `(e^{-t} ⟨a_k⟩_{a|x} − x_k)/Δ_t` (Tweedie).

## Document structure (`main.tex` → `main.pdf`)

1. Model — AR(1), stationarity, autocovariance, tridiagonal `Σ_0^{-1}`.
2. Clean joint `P_0(a)` — chain rule + Markov.
3. Noisy joint — OU channel, `P_t(x)=N(0,Σ_t)`.
4. Score (direct) — `S=−Σ_t^{-1}x`.
5. Score, Bayes/Mézard way — Tweedie/denoiser; equivalence of the two forms.
6. Posterior `P(a|x)` — Gaussian, factor graph = caterpillar tree, Gaussian-MRF reading.
7. **BP, explicit K=3** — draw the factor graph; derive every message; visualize the
   forward and backward sweeps on the factor nodes (static TikZ).
8. **General K + Gaussian specialization** — full sum–product equations; product of
   Gaussians ⇒ Gaussian messages (info form), forward=Kalman filter, backward=RTS
   smoother, closure proof (answers Jérôme's "are messages/posterior Gaussian?").
9. **AMP** — general idea + general formulas (Onsager/cavity), then our degree-2 chain:
   dense-graph CLT does not hold ⇒ compute AMP and compare numerically to BP and exact.
10. Verification (machine precision) — `Σ_0^{-1}` tridiagonal; BP marginals reconstruct
    `Σ_t^{-1}` ⇒ `S=−Σ_t^{-1}x`; BP vs AMP.

## Experiments (companion notebook focus)

- **Exp A — local vs full BP:** strictly nearest-neighbour ("local") messages vs full
  forward+backward sweeps; error vs exact `−Σ_t^{-1}x`; how the local error tracks the
  off-diagonal coupling decay with `t`.
- **Exp B — precision lifecycle / spectral:** eigendecomposition of `Σ_t^{-1}` (near-Fourier
  modes, Toeplitz); lifecycle `Σ_0^{-1}` tridiagonal (t=0) → band fills (t>0) → → I (t→∞);
  quantify tridiagonal loss via `‖Σ_t − Σ_0‖` and the **band-fill scaling with NO 1/(d-1)!
  factor** (corrected formula).

## Reuse and correctness

- Reuse audited code from `unified_document/code`: `ar1_utils.py`, `bp_score.py`,
  `numerical_audit.py` (Convention A — avoids the BP double-counting trap).
- New code: `amp.py` (AMP/GAMP for the linear-Gaussian chain), `local_bp.py` (truncated
  local messages), `experiments.py` (figures + K=3 message tables).
- Every `=` / "exact" in the prose is backed by a passing check in `numerical_audit.py`.

## References

- Mézard & Montanari, *Information, Physics, and Computation* (Oxford, 2009) — ch. 14, BP.
- Zdeborová & Krzakala, *Statistical Physics for Optimization and Learning*, EPFL Doctoral
  Lectures 2021 — https://sphinxteam.github.io/EPFLDoctoralLecture2021/ — AMP/TAP/cavity.
- Genovese & Piana (2026), *Derivation of the AMP equations from belief propagation for the
  ℓ₂ minimisation problem*, arXiv:2602.15191 — BP→AMP (asymptotic).
- Mézard, *Generative diffusion* lecture notes (2025) — OU diffusion, score.
- A. Jones, *Belief propagation* (blog); J. Stringham, *Sum-product message passing* (blog);
  krashkov/Belief-Propagation (GitHub) — pedagogy.

## Deliverables

`main.tex`, `main.pdf`, `companion.ipynb` (executed), `figures/`, `references.bib`,
`README.md` → pushed to new repo `GioviManto/gaussian-bp-diffusion` (+ optional Pages).
