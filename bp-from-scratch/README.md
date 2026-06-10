# BP from scratch — the equations, one by one

A self-contained, from-first-principles derivation of belief propagation for
the Gaussian AR(1) diffusion model, written as an independent reconstruction
based **only** on the call with J. Garnier-Brun of 5 June 2026 (the
"BP continuous case" transcript). Nothing is assumed known; nothing is left
implicit.

- 📄 **[`main.pdf`](main.pdf)** (16 pages) —
  1. the agenda fixed by the call (continuous messages are functions →
     parametrise as Gaussians → AMP/tapification; the checkpoint; the doubts;
     the experiments)
  2. the model with every symbol defined; the exact score derived from zero
     (direct and via Tweedie, both in full)
  3. the factor graph constructed step by step (why observations are factors,
     not nodes; why the graph is a tree)
  4. the sum–product equations (SP1)–(SP3) explained term by term, the two
     Gaussian lemmas proved, and the chain recursions (F0)–(F2), (B0)–(B2),
     (C) derived with every operation labelled — including the
     double-counting trap, stated loudly
  5. the checkpoint: a fully printed K=4 instance with one step verified by
     hand, then a 64-configuration sweep at 1e-14
  6. answers to the call's questions: messages are *exactly* Gaussian here
     (algebraic closure, no CLT); node-wise Gaussianity does **not** imply
     joint Gaussianity in general; the "CLT on two points" objection bites
     the AMP **variance**, never the score
  7. Experiment 1: truncated score matrix vs truncated message range —
     deterministic relative RMS errors, peak ~30% at intermediate t
  8. Experiment 2: AMP vs exact score — score exact at all t, variance loses
     its fixed point past a threshold
- 📓 **[`companion.ipynb`](companion.ipynb)** — executed interactive twin
- 🧮 **[`bp_gaussian.py`](bp_gaussian.py)** — the code, with built-in
  self-checks (`python bp_gaussian.py`)
- 📊 **[`experiments.py`](experiments.py)** — every figure and number quoted

## Reproduce

```bash
python bp_gaussian.py        # self-checks: ALL PASS
python experiments.py        # figures/ + printed numbers
python build_notebook.py     # re-execute the notebook
tectonic main.tex            # main.pdf
```
