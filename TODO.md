# BLayers roadmap / to-dos

Positioning: BLayers is a **structured Bayesian regression** toolkit (GLMs,
hierarchical models, factorization machines, splines, sparse priors) — layers are
*added* into a linear predictor, not stacked into a deep net. The list below is
ordered by value/effort for that niche.

## Done
- [x] Cut `AttentionLayer` from core (off-brand; worst-served by mean-field VI;
      interpretability oversold). Removed from `layers.py`, `__init__.py`, README,
      and tests.
- [x] README scope note clarifying GLM/GAM focus + pointer to
      `random_flax_module` for true Bayesian neural nets.

## Tier 1 — Bayesian workflow tooling (highest leverage)
Inference exists; evaluation barely does. Mostly plumbing over NumPyro/ArviZ.
- [x] `FittedModel.to_arviz()` — MCMC via `az.from_numpyro` (posterior +
      sample_stats + log_likelihood, coerced to NumPy for arviz-stats); VI via
      `az.from_dict` + `numpyro.infer.log_likelihood`. Unlocks `az.summary`
      (R-hat/ESS), `az.loo` (PSIS-LOO), and `az.compare`. SVGD unsupported.
      Optional `blayers[arviz]` extra (arviz >= 1.0, Python >= 3.12).
      NOTE: arviz 1.x dropped WAIC — LOO is the comparison metric.
- [ ] MCMC diagnostics surfaced in `summary()` too (R-hat, ESS, divergence
      count) for users who don't reach for ArviZ.
- [x] `sample_prior(model, num_samples=...)` prior-predictive helper. Returns
      the raw draws dict (latents + prior-predictive `obs`); rejects `y`.
      Exported from `blayers`.

## Tier 2 — Close the GLM likelihood gaps
- [x] `categorical_link` (multiclass softmax) — takes `(n, num_classes)` logits
      from a layer's `units=K`; reads K from the trailing dim. Exported.
- [ ] `gamma_link` / `exponential_link` (positive continuous, survival).
- [ ] Censored / Tobit likelihood.
- [ ] `zinb_link` (zero-inflated negative binomial; have ZIP, not ZINB).

## Tier 3 — Production readiness
- [ ] `FittedModel.save()` / `load()` (params are pytrees — pickle / orbax /
      safetensors).
- [ ] Guide shortcuts in `fit()`: `guide="mvn" | "lowrank" | "flow" | "laplace"`,
      plus `init_loc_fn` passthrough. (Diagonal-normal VI underestimates the
      posterior correlations that hierarchical models produce.)

## Tier 4 — New marquee layer
- [ ] Hilbert-Space approximate GP layer (HSGP, Riutort-Mayol et al.) — reduces to
      a basis-function layer, fast/batchable, sits naturally next to splines and
      `RandomWalkLayer`.

## Correctness / robustness fixes (small, do alongside)
- [ ] `_utils.yield_batches` never shuffles — same fixed batches, same order every
      epoch. Add per-epoch permutation (biases minibatch VI, esp. on sorted data).
- [ ] Document that `Batched_Trace_ELBO` assumes **all latents are global** (it
      rescales the whole observed log-lik by N/B and never subsamples local
      latents). State as a hard constraint, not just a plate warning.
- [ ] `EmbeddingLayer` / `RandomEffectsLayer` use `theta[x.squeeze()]` — `squeeze`
      collapses a size-1 batch to a scalar index and misbehaves on multi-column x.
      Prefer `x.reshape(-1).astype(int)`.
- [ ] Note that `predict`/`summary` default seeds are constant (1, 2) so identical
      reruns aren't mistaken for method determinism.

## Docs
- [ ] Short "how BLayers composes with `random_flax_module`" note for people who
      want to mix structured terms with a neural component.
