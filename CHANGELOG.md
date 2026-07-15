# Changelog

All notable changes to BLayers are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow semantic versioning (with the usual 0.x caveat that minor releases
may carry breaking changes).

## [0.3.1]

### Added
- `FittedModel.to_arviz()` — convert a fit to an ArviZ `InferenceData` for
  diagnostics (R-hat, ESS, divergences) and model comparison (PSIS-LOO via
  `az.loo`, `az.compare`). MCMC uses `arviz.from_numpyro`; VI builds the
  `log_likelihood` group via `numpyro.infer.log_likelihood`. SVGD is
  unsupported. Requires the optional `blayers[arviz]` extra (arviz >= 1.0).
- `sample_prior(model, **inputs)` — draw from the prior / prior predictive
  before fitting, for prior checks.
- `categorical_link` — Categorical (softmax) likelihood for multiclass
  classification (`units = num_classes`).
- `__version__` on the top-level package.

### Changed
- **`Batched_Trace_ELBO` now raises `ValueError` on models that use
  `numpyro.plate`** (previously it emitted a `UserWarning` and continued).
  The `num_obs / batch_size` rescaling double-counts plate-subsampled sites,
  so the ELBO was silently wrong — better to fail closed. Use the standard
  `Trace_ELBO` with plates instead.
- **Requires Python >= 3.12** (was declared `>=3.9`, but the codebase's
  `X | None` annotations never actually supported 3.9; arviz 1.x also needs
  3.12). CI, docs, and publish workflows now run on 3.12.
- Documented BLayers' scope in the README: it is a **structured Bayesian
  regression** toolkit (GLMs, hierarchical models, factorization machines,
  splines, sparse priors) whose layers are *added* into a linear predictor —
  not stacked into a deep network. For true Bayesian neural nets, use
  NumPyro's `random_flax_module` / `random_haiku_module`.

### Removed
- **`AttentionLayer`** — removed. It was the one primitive at odds with the
  library's additive/interpretable focus, and mean-field VI serves its
  weight space poorly. If you need it, pin `blayers==0.3.0`, or use
  NumPyro's neural-network module integration.

### Fixed
- Minibatch VI now **shuffles** the data each epoch (`svi_run_batched` /
  `yield_batches`), instead of iterating the same fixed batches in the same
  order every pass. Removes a bias in the ELBO gradient estimate, especially
  on sorted data.
- `EmbeddingLayer` / `RandomEffectsLayer` / `RandomWalkLayer` index lookups no
  longer collapse a single-row batch to a scalar and now accept float-typed
  indices (`reshape(-1).astype(int)` instead of `squeeze()`).
- Fixed a duplicated (and mutually conflicting) `black` hook in the
  pre-commit config.

### Documentation
- `Batched_Trace_ELBO` documents that it assumes **all latents are global**
  (per-observation latents are unsupported in batched mode).
- `FittedModel.predict` notes that `.mean` / `.std` are not meaningful for
  classification / discrete links — work from `.samples` instead.
