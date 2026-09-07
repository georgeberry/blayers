# Changelog

All notable changes to BLayers are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims
to follow semantic versioning (with the usual 0.x caveat that minor releases
may carry breaking changes).

## [Unreleased]

### Added

Three new additive regression layers, exported from `blayers`, with support for
full-batch VI, row-wise minibatched VI, and ordinary HMC/NUTS:

- **`RandomSlopesLayer`** adds partially pooled group-specific slope deviations:
  `u[g, j] ~ Normal(0, tau[j])`. Each predictor/output learns a pooling scale
  shared across groups. Population slopes belong in a separate layer;
  correlations between slope deviations are not estimated. Keep group IDs and
  the full group count fixed; unseen groups require slots reserved before fit.
- **`PSplineLayer`** learns a nonlinear curve using B-splines with Normal priors
  on second differences of neighboring coefficients. Each output learns its
  own smoothing scale. A separate proper prior handles the unpenalized
  coefficient trend, and anchoring at a fixed reference removes the constant
  component. With clamped/uneven knots, the coefficient trend is not necessarily
  exactly linear in the input. Reuse knots and reference across batches and
  prediction; outside the knot domain the curve holds its boundary value.
- **`AR1Layer`** adds stationary temporal effects:
  `theta[t] = rho * theta[t-1] + epsilon[t]`. Both persistence `rho` and innovation
  scale are learned per output and shared across periods. The default prior is
  `rho = 2*p - 1`, with `p ~ Beta(2, 2)`, allowing positive and negative
  persistence. The initial state has the stationary prior. Equally spaced
  integer time slots preserve missing periods and support forecasting through
  future slots reserved before fitting.
- README examples for composing all three layers with intercepts, population
  effects, and observation links, including prediction and prior configuration.

### Inference notes

- `AR1Layer` samples states directly by default. `noncentered=True` samples
  standardized innovations instead; this choice is independent of
  `fit(autoreparam_model=...)`. Both forms define the same prior. Direct-state
  diagonal VI recovered persistence in the informative-data test where
  innovation-based diagonal VI substantially underestimated it.
- Random slopes and penalized splines use the existing automatic non-centering
  support. Parameterization, optimization budget, and guide choice still matter.
  Correlated uncertainty, especially across missing/future AR periods, benefits
  from a full-covariance VI guide or NUTS. Compatibility tests establish support,
  not convergence guarantees for arbitrary datasets.

### Changed
- Removed `SpikeAndSlabLayer` and its Gibbs-specific fitting path: every
  built-in layer must support both VI and HMC/NUTS. Use `HorseshoeLayer` for
  continuous sparse shrinkage. Neither the earlier relaxed gate nor the exact
  discrete spike-and-slab implementation remains in the public API.
- VI fits now honor `autoreparam_model=True` by default, constructing the guide
  on the non-centered model and retaining it for prediction and posterior export.
  Set `autoreparam_model=False` to retain the supplied parameterization. Prebuilt
  guides and custom guide callables require this opt-out and must match the model.

### Fixed
- VI summaries recover original coefficient sites from non-centered guide
  draws, including nested LogNormal transformations, instead of only exposing
  the transformed coordinates.
- Likelihood helpers align `(n,)` and `(n, 1)` targets, preventing silent
  `(n, n)` broadcasting with layers and `@autoreshape`. Incompatible row/output
  shapes raise an error. Location-scale links align per-row scales as well;
  scalar-response links preserve the row axis for a single observation.
- Batched VI uses the actual input row count for likelihood scaling, including
  short remainder batches and batch sizes larger than the dataset.
- Batched ELBO site densities now use NumPyro's density calculation, preserving
  model/guide scales, masks, and distribution intermediates. Model parameters
  are substituted alongside guide samples. Row-wise factors receive likelihood
  scaling; global factors remain unsupported for minibatching.

### Tests

- Random slopes: coefficient lookup and multi-output shapes, prior scaling and
  independence, eager/JIT group-ID safety, recovery of distinct pooling scales,
  and analytical Gaussian posterior means/standard deviations under full and
  minibatched VI and NUTS, including a reserved unobserved group.
- Penalized splines: exact second-difference construction, anchoring and batch
  invariance, constant extrapolation, prior covariance and roughness, input
  validation, and learning different smoothing strengths for simple and wavy
  functions.
- AR(1): stationary covariance with positive, zero, and negative persistence in
  both parameterizations; exact direct-state prior density; recurrence, repeated
  indices, forecast innovation variance, index safety, and recovery of
  persistence and innovation scales under diagonal VI and NUTS.
- Splines and both AR parameterizations: posterior predictive means and
  covariances checked against analytical conditional Gaussian solutions under
  full/minibatched VI and NUTS, including unobserved positions. These VI
  uncertainty checks use `AutoMultivariateNormal` to represent correlations.
- Added an inference-compatibility test covering every built-in layer under
  VI and HMC/NUTS, with full and minibatched VI.
- Added likelihood density checks for all 13 links with vector/column targets,
  `@autoreshape`, and single-row inputs; regression and heteroscedastic gradient
  checks; exact expected minibatch objective/gradient checks; end-to-end uneven
  batch fits; and comparisons with NumPyro for scaled/masked sites and factors.

## [0.3.5]

### Fixed
- **MCMC `predict` / `to_arviz` gave wrong results after an autoreparameterized
  fit.** `fit(method="mcmc")` applies `autoreparam` (non-centering) to improve NUTS
  mixing, but `get_samples()` returns the *original* parameterization. The
  `FittedModel` stored the reparam'd model and ran `Predictive` (and
  `log_likelihood`) against it with those original-space samples — a mismatch that
  silently produced garbage posterior-predictive means (in an unpredictable
  direction: below the noise floor on well-identified models, well above it on
  funnels), despite a perfectly good posterior. The fit now keeps the reparam'd
  model only for the NUTS kernel and stores the original model for prediction, so
  `predict` matches the centered fit and the true noise floor. Regression tests
  assert autoreparam on/off predict-invariance.

## [0.3.4]

### Added
- **`HorseshoeLayer(tau0=...)` / `HorseshoeInteractionLayer(tau0=...)`** — the scale
  of the `HalfCauchy` prior on the global shrinkage `tau`, i.e. how aggressively the
  layer shrinks (default `1.0`, unchanged from earlier releases; smaller pushes
  harder toward sparsity, per Piironen & Vehtari). The coefficient prior stays
  **centered** (`beta ~ Normal(0, tau·λ̃)`): NUTS handles the funnel, so MCMC
  identifies `tau` fine, while mean-field VI fits it poorly — prefer MCMC for
  horseshoe selection. (A layer-level non-centering was trialled and reverted: it
  destabilised mean-field VI on high-dimensional interaction bases.)

## [0.3.3]

### Changed
- **`MixtureLayer` learned weights are now logistic-normal, not `Dirichlet`.**
  When `weights=None` the mixing weights are `softmax` of `Normal(0, weight_scale)`
  logits (new `weight_scale` arg, default `1.0`, replacing `dirichlet_concentration`).
  This keeps every latent in unconstrained space, so `MixtureLayer` now fits under
  **SVGD** as well as VI and MCMC — a raw `Dirichlet` simplex site broke SVGD's
  particle flattener (its constrained dimension `k` differs from its unconstrained
  `k-1`). The sampled site is renamed `MixtureLayer_<name>_weights` →
  `MixtureLayer_<name>_logits`; `model_to_latex` renders the softmax accordingly.
- **`InteractionLayer` now takes an optional `z`.** Called with one feature set
  (`("beta", x)`) it builds the unique within-`x` pairs `x_i x_j`, `i < j` (no squares,
  no duplicates); called with two (`("beta", x, z)`) it builds the full `d1*d2` cross
  set as before. Existing two-argument calls are unchanged.

### Added
- **`HorseshoeInteractionLayer`** — explicit pairwise interactions with a per-pair
  **horseshoe** prior instead of a single global scale, so most interaction
  coefficients shrink to zero and the few real ones stand out. The layer to reach for
  to *identify* sparse interactions. Omit `z` (`("int", x)`) for the unique within-`x`
  pairs `x_i x_j`, `i < j` — the sparse-interaction case; pass `z` (`("int", x, z)`)
  for the full `d1*d2` cross-set grid, like `InteractionLayer`. Use
  `LowRankInteractionLayer` / `FMLayer` when you only need prediction at large `d`.
- **`shuffle` option for batched VI** (`fit(..., shuffle=...)` /
  `svi_run_batched`). Default `True` keeps the per-epoch reshuffle (unbiased
  ELBO gradient). `shuffle=False` iterates fixed contiguous slices — skipping
  the per-epoch permutation *and* the per-row gather (the no-shuffle path now
  slices arrays directly instead of index-gathering) — which is noticeably
  faster; use it when rows are already in random order.
- **`model_to_latex(model, **inputs)`** — render a model's generative form (all
  priors + the likelihood) as LaTeX by tracing it once. Hierarchy is recovered
  automatically (a coefficient whose scale is a sampled site prints
  `Normal(0, λ)`, not a number); the linear predictor is denoted `η_i` since its
  deterministic assembly isn't recoverable from a trace. Returns a `str` that
  also renders inline in Jupyter. Special-layer overrides for Horseshoe,
  SpikeAndSlab, Mixture, and HSGP; degrades gracefully on arbitrary NumPyro
  models.

## [0.3.2]

### Added
- **GLM likelihoods**: `gamma_link` and `exponential_link` (positive
  continuous / survival, log link) and `zinb_link` (zero-inflated
  NegativeBinomial2 for overdispersed, zero-heavy counts).
- **`MixtureLayer`** — coefficients from a finite mixture-of-priors (default
  Normal + Laplace) with a `Dirichlet` or fixed mixing weight. The component
  indicator is marginalised via `MixtureGeneral`, so the log-density is smooth
  and works under both VI and MCMC. For robustness / elastic-net-style priors.
- **`HSGPLayer`** — Hilbert-space approximate Gaussian process (1-D,
  squared-exponential; Riutort-Mayol et al. 2021), a GP smoother that learns
  its own lengthscale. Helper `hsgp_L(x)` picks the domain boundary; reuse the
  same `L` at fit and predict time.
- **`FixedEffectsLayer`** — per-category coefficients with a fixed,
  user-specified prior; the no-pooling counterpart of `RandomEffectsLayer`
  (whose learned variance component is what drives partial pooling).

### Fixed
- **Packaging**: wheels no longer ship the `tests/` package and `docs/conf.py`
  (they leaked into site-packages via an unfiltered `packages.find`); license
  metadata now correctly reports `License: MIT` (previously pointed at a
  nonexistent file).

### Development
- CI test suite parallelized with `pytest-xdist` (`pytest -n auto`).

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
