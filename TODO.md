# BLayers roadmap / to-dos

Positioning: BLayers is a **structured Bayesian regression** toolkit (GLMs,
hierarchical models, factorization machines, splines, sparse priors) — layers are
*added* into a linear predictor, not stacked into a deep net. The list below is
ordered by value/effort for that niche.

## Done
- [x] Cut `AttentionLayer` from core (off-brand; worst-served by mean-field VI;
      interpretability oversold).
- [x] README scope note clarifying GLM/GAM focus + pointer to
      `random_flax_module` for true Bayesian neural nets.
- [x] `FittedModel.to_arviz()` — MCMC via `az.from_numpyro`, VI via `az.from_dict`
      + `numpyro.infer.log_likelihood`. Unlocks `az.summary` (R-hat/ESS),
      `az.loo` (PSIS-LOO), `az.compare`. SVGD unsupported. Optional
      `blayers[arviz]` extra (arviz >= 1.0, Python >= 3.12).
      NOTE: arviz 1.x dropped WAIC — LOO is the comparison metric.
- [x] `sample_prior(model, ...)` prior-predictive helper (rejects `y`).
- [x] GLM links: `categorical_link` (multiclass softmax), `gamma_link` /
      `exponential_link` (positive continuous / survival, log link),
      `zinb_link` (zero-inflated NegativeBinomial2).
- [x] `MixtureLayer` — finite mixture-of-priors (default Normal + Laplace) via
      `MixtureGeneral`; indicator marginalised so it works under VI *and* MCMC.
- [x] `HSGPLayer` (1-D squared-exponential) + `hsgp_L` helper — Hilbert-space
      approximate GP smoother that learns its own lengthscale.
- [x] Correctness: per-epoch batch shuffling; `Batched_Trace_ELBO` now
      **raises** on `numpyro.plate` (was a silent-wrong warning); embedding /
      random-effects / random-walk index safety (`reshape(-1).astype(int)`);
      default-seed docs on `predict`/`summary`.
- [x] Release/CI hygiene: `requires-python >= 3.12` (the `X | None` annotations
      never supported 3.9), `__version__`, CI on 3.12, deduped the `black`
      pre-commit hook, `pytest -n auto` (pytest-xdist) to parallelize CI.

## Proposed regression layers (banked for exploration)

Keep the existing layer APIs; no interaction-layer consolidation is planned.
Every new layer must support both VI and HMC/NUTS and work with the global-latent,
row-wise minibatching contract. These are design tasks, not settled APIs.

- [x] `RandomSlopesLayer`: independent, zero-centered group slope deviations
      with one learned scale per predictor/output. Population coefficients are
      separate. Uses the existing automatic non-centering support; unseen groups
      use reserved slots. Includes conditional Gaussian posterior validity tests.
- [ ] Correlated random slopes: extend `RandomSlopesLayer` with a
      scale/correlation decomposition and an LKJ prior; preserve the independent
      default and validate identifiability and VI/HMC behavior.
- [x] `PSplineLayer`: second-difference coefficient prior, separate proper
      coefficient-trend prior, fixed-reference anchoring, and constant boundary
      extrapolation. Prior and Gaussian posterior validity tests for VI/HMC.
- [x] `AR1Layer`: stationary initial state, learned persistence and innovation
      scale, continuous innovations on a fixed equally spaced grid, and reserved
      forecast slots. Covariance, forecast, and VI/HMC posterior validity tests.
- [ ] Tensor-product smooths: nonlinear interactions between covariates with
      separate marginal smoothing scales. Build on penalized splines and
      separate interaction terms from main-effect smooths for identifiability.
- [ ] Group-specific smooth deviations: partially pooled curves around a
      population smooth, with shared smoothness/shrinkage structure. Build on
      the varying-effects and penalized-spline designs.

## Tier 1 — remaining workflow tooling
- [ ] **Deferred / out of scope for now:** MCMC diagnostics in `summary()`
      (R-hat, ESS, divergences). NumPyro already provides these; revisit only
      if a BLayers convenience wrapper is needed.
- [ ] `posterior_predictive` group in `to_arviz()` so `az.plot_ppc_*` works
      natively (the one missing group in the diagnostic loop).

## Tier 2 — remaining GLM likelihoods
- [ ] Censored / Tobit likelihood. Deferred: needs a censoring-mask argument,
      so it breaks the clean `link(mu, y)` signature — design it deliberately
      (probably `censored_link(mu, y, censored_mask, ...)` + `numpyro.factor`).

## Tier 3 — production readiness
- [ ] `FittedModel.save()` / `load()` (params are pytrees — pickle / orbax /
      safetensors).
- [ ] Guide shortcuts in `fit()`: `guide="mvn" | "lowrank" | "flow" | "laplace"`,
      plus `init_loc_fn` passthrough. (Diagonal-normal VI underestimates the
      posterior correlations that hierarchical models produce.)

## GP follow-ups (extend HSGP; stay scalable/on-brand)
- [ ] Matérn kernels (3/2, 5/2) for `HSGPLayer` via a `kernel=` arg — just their
      spectral densities. Matérn is what most people want for rougher functions.
- [ ] Additive / multi-input HSGP — one HSGP term per covariate, summed
      (`f1(x1) + f2(x2) + ...`). This is a Bayesian GAM and fits the additive
      linear-predictor model exactly. Higher value than a joint multi-dim GP.
- [ ] (Maybe) exact GP layer, but ONLY as a clearly-fenced small-n reference:
      O(n^3), does not batch, and per-obs latent trips the `Batched_Trace_ELBO`
      hard-fail — full-batch `Trace_ELBO` / MCMC only. Not a default. HSGP is
      the scalable substitute for the common case.

## Docs
- [ ] Short "how BLayers composes with `random_flax_module`" note for people who
      want to mix structured terms with a neural component.
