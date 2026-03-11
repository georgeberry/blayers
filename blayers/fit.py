"""
High-level fitting API for blayers models.

Reduces boilerplate when fitting Bayesian models by providing sensible defaults
for guides, optimizers, learning rate schedules, and prediction.

Example
-------

.. code-block:: python

    from blayers.layers import AdaptiveLayer
    from blayers.links import gaussian_link_exp
    from blayers.fit import fit

    def model(x, y=None):
        mu = AdaptiveLayer()('mu', x)
        return gaussian_link_exp(mu, y)

    # Fit with batched VI
    result = fit(model, y=y_train, batch_size=1024, num_epochs=100, x=x_train)

    # Predict on new data
    preds = result.predict(x=x_test)
    print(preds.mean, preds.std)

    # Fit with MCMC
    result = fit(model, y=y_train, method="mcmc", x=x_train)
    preds = result.predict(x=x_test)

    # Fit with Stein Variational Gradient Descent
    result = fit(model, y=y_train, method="svgd", num_steps=500, x=x_train)
    preds = result.predict(x=x_test)


Constants (non-array kwargs) are automatically bound via ``functools.partial``,
so you never need to wrap your model manually:

.. code-block:: python

    def model(x, n_conditions, y=None):
        ...

    # n_conditions is an int → auto-bound; x is an array → batched
    result = fit(model, y=y_train, batch_size=4096, num_epochs=250,
                 x=x_train, n_conditions=10)
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import optax
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoGuide

from blayers.vi_infer import Batched_Trace_ELBO, svi_run_batched
from blayers.decorators import autoreparam


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _is_array(v: Any) -> bool:
    """True if *v* is an array that should be batched over rows.

    Arrays with at least one dimension (shape ``(n,)`` or ``(n, d)``, etc.)
    return True.  Scalars, Python ints/floats/strings, and 0-d arrays return
    False.
    """
    return hasattr(v, "shape") and hasattr(v, "dtype") and len(v.shape) >= 1


def _split_data_and_constants(
    kwargs: dict[str, Any],
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Separate array-valued kwargs (data) from everything else (constants)."""
    data: dict[str, jax.Array] = {}
    constants: dict[str, Any] = {}
    for k, v in kwargs.items():
        if _is_array(v):
            data[k] = v
        else:
            constants[k] = v
    return data, constants


def _make_schedule(
    schedule: str,
    lr: float,
    total_steps: int,
) -> Any:
    """Build an optax learning rate schedule from a short name."""
    if schedule == "cosine":
        return optax.cosine_decay_schedule(lr, total_steps)
    elif schedule == "warmup_cosine":
        return optax.cosine_onecycle_schedule(
            transition_steps=total_steps,
            peak_value=lr,
            pct_start=0.1,
            div_factor=25.0,
        )
    elif schedule == "constant":
        return lr
    else:
        raise ValueError(
            f"Unknown schedule {schedule!r}. "
            "Choose from 'cosine', 'warmup_cosine', or 'constant'."
        )


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class Predictions:
    """Posterior predictive output from :meth:`FittedModel.predict`.

    Attributes
    ----------
    mean : jax.Array
        Point predictions averaged over posterior samples.  Shape ``(n,)``.
    std : jax.Array
        Predictive standard deviation over posterior samples.  Shape ``(n,)``.
    samples : jax.Array
        Raw posterior predictive draws.  Shape ``(num_samples, n, ...)``.
    """

    mean: jax.Array
    std: jax.Array
    samples: jax.Array


@dataclass
class FittedModel:
    """A fitted blayers model.

    Created by :func:`fit`.  Provides :meth:`predict` for posterior predictive
    inference and :meth:`summary` for inspecting latent variable posteriors.

    Attributes
    ----------
    model_fn : Callable
        The model function with any constants already bound.
    method : str
        ``"vi"`` or ``"mcmc"``.
    params : dict or None
        SVI parameters (VI only).
    guide : AutoGuide or None
        Fitted variational guide (VI only).
    losses : jax.Array or None
        Per-step ELBO loss curve (VI only).
    posterior_samples : dict or None
        MCMC posterior samples (MCMC only).
    """

    model_fn: Callable
    method: str
    # VI / SVGD
    params: dict | None = None
    guide: Any | None = None
    losses: jax.Array | None = field(default=None, repr=False)
    # MCMC
    posterior_samples: dict | None = field(default=None, repr=False)
    # SVGD
    num_particles: int | None = None

    def predict(
        self,
        *,
        num_samples: int = 100,
        seed: int = 1,
        **data: Any,
    ) -> Predictions:
        """Generate posterior predictive predictions on new data.

        Parameters
        ----------
        num_samples : int
            Number of posterior samples to draw.  For VI this controls the
            guide; for MCMC all posterior samples are used regardless.
        seed : int
            Random seed for the predictive distribution.
        **data
            Model inputs **excluding** ``y``.  Constants that were auto-bound
            during :func:`fit` should *not* be passed again.

        Returns
        -------
        Predictions
        """
        rng_key = jax.random.PRNGKey(seed)

        if self.method == "vi":
            predictive = Predictive(
                self.model_fn,
                guide=self.guide,
                params=self.params,
                num_samples=num_samples,
            )
        elif self.method == "svgd":
            predictive = Predictive(
                self.model_fn,
                guide=self.guide,
                params=self.params,
                num_samples=num_samples,
                batch_ndims=1,
            )
        elif self.method == "mcmc":
            predictive = Predictive(
                self.model_fn,
                posterior_samples=self.posterior_samples,
            )
        else:
            raise ValueError(f"Unknown method {self.method!r}")

        ppc = predictive(rng_key, **data)
        obs = ppc["obs"]

        if self.method == "svgd":
            # obs shape is (num_samples, num_particles, n, ...) — flatten the
            # sample and particle dimensions so mean/std marginalise over both.
            obs = obs.reshape(-1, *obs.shape[2:])

        return Predictions(
            mean=obs.mean(axis=0).squeeze(),
            std=obs.std(axis=0).squeeze(),
            samples=obs,
        )

    def summary(
        self,
        *,
        num_samples: int = 1000,
        seed: int = 2,
        **data: Any,
    ) -> dict[str, dict[str, Any]]:
        """Summarize the posterior of each latent variable.

        Parameters
        ----------
        num_samples : int
            Samples to draw from the guide (VI only; ignored for MCMC).
        seed : int
            Random seed.
        **data
            Model inputs (excluding ``y``) needed so the guide can determine
            parameter shapes.  Required for VI; ignored for MCMC.

        Returns
        -------
        dict
            ``{site_name: {"mean": ..., "std": ..., "q025": ..., "q975": ...,
            "shape": ...}}``
        """
        rng_key = jax.random.PRNGKey(seed)

        if self.method == "vi":
            if self.guide is None or self.params is None:
                raise RuntimeError("VI results missing guide or params")
            predictive = Predictive(
                self.guide,
                params=self.params,
                num_samples=num_samples,
            )
            samples = predictive(rng_key, **data)
        elif self.method == "svgd":
            # For SVGD the params dict already contains per-particle values
            # with shape (num_particles, ...).  Treat particles as samples.
            if self.params is None:
                raise RuntimeError("SVGD results missing params")
            samples = self.params
        elif self.method == "mcmc":
            if self.posterior_samples is None:
                raise RuntimeError("MCMC results missing posterior_samples")
            samples = self.posterior_samples
        else:
            raise ValueError(f"Unknown method {self.method!r}")

        result: dict[str, dict[str, Any]] = {}
        for name, vals in samples.items():
            if name == "obs":
                continue
            result[name] = {
                "mean": vals.mean(axis=0),
                "std": vals.std(axis=0),
                "q025": jnp.percentile(vals, 2.5, axis=0),
                "q975": jnp.percentile(vals, 97.5, axis=0),
                "shape": vals.shape[1:],
            }
        return result


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def fit(
    model_fn: Callable,
    *,
    y: jax.Array,
    method: Literal["vi", "mcmc", "svgd"] = "vi",
    # VI parameters
    batch_size: int | None = None,
    num_epochs: int | None = None,
    num_steps: int | None = None,
    lr: float = 0.01,
    schedule: str = "cosine",
    guide: type | AutoGuide | None = None,
    optimizer: optax.GradientTransformation | None = None,
    # MCMC parameters
    num_warmup: int = 500,
    num_mcmc_samples: int = 1000,
    num_chains: int = 1,
    autoreparam_model: bool = True,
    # SVGD parameters
    num_particles: int = 10,
    kernel_fn: Any = None,
    # Common
    seed: int = 0,
    **kwargs: Any,
) -> FittedModel:
    """Fit a blayers model via variational inference, MCMC, or SVGD.

    Keyword arguments that are JAX/numpy arrays are treated as **data** and
    batched during training.  Non-array keyword arguments (ints, floats,
    strings, etc.) are treated as **constants** and bound to the model via
    ``functools.partial`` so they don't need to be passed again at predict
    time.

    Parameters
    ----------
    model_fn : Callable
        A NumPyro model function that accepts ``y`` as a keyword argument.
    y : jax.Array
        Target / observed values.
    method : ``"vi"``, ``"mcmc"``, or ``"svgd"``
        Inference method.  Default ``"vi"``.

    batch_size : int, optional
        Mini-batch size for VI.  If *None* the full dataset is used each step
        (appropriate for small datasets).
    num_epochs : int, optional
        Number of full passes through the data.  Exactly one of *num_epochs*
        or *num_steps* is required for VI and SVGD.
    num_steps : int, optional
        Total number of gradient updates.  Exactly one of *num_epochs* or
        *num_steps* is required for VI and SVGD.
    lr : float
        Peak learning rate (default 0.01).  For SVGD this is the Adagrad
        step size.  Ignored when *optimizer* is given.
    schedule : str
        LR schedule name: ``"cosine"`` (default), ``"warmup_cosine"``, or
        ``"constant"``.  Only used for VI.
    guide : type or AutoGuide instance, optional
        Variational family.  Pass a **class** (instantiated on *model_fn*) or
        a ready-to-use **instance**.  Default: ``AutoDiagonalNormal``.
        Not used for SVGD (which auto-generates an ``AutoDelta`` guide).
    optimizer : optax.GradientTransformation, optional
        A fully-constructed optax optimizer.  When provided, *lr* and
        *schedule* are ignored.  Not used for SVGD.

    num_warmup : int
        MCMC warmup iterations (default 500).
    num_mcmc_samples : int
        MCMC posterior samples to draw (default 1000).
    num_chains : int
        Number of MCMC chains (default 1).
    autoreparam_model : bool
        Automatically reparameterize LocScale distributions for MCMC
        (default True).

    num_particles : int
        Number of Stein particles (default 10).  Only used for SVGD.
    kernel_fn : SteinKernel, optional
        Kernel for SVGD.  Default: ``RBFKernel()``.

    seed : int
        Random seed (default 0).
    **kwargs
        Model inputs.  Arrays → batched data.  Non-arrays → constants bound
        via ``partial``.

    Returns
    -------
    FittedModel
        Object with ``.predict(**data)`` and ``.summary(**data)`` methods.

    Examples
    --------
    Batched VI (the common case for large datasets):

    >>> result = fit(model, y=y_train, batch_size=4096, num_epochs=250,
    ...              x=x_train, n_conditions=10)
    >>> preds = result.predict(x=x_test)

    Full-dataset VI (small datasets):

    >>> result = fit(model, y=y_train, num_steps=20000, x=x_train)

    MCMC:

    >>> result = fit(model, y=y_train, method="mcmc", x=x_train)

    SVGD:

    >>> result = fit(model, y=y_train, method="svgd", num_steps=500,
    ...              num_particles=20, x=x_train)
    """
    # ------------------------------------------------------------------ #
    # Separate arrays (→ batched data) from scalars (→ partial-bound)
    # ------------------------------------------------------------------ #
    data, constants = _split_data_and_constants(kwargs)

    bound_model = partial(model_fn, **constants) if constants else model_fn

    data["y"] = y
    n_obs = y.shape[0]

    rng_key = jax.random.PRNGKey(seed)

    if method == "vi":
        return _fit_vi(
            bound_model,
            data=data,
            n_obs=n_obs,
            batch_size=batch_size,
            num_epochs=num_epochs,
            num_steps=num_steps,
            lr=lr,
            schedule=schedule,
            guide=guide,
            optimizer=optimizer,
            rng_key=rng_key,
        )
    elif method == "mcmc":
        return _fit_mcmc(
            bound_model,
            data=data,
            num_warmup=num_warmup,
            num_mcmc_samples=num_mcmc_samples,
            num_chains=num_chains,
            autoreparam_model=autoreparam_model,
            rng_key=rng_key,
        )
    elif method == "svgd":
        # Compute total steps (same logic as unbatched VI)
        if (num_epochs is None) == (num_steps is None):
            raise ValueError(
                "Provide exactly one of num_epochs or num_steps, not both (or neither)."
            )
        total_steps = num_epochs if num_epochs is not None else num_steps

        return _fit_svgd(
            bound_model,
            data=data,
            num_steps=total_steps,
            num_particles=num_particles,
            kernel_fn=kernel_fn,
            lr=lr,
            rng_key=rng_key,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'vi', 'mcmc', or 'svgd'."
        )


# --------------------------------------------------------------------------- #
# VI
# --------------------------------------------------------------------------- #


def _fit_vi(
    model_fn: Callable,
    *,
    data: dict[str, jax.Array],
    n_obs: int,
    batch_size: int | None,
    num_epochs: int | None,
    num_steps: int | None,
    lr: float,
    schedule: str,
    guide: type | AutoGuide | None,
    optimizer: optax.GradientTransformation | None,
    rng_key: jax.Array,
) -> FittedModel:
    """Fit a model with variational inference."""
    # ---- Validate epoch / step args ----
    if (num_epochs is None) == (num_steps is None):
        raise ValueError(
            "Provide exactly one of num_epochs or num_steps, not both (or neither)."
        )

    # ---- Compute total gradient steps (needed for LR schedule) ----
    batched = batch_size is not None

    if batched:
        steps_per_epoch = (n_obs + batch_size - 1) // batch_size
        total_steps = (
            steps_per_epoch * num_epochs if num_epochs is not None else num_steps
        )
    else:
        # Without batching each gradient step sees the full dataset,
        # so one step ≡ one epoch.
        total_steps = num_epochs if num_epochs is not None else num_steps

    # ---- Guide ----
    if guide is None:
        guide_instance = AutoDiagonalNormal(model_fn)
    elif isinstance(guide, type):
        guide_instance = guide(model_fn)
    else:
        # Already an instance — use as-is.
        guide_instance = guide

    # ---- Optimizer ----
    if optimizer is None:
        lr_or_schedule = _make_schedule(schedule, lr, total_steps)
        opt = optax.adam(lr_or_schedule)
    else:
        opt = optimizer

    # ---- Loss ----
    if batched:
        loss = Batched_Trace_ELBO(num_obs=n_obs, batch_size=batch_size)
    else:
        loss = Trace_ELBO()

    # ---- Run SVI ----
    svi = SVI(model_fn, guide_instance, opt, loss=loss)

    if batched:
        result = svi_run_batched(
            svi,
            rng_key,
            batch_size=batch_size,
            num_steps=num_steps,
            num_epochs=num_epochs,
            **data,
        )
    else:
        result = svi.run(rng_key, total_steps, **data)

    return FittedModel(
        model_fn=model_fn,
        method="vi",
        params=result.params,
        guide=guide_instance,
        losses=result.losses,
    )


# --------------------------------------------------------------------------- #
# MCMC
# --------------------------------------------------------------------------- #


def _fit_mcmc(
    model_fn: Callable,
    *,
    data: dict[str, jax.Array],
    num_warmup: int,
    num_mcmc_samples: int,
    num_chains: int,
    autoreparam_model: bool,
    rng_key: jax.Array,
) -> FittedModel:
    """Fit a model with NUTS MCMC."""
    if autoreparam_model:
        model_fn = autoreparam()(model_fn)

    kernel = NUTS(model_fn)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_mcmc_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(rng_key, **data)

    return FittedModel(
        model_fn=model_fn,
        method="mcmc",
        posterior_samples=mcmc.get_samples(),
    )


# --------------------------------------------------------------------------- #
# SVGD
# --------------------------------------------------------------------------- #


def _fit_svgd(
    model_fn: Callable,
    *,
    data: dict[str, jax.Array],
    num_steps: int,
    num_particles: int,
    kernel_fn: Any,
    lr: float,
    rng_key: jax.Array,
) -> FittedModel:
    """Fit a model with Stein Variational Gradient Descent.

    Uses ``numpyro.contrib.einstein.SVGD`` which auto-generates an
    ``AutoDelta`` guide and maintains a set of particles that are
    iteratively pushed toward the posterior via a kernelised gradient.
    """
    from numpyro.contrib.einstein import SVGD, RBFKernel
    from numpyro.optim import Adagrad

    if kernel_fn is None:
        kernel_fn = RBFKernel()

    opt = Adagrad(step_size=lr)
    svgd = SVGD(model_fn, opt, kernel_fn, num_stein_particles=num_particles)
    result = svgd.run(rng_key, num_steps, **data)

    return FittedModel(
        model_fn=model_fn,
        method="svgd",
        params=result.params,
        guide=svgd.guide,
        losses=result.losses,
        num_particles=num_particles,
    )
