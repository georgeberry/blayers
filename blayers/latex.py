"""
Render a blayers / NumPyro model as the LaTeX of its generative form.

A blayers model is a plain NumPyro function that *adds* layers into a linear
predictor and caps it with a link.  :func:`model_to_latex` traces the model once
and prints the priors and likelihood as a block of sampling statements — the
"methods section" version of the model — so you never hand-transcribe it.

Example
-------

.. code-block:: python

    from blayers import AdaptiveLayer, InterceptLayer, gaussian_link
    from blayers.latex import model_to_latex

    def model(x, y=None):
        mu = InterceptLayer()("intercept") + AdaptiveLayer()("mu", x)
        return gaussian_link(mu, y)

    print(model_to_latex(model, x=x_train))   # raw LaTeX
    model_to_latex(model, x=x_train)           # renders inline in Jupyter

What is and isn't recovered
---------------------------

A trace exposes every ``sample`` site, its distribution, and shape, so the
**priors and likelihood are exact**.  How the layers combine into the linear
predictor (``a + b·x``) lives in plain Python, not the trace, so the output is
the standard generative model (a list of ``\\sim`` statements), not a
reconstructed regression equation.  The linear predictor is denoted
``\\eta_i``.
"""

from typing import Any, Callable

import jax
import numpy as np
from numpyro.handlers import seed, trace

from blayers import layers as _layers

# Layer class names are CamelCase (no underscores), so they can only appear as
# the first token of a site name.  Derived from BLayer subclasses so it tracks
# new layers automatically.
_LAYER_CLASSES: set[str] = {
    cls.__name__
    for cls in vars(_layers).values()
    if isinstance(cls, type)
    and issubclass(cls, _layers.BLayer)
    and cls is not _layers.BLayer
}

# suffix (last token of a blayers site name) -> LaTeX symbol (no subscript yet)
_SUFFIX_SYMBOLS: dict[str, str] = {
    "beta": r"\beta",
    "theta": r"\theta",
    "theta1": r"\theta^{(1)}",
    "theta2": r"\theta^{(2)}",
    "scale": r"\lambda",
    "scale1": r"\lambda^{(1)}",
    "scale2": r"\lambda^{(2)}",
    "tau": r"\tau",
    "c2": r"c^2",
    "z": r"z",
    "W": r"W",
    "A": r"A",
    "B": r"B",
    "weights": r"w",
    "logits": r"\gamma",
    "lengthscale": r"\ell",
    "sigma": r"\alpha",  # HSGP marginal std (namespaced under the layer)
}

# standalone sites (link parameters with no layer prefix) -> LaTeX symbol
_STANDALONE_SYMBOLS: dict[str, str] = {
    "sigma": r"\sigma",
    "gamma_shape": r"k",
    "beta_phi": r"\phi",
    "zip_gate": r"\pi",
    "zinb_gate": r"\pi",
    "zinb_concentration": r"\phi",
    "ordinal_c0": r"c_0",
    "ordinal_gaps": r"\delta",
}

# distribution class name -> (display name, ordered param attributes)
_DIST_TABLE: dict[str, tuple[str, tuple[str, ...]]] = {
    "Normal": ("Normal", ("loc", "scale")),
    "LogNormal": ("LogNormal", ("loc", "scale")),
    "Laplace": ("Laplace", ("loc", "scale")),
    "Cauchy": ("Cauchy", ("loc", "scale")),
    "StudentT": ("StudentT", ("df", "loc", "scale")),
    "HalfNormal": ("HalfNormal", ("scale",)),
    "HalfCauchy": ("HalfCauchy", ("scale",)),
    "Exponential": ("Exponential", ("rate",)),
    "Gamma": ("Gamma", ("concentration", "rate")),
    "InverseGamma": ("InverseGamma", ("concentration", "rate")),
    "Beta": ("Beta", ("concentration1", "concentration0")),
    "Dirichlet": ("Dirichlet", ("concentration",)),
    "Poisson": ("Poisson", ("rate",)),
    "NegativeBinomial2": ("NegativeBinomial2", ("mean", "concentration")),
}


class LatexStr(str):
    """A ``str`` of LaTeX that also renders inline in Jupyter.

    ``print(x)`` / ``str(x)`` give the raw LaTeX; in a notebook the object
    displays as typeset math via ``_repr_latex_``.
    """

    def _repr_latex_(self) -> str:  # pragma: no cover - notebook only
        return f"$$\n{self}\n$$"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _unwrap(fn: Any) -> Any:
    """Peel ExpandedDistribution / Independent / Masked wrappers to the base."""
    wrappers = {"ExpandedDistribution", "Independent", "MaskedDistribution"}
    while type(fn).__name__ in wrappers and hasattr(fn, "base_dist"):
        fn = fn.base_dist
    return fn


def _parse_site(name: str) -> tuple[str, str, str] | None:
    """``"AdaptiveLayer_mu_beta"`` -> ``("AdaptiveLayer", "mu", "beta")``.

    Returns ``None`` for sites that are not blayers-layer sites (link params
    such as ``sigma``, or arbitrary NumPyro sites).
    """
    parts = name.split("_")
    if len(parts) < 3:
        return None
    cls, suffix = parts[0], parts[-1]
    if cls not in _LAYER_CLASSES or suffix not in _SUFFIX_SYMBOLS:
        return None
    return cls, "_".join(parts[1:-1]), suffix


def _sub(symbol: str, layer_name: str, index: str | None = None) -> str:
    """Attach a ``\\mathrm{name}`` (optionally ``,index``) subscript.

    Keeps everything in a single ``_{...}`` group so a following per-element
    index never produces an illegal double subscript.
    """
    esc = layer_name.replace("_", r"\_")
    inner = rf"\mathrm{{{esc}}}" + (f",{index}" if index is not None else "")
    return f"{symbol}_{{{inner}}}"


def _symbol_for(name: str) -> str:
    """LaTeX symbol for any latent site name."""
    parsed = _parse_site(name)
    if parsed is not None:
        _, layer_name, suffix = parsed
        return _sub(_SUFFIX_SYMBOLS[suffix], layer_name)
    if name in _STANDALONE_SYMBOLS:
        return _STANDALONE_SYMBOLS[name]
    return r"\mathrm{" + name.replace("_", r"\_") + "}"


def _fmt_scalar(x: Any) -> str:
    """Compact rendering of a scalar prior parameter."""
    v = float(x)
    if np.isclose(v, round(v)):
        return str(int(round(v)))
    return f"{v:.3g}"


def _render_value(v: Any, value_symbols: dict[int, str]) -> str | None:
    """LaTeX for a distribution parameter value.

    * coupled to another latent site (same array object) -> that site's symbol
    * constant (broadcast) array -> the scalar
    * otherwise (a computed, data-dependent array) -> ``None``
    """
    if id(v) in value_symbols:
        return value_symbols[id(v)]
    arr = np.asarray(v)
    if arr.size == 0:
        return None
    flat = arr.reshape(-1)
    if bool(np.all(flat == flat[0])):
        return _fmt_scalar(flat[0])
    return None


def _dist_latex(
    fn: Any, value_symbols: dict[int, str], *, fallback: str = r"\cdot"
) -> str:
    """Render a (prior) distribution as ``Name(param, ...)``."""
    base = _unwrap(fn)
    cls = type(base).__name__

    if cls.startswith("Bernoulli"):
        arg = "logits" if cls.endswith("Logits") else "probs"
        val = _render_value(getattr(base, arg), value_symbols) or fallback
        return rf"\mathrm{{Bernoulli}}(\mathrm{{{arg}}}={val})"
    if cls.startswith("Categorical"):
        arg = "logits" if cls.endswith("Logits") else "probs"
        val = _render_value(getattr(base, arg), value_symbols) or fallback
        return rf"\mathrm{{Categorical}}(\mathrm{{{arg}}}={val})"

    if cls in _DIST_TABLE:
        display, attrs = _DIST_TABLE[cls]
        rendered = [
            _render_value(getattr(base, a), value_symbols) or fallback
            for a in attrs
        ]
        return rf"\mathrm{{{display}}}(" + ", ".join(rendered) + ")"

    return rf"\mathrm{{{cls}}}(\dots)"


# --------------------------------------------------------------------------- #
# Per-layer overrides (layers whose true math a flat trace can't express)
# --------------------------------------------------------------------------- #


def _render_horseshoe(
    layer_name: str, sites: dict[str, Any], value_symbols: dict[int, str]
) -> list[str]:
    tau = _sub(r"\tau", layer_name)
    lam = _sub(r"\lambda", layer_name, "j")
    beta = _sub(r"\beta", layer_name, "j")
    lines = [
        rf"{tau} &\sim \mathrm{{HalfCauchy}}(1)",
        rf"{lam} &\sim \mathrm{{HalfCauchy}}(1)",
    ]
    if "c2" in sites:
        c2 = _sub(r"c^2", layer_name)
        lines.append(rf"{c2} &\sim \mathrm{{InverseGamma}}(\nu/2,\; \nu s^2/2)")
        lines.append(
            rf"\tilde{{\lambda}}_j^2 &= \frac{{{c2}\,{lam}^2}}"
            rf"{{{c2} + {tau}^2 {lam}^2}}"
        )
        lines.append(
            rf"{beta} &\sim \mathrm{{Normal}}(0,\; {tau}\tilde{{\lambda}}_j)"
        )
    else:
        lines.append(rf"{beta} &\sim \mathrm{{Normal}}(0,\; {tau}\,{lam})")
    return lines


def _render_spike_slab(
    layer_name: str, sites: dict[str, Any], value_symbols: dict[int, str]
) -> list[str]:
    z = _sub(r"z", layer_name, "j")
    beta = _sub(r"\beta", layer_name, "j")
    z_dist = _dist_latex(sites["z"]["fn"], value_symbols)
    b_dist = _dist_latex(sites["beta"]["fn"], value_symbols)
    return [
        rf"{z} &\sim {z_dist}",
        rf"{beta} &\sim {b_dist}",
        rf"\tilde{{\beta}}_j &= {z}\,{beta} \quad\text{{(gated coefficient)}}",
    ]


def _render_mixture(
    layer_name: str, sites: dict[str, Any], value_symbols: dict[int, str]
) -> list[str]:
    lines = []
    if "logits" in sites:
        g = _sub(r"\gamma", layer_name, "k")
        w = _sub(r"w", layer_name, "k")
        lines.append(
            rf"{g} &\sim {_dist_latex(sites['logits']['fn'], value_symbols)}"
        )
        lines.append(rf"{w} &= \mathrm{{softmax}}({g})_k")
    elif "weights" in sites:
        w = _sub(r"w", layer_name)
        lines.append(
            rf"{w} &\sim {_dist_latex(sites['weights']['fn'], value_symbols)}"
        )
    beta = _sub(r"\beta", layer_name, "j")
    comps = _unwrap(sites["beta"]["fn"]).component_distributions
    parts = [
        rf"w_{k}\,{_dist_latex(c, value_symbols)}"
        for k, c in enumerate(comps, start=1)
    ]
    lines.append(rf"{beta} &\sim " + r" + ".join(parts))
    return lines


def _render_hsgp(
    layer_name: str, sites: dict[str, Any], value_symbols: dict[int, str]
) -> list[str]:
    ell = _sub(r"\ell", layer_name)
    alpha = _sub(r"\alpha", layer_name)
    beta = _sub(r"\beta", layer_name, "j")
    f = _sub(r"f", layer_name)
    lines = [
        rf"{ell} &\sim {_dist_latex(sites['lengthscale']['fn'], value_symbols)}",
        rf"{alpha} &\sim {_dist_latex(sites['sigma']['fn'], value_symbols)}",
        rf"{beta} &\sim \mathrm{{Normal}}(0, 1)",
        rf"{f}(x) &= \textstyle\sum_j \phi_j(x)\,"
        rf"\sqrt{{S_{{{alpha},{ell}}}(\sqrt{{\lambda_j}})}}\,{beta}",
    ]
    return lines


_OVERRIDES: dict[
    str, Callable[[str, dict[str, Any], dict[int, str]], list[str]]
] = {
    "HorseshoeLayer": _render_horseshoe,
    "HorseshoeInteractionLayer": _render_horseshoe,
    "SpikeAndSlabLayer": _render_spike_slab,
    "MixtureLayer": _render_mixture,
    "HSGPLayer": _render_hsgp,
}


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #


def _render_likelihood(
    fn: Any,
    value_symbols: dict[int, str],
    scalar_latents: list[tuple[float, str]],
) -> str:
    """LaTeX RHS for the observed ``obs`` site, using ``\\eta_i`` for the
    linear predictor and matched symbols for auxiliary parameters."""
    base = _unwrap(fn)
    cls = type(base).__name__
    eta = r"\eta_i"

    def aux(attr: str, default: str) -> str:
        # Auxiliary likelihood parameters (scale, concentration) are latent
        # scalars. Prefer the coupled site symbol; otherwise match by value
        # against the scalar latents (NumPyro often broadcasts the scale into a
        # fresh array, breaking object identity) so the symbol stays consistent
        # with the site's own prior line. Fall back to the conventional symbol —
        # never the raw sampled number.
        v = getattr(base, attr, None)
        if v is None:
            return default
        if id(v) in value_symbols:
            return value_symbols[id(v)]
        arr = np.asarray(v).reshape(-1)
        if arr.size >= 1 and bool(np.all(arr == arr[0])):
            for val, sym in scalar_latents:
                if np.isclose(float(arr[0]), val):
                    return sym
        return default

    def const(attr: str, default: str) -> str:
        # A genuinely fixed hyperparameter (e.g. StudentT df) — show its value.
        rendered = _render_value(getattr(base, attr, None), value_symbols)
        return rendered if rendered is not None else default

    sigma = aux("scale", r"\sigma")
    if cls in ("Normal", "Laplace", "Cauchy"):
        return rf"\mathrm{{{cls}}}({eta},\; {sigma})"
    if cls == "LogNormal":
        return rf"\mathrm{{LogNormal}}({eta},\; {sigma})"
    if cls == "StudentT":
        df = const("df", "4")
        return rf"\mathrm{{StudentT}}({df},\; {eta},\; {sigma})"
    if cls.startswith("Bernoulli"):
        return rf"\mathrm{{Bernoulli}}\!\left(\mathrm{{logit}}^{{-1}}({eta})\right)"
    if cls.startswith("Categorical"):
        return (
            r"\mathrm{Categorical}\!\left(\mathrm{softmax}"
            r"(\boldsymbol{\eta}_i)\right)"
        )
    if cls == "Poisson":
        return rf"\mathrm{{Poisson}}(e^{{{eta}}})"
    if cls == "NegativeBinomial2":
        phi = aux("concentration", r"\phi")
        return rf"\mathrm{{NegativeBinomial2}}(e^{{{eta}}},\; {phi})"
    if cls == "Gamma":
        return rf"\mathrm{{Gamma}}(k,\; k\,e^{{-{eta}}})"
    if cls == "Exponential":
        return rf"\mathrm{{Exponential}}(e^{{-{eta}}})"
    if cls == "Beta":
        return (
            r"\mathrm{Beta}(\bar{\mu}_i\phi,\; (1-\bar{\mu}_i)\phi),\quad "
            rf"\bar{{\mu}}_i = \mathrm{{logit}}^{{-1}}({eta})"
        )
    if cls.startswith("ZeroInflated"):
        inner = _unwrap(getattr(base, "base_dist", base))
        inner_name = type(inner).__name__
        return (
            rf"\mathrm{{ZeroInflated}}\text{{-}}{inner_name}"
            rf"(e^{{{eta}}};\; \pi)"
        )
    return rf"\mathrm{{{cls}}}({eta})"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def model_to_latex(
    model_fn: Callable[..., Any],
    *,
    env: str = "align",
    seed_value: int = 0,
    **data: Any,
) -> LatexStr:
    """Render a blayers / NumPyro model as the LaTeX of its generative form.

    Traces ``model_fn`` once (with no observed ``y``) and emits a block of
    sampling statements: every latent's prior, then the likelihood.  Hierarchy
    is recovered automatically — a coefficient whose scale is a sampled site
    prints ``\\sim \\mathrm{Normal}(0, \\lambda_{\\mathrm{name}})`` rather than a
    number.

    Parameters
    ----------
    model_fn : Callable
        A blayers / NumPyro model.  Pass inputs the same way you would to
        :func:`blayers.fit`, but **without** ``y`` (supplying it would condition
        the ``obs`` site; the likelihood is rendered from its distribution
        family regardless).
    env : ``"align"``, ``"aligned"``, or ``"none"``
        LaTeX environment to wrap the lines in.  ``"none"`` returns the bare
        ``\\\\``-separated body.
    seed_value : int
        PRNG seed for the trace (only affects sampled shapes, not the output).
    **data
        Model inputs (e.g. ``x``) and constants, used to fix site shapes.

    Returns
    -------
    LatexStr
        The LaTeX source (a ``str`` subclass that also renders inline in
        Jupyter).

    Examples
    --------
    >>> print(model_to_latex(model, x=x_train))
    \\begin{align}
    ...
    \\end{align}
    """
    if "y" in data:
        raise ValueError(
            "model_to_latex() renders the generative model; do not pass `y` "
            "(it would condition the obs site). Pass only model inputs."
        )

    tr = trace(seed(model_fn, jax.random.PRNGKey(seed_value))).get_trace(**data)

    latent = {
        name: site
        for name, site in tr.items()
        if site["type"] == "sample"
        and not site["is_observed"]
        and name != "obs"
    }

    # Map each latent site's *value object* to its symbol so downstream
    # distributions that reuse it (β's scale = λ) render symbolically.
    value_symbols: dict[int, str] = {
        id(site["value"]): _symbol_for(name) for name, site in latent.items()
    }
    # Scalar latents, for value-based matching of likelihood auxiliaries whose
    # object identity was lost to broadcasting (e.g. sigma, concentration).
    scalar_latents: list[tuple[float, str]] = [
        (float(np.asarray(site["value"]).reshape(-1)[0]), _symbol_for(name))
        for name, site in latent.items()
        if np.asarray(site["value"]).size == 1
    ]

    lines: list[str] = []

    # Group blayers-layer sites by (class, name); keep everything else as a
    # standalone latent.  Preserve trace order throughout.
    seen_groups: set[tuple[str, str]] = set()
    for name, site in latent.items():
        parsed = _parse_site(name)
        if parsed is None:
            sym = _symbol_for(name)
            lines.append(
                rf"{sym} &\sim {_dist_latex(site['fn'], value_symbols)}"
            )
            continue
        cls, layer_name, _ = parsed
        key = (cls, layer_name)
        if key in seen_groups:
            continue
        seen_groups.add(key)
        group = {
            _parse_site(n)[2]: s  # type: ignore[index]
            for n, s in latent.items()
            if _parse_site(n) is not None
            and _parse_site(n)[:2] == (cls, layer_name)  # type: ignore[index]
        }
        if cls in _OVERRIDES:
            lines.extend(_OVERRIDES[cls](layer_name, group, value_symbols))
        else:
            for suffix, s in group.items():
                sym = _sub(_SUFFIX_SYMBOLS[suffix], layer_name)
                lines.append(
                    rf"{sym} &\sim {_dist_latex(s['fn'], value_symbols)}"
                )

    # Likelihood
    if "obs" in tr:
        rhs = _render_likelihood(tr["obs"]["fn"], value_symbols, scalar_latents)
        lines.append(rf"y_i &\sim {rhs},\quad i = 1, \dots, n")

    body = " \\\\\n".join(lines)
    if env == "none":
        return LatexStr(body)
    return LatexStr(f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}")
