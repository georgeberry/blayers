"""Tests for blayers.latex.model_to_latex."""

import jax.numpy as jnp
import numpyro.distributions as d
import pytest
from numpyro import sample

from blayers.latex import LatexStr, model_to_latex
from blayers.layers import (
    AdaptiveLayer,
    FixedEffectsLayer,
    HorseshoeInteractionLayer,
    HorseshoeLayer,
    HSGPLayer,
    InterceptLayer,
    MixtureLayer,
    RandomEffectsLayer,
)
from blayers.links import (
    beta_link,
    categorical_link,
    exponential_link,
    gamma_link,
    gaussian_link,
    logit_link,
    lognormal_link,
    negative_binomial_link,
    ordinal_link,
    poisson_link,
    student_t_link,
    zinb_link,
    zip_link,
)

X = jnp.ones((6, 3))
G = jnp.array([0, 1, 2, 0, 1, 2]).reshape(-1, 1)
XC = jnp.linspace(-2.0, 2.0, 6)


def _has_double_subscript(s: str) -> bool:
    """True if a subscript is immediately followed by another (illegal LaTeX).

    Flags ``x_{...}_y`` and ``x_a_b`` but not ``\\tilde{\\beta}_j`` (where the
    braces belong to ``\\tilde``, not to a subscript).
    """
    i = 0
    while i < len(s):
        if s[i] != "_":
            i += 1
            continue
        j = i + 1
        if j < len(s) and s[j] == "{":
            depth, j = 1, j + 1
            while j < len(s) and depth:
                depth += {"{": 1, "}": -1}.get(s[j], 0)
                j += 1
        else:
            j += 1
        if j < len(s) and s[j] == "_":
            return True
        i = j
    return False


# --------------------------------------------------------------------------- #
# Core behaviour
# --------------------------------------------------------------------------- #


def test_hierarchy_is_recovered():
    """A sampled scale feeding a coefficient prints symbolically, not numerically."""

    def model(x, y=None):
        return gaussian_link(AdaptiveLayer()("mu", x), y)

    tex = model_to_latex(model, x=X)
    # scale line
    assert r"\lambda_{\mathrm{mu}} &\sim \mathrm{HalfNormal}(1)" in tex
    # coefficient couples to the scale symbol (not a sampled number)
    assert (
        r"\beta_{\mathrm{mu}} &\sim \mathrm{Normal}(0, \lambda_{\mathrm{mu}})"
        in tex
    )
    # likelihood uses the linear-predictor symbol + plate
    assert r"y_i &\sim \mathrm{Normal}(\eta_i,\; \sigma)" in tex
    assert r"i = 1, \dots, n" in tex


def test_returns_latexstr_and_env_wrapping():
    def model(x, y=None):
        return gaussian_link(AdaptiveLayer()("mu", x), y)

    tex = model_to_latex(model, x=X)
    assert isinstance(tex, LatexStr)
    assert tex.startswith(r"\begin{align}")
    assert tex.endswith(r"\end{align}")
    assert tex._repr_latex_().startswith("$$")

    aligned = model_to_latex(model, x=X, env="aligned")
    assert aligned.startswith(r"\begin{aligned}")

    bare = model_to_latex(model, x=X, env="none")
    assert "begin" not in bare
    assert r"\sim" in bare


def test_multiple_layers_and_intercept():
    def model(x, g, y=None):
        mu = (
            InterceptLayer()("intercept")
            + AdaptiveLayer()("mu", x)
            + RandomEffectsLayer()("grp", g, num_categories=3)
        )
        return gaussian_link(mu, y)

    tex = model_to_latex(model, x=X, g=G)
    assert r"\beta_{\mathrm{intercept}}" in tex
    assert (
        r"\theta_{\mathrm{grp}} &\sim \mathrm{Normal}(0, \lambda_{\mathrm{grp}})"
        in tex
    )


def test_passing_y_raises():
    def model(x, y=None):
        return gaussian_link(AdaptiveLayer()("mu", x), y)

    with pytest.raises(ValueError, match="do not pass `y`"):
        model_to_latex(model, x=X, y=jnp.ones(6))


# --------------------------------------------------------------------------- #
# Likelihood families
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "link, expect",
    [
        (
            logit_link,
            r"\mathrm{Bernoulli}\!\left(\mathrm{logit}^{-1}(\eta_i)\right)",
        ),
        (poisson_link, r"\mathrm{Poisson}(e^{\eta_i})"),
        (exponential_link, r"\mathrm{Exponential}(e^{-\eta_i})"),
        (gamma_link, r"\mathrm{Gamma}(k,\; k\,e^{-\eta_i})"),
        (lognormal_link, r"\mathrm{LogNormal}(\eta_i,\; \sigma)"),
    ],
)
def test_likelihood_families(link, expect):
    def model(x, y=None):
        return link(AdaptiveLayer()("mu", x), y)

    assert expect in model_to_latex(model, x=X)


def test_categorical_likelihood():
    def model(x, y=None):
        return categorical_link(AdaptiveLayer()("mu", x, units=4), y)

    assert r"\mathrm{softmax}" in model_to_latex(model, x=X)


def test_negative_binomial_symbol_is_consistent():
    """The concentration site and its use in the likelihood share one symbol."""

    def model(x, y=None):
        return negative_binomial_link(AdaptiveLayer()("mu", x), y)

    tex = model_to_latex(model, x=X)
    # sigma is the literal site name of the NB2 concentration; it must appear
    # both on its prior line and inside the likelihood (no unbound phi).
    assert r"\sigma &\sim \mathrm{Exponential}(1)" in tex
    assert r"\mathrm{NegativeBinomial2}(e^{\eta_i},\; \sigma)" in tex


# --------------------------------------------------------------------------- #
# Special-layer overrides
# --------------------------------------------------------------------------- #


def test_horseshoe_override():
    def model(x, y=None):
        return gaussian_link(HorseshoeLayer()("hs", x), y)

    tex = model_to_latex(model, x=X)
    assert r"\tau_{\mathrm{hs}} &\sim \mathrm{HalfCauchy}(1)" in tex
    assert r"\lambda_{\mathrm{hs},j} &\sim \mathrm{HalfCauchy}(1)" in tex
    assert r"\tau_{\mathrm{hs}}\,\lambda_{\mathrm{hs},j}" in tex
    # no illegal double subscript
    assert not _has_double_subscript(tex)


def test_regularized_horseshoe_has_slab_line():
    def model(x, y=None):
        return gaussian_link(HorseshoeLayer(slab_scale=2.0)("hs", x), y)

    tex = model_to_latex(model, x=X)
    assert r"c^2_{\mathrm{hs}}" in tex
    assert r"\tilde{\lambda}_j" in tex


def test_horseshoe_interaction_override():
    def model(x, y=None):
        return gaussian_link(HorseshoeInteractionLayer()("int", x), y)

    tex = model_to_latex(model, x=X)
    assert r"\tau_{\mathrm{int}} &\sim \mathrm{HalfCauchy}(1)" in tex
    assert r"\lambda_{\mathrm{int},j} &\sim \mathrm{HalfCauchy}(1)" in tex
    assert not _has_double_subscript(tex)


def test_mixture_override():
    def model(x, y=None):
        return gaussian_link(MixtureLayer()("mx", x), y)

    tex = model_to_latex(model, x=X)
    assert r"\gamma_{\mathrm{mx},k} &\sim \mathrm{Normal}" in tex
    assert r"w_{\mathrm{mx},k} &= \mathrm{softmax}" in tex
    assert r"w_1\,\mathrm{Normal}(0, 1) + w_2\,\mathrm{Laplace}(0, 1)" in tex


def test_hsgp_override():
    def model(xc, y=None):
        return gaussian_link(HSGPLayer()("gp", xc, L=3.0, m=8), y)

    tex = model_to_latex(model, xc=XC)
    assert r"\ell_{\mathrm{gp}} &\sim \mathrm{InverseGamma}(5, 5)" in tex
    assert r"\alpha_{\mathrm{gp}} &\sim \mathrm{HalfNormal}(1)" in tex
    assert r"f_{\mathrm{gp}}(x) &=" in tex


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #


def test_no_double_subscripts_anywhere():
    """`x_{...}_j` is invalid LaTeX; make sure no override emits it."""

    def model(x, xc, g, y=None):
        mu = (
            HorseshoeLayer()("hs", x)
            + MixtureLayer()("mx", x)
            + HSGPLayer()("gp", xc, L=3.0, m=6)
            + FixedEffectsLayer()("fe", g, num_categories=3)
        )
        return gaussian_link(mu, y)

    tex = model_to_latex(model, x=X, xc=XC, g=G)
    assert not _has_double_subscript(tex)


def test_plain_numpyro_model_degrades_gracefully():
    """Non-blayers sites render with escaped names and never raise."""

    def model(x, y=None):
        a = sample("alpha", d.Normal(0, 1))
        b = sample("beta_coef", d.Normal(0, 1).expand([x.shape[1]]))
        mu = a + x @ b
        return sample("obs", d.Normal(mu, 1.0), obs=y)

    tex = model_to_latex(model, x=X)
    assert r"\mathrm{alpha} &\sim \mathrm{Normal}(0, 1)" in tex
    assert r"\mathrm{beta\_coef}" in tex


def test_off_table_distributions_use_generic_form():
    """Priors / likelihoods not in the symbol tables degrade to a generic form."""

    def model(x, y=None):
        w = sample("weird", d.Gumbel(0.0, 1.0))  # off-table prior
        mu = w + x[:, 0]
        return sample("obs", d.Gumbel(mu, 1.0), obs=y)  # off-table likelihood

    tex = model_to_latex(model, x=X)
    assert r"\mathrm{Gumbel}(\dots)" in tex  # generic prior fallback
    assert r"\mathrm{Gumbel}(\eta_i)" in tex  # generic likelihood fallback


@pytest.mark.parametrize(
    "link",
    [
        gaussian_link,
        student_t_link,
        logit_link,
        poisson_link,
        negative_binomial_link,
        gamma_link,
        exponential_link,
        beta_link,
        zip_link,
        zinb_link,
        lambda mu, y=None: ordinal_link(mu, y, num_classes=3),
    ],
)
def test_every_link_is_non_raising(link):
    def model(x, y=None):
        return link(AdaptiveLayer()("mu", x), y)

    tex = model_to_latex(model, x=X)
    assert tex.startswith(r"\begin{align}")
    assert r"y_i &\sim" in tex
