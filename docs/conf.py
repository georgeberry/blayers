# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "blayers"
copyright = "2025, George Berry"
author = "George Berry"
release = "v0.1.0a5"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

autodoc_typehints = "description"
# autodoc_type_aliases = {
#    'Distribution': 'numpyro.distributions.Distribution',
#    'HalfNormal': 'numpyro.distributions.HalfNormal',
#    'Normal': 'numpyro.distributions.Normal',
# }
autodoc_default_options = {
    "members": True,
    "special-members": "__init__, __call__",
    "undoc-members": True,
    "no-value": True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/georgeberry/blayers",
    "use_repository_button": True,
    "use_edit_page_button": True,
}

autodoc_member_order = "bysource"
