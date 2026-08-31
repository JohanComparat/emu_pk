"""Sphinx configuration for emu_pk.

The docs build on ReadTheDocs, which cannot compile a Boltzmann solver, so this
must work with the **core install only** -- numpy and jax.  Nothing here may
import `classy` or `optax`, and every tutorial figure is a committed artefact
produced by `docs/make_figures.py` rather than generated at build time.
"""
import importlib.metadata

project = "emu_pk"
author = "Johan Comparat"
copyright = "2026, Johan Comparat"
try:
    release = importlib.metadata.version("emu_pk")
except importlib.metadata.PackageNotFoundError:  # building from a source tree
    import re
    import pathlib
    release = re.search(r'__version__ = "([^"]+)"',
                        (pathlib.Path(__file__).parent.parent
                         / "emu_pk" / "__init__.py").read_text()).group(1)
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

# Markdown throughout, matching the repository's prose.
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"emu_pk {release}"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
# `classy` and `optax` are extras; autodoc must not need them to document the
# modules that use them.
autodoc_mock_imports = ["classy", "optax"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

# A warning is a broken cross-reference or a malformed directive; CI builds
# with -W so neither can reach the published site.
nitpicky = False
