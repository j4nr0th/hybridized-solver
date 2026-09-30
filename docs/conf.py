"""Sphinx configuration for the hybsol documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../python"))

project = "hybsol"
author = "Jan Roth"
copyright = "2026, Jan Roth"
release = "0.0.1a"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "hawkmoth",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "hybsol"

# -- Options for the Python API ----------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
python_use_unqualified_type_names = True

napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for hawkmoth (the C API) -----------------------------------------
# The public headers are self-contained and need no external include paths.
hawkmoth_root = os.path.abspath("..")
hawkmoth_clang = [f"-I{os.path.abspath('../include')}", "-DHYBSOL_DOCS"]

# -- Options for sphinx-gallery (the examples) ---------------------------------
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r".*\.py",
    "ignore_pattern": r"_utils\.py",
    "remove_config_comments": True,
    "abort_on_example_error": True,
}
