import os
import sys
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "Email Spam Classifier"
author = "Shahzada"
release = "1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # for Google/NumPy style docstrings
    "sphinx.ext.viewcode",   # adds [source] links
    "sphinx_autodoc_typehints",
]

autodoc_default_options = {"members": True, "undoc-members": False, "inherited-members": True}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
