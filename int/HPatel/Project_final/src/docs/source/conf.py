# Email Spam or Ham Checker Documentation
# Author: H.Patel

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Email Spam or Ham Checker'
author = 'H.Patel'
release = '1.0'
copyright = f'2025, {author}'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'sphinx_copybutton',
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = []

# HTML Theme Setup
html_theme = 'furo'
html_title = 'Email Spam or Ham Checker'
html_static_path = ['_static']
html_logo = '_static/logo.png'  # Optional: add your logo file here

# Custom look tweaks
html_theme_options = {
    "light_logo": "logo.png",
    "dark_logo": "logo.png",
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
html_css_files = ['custom.css']
