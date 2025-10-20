# PROJECT INFORMATION
project = 'SPAM vs HAM'


# GENERAL CONFIGURATION
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',]

templates_path = ['_templates']
exclude_patterns = []

# ADD PATHS SO SPHINX CAN FIND THE CODE | src FOLDER
import os, sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src'))

# HTML OUTPUT THEME
html_theme = 'furo'
html_static_path = ['_static']
