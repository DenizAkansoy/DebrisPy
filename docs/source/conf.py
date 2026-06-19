# Configuration file for the Sphinx documentation builder.
#

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DebrisPy'
copyright = '2025, Deniz Akansoy'
author = 'Deniz Akansoy'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',     # for Google/Numpy-style docstrings
    'sphinx.ext.viewcode',     # adds source code links
    'sphinx.ext.mathjax',      # renders math equations
    'sphinx.ext.autosummary',  # (optional) for summary tables
]

extensions += ['nbsphinx']


autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

templates_path = ['_templates']

exclude_patterns = ["**/.ipynb_checkpoints"]
nbsphinx_execute = 'auto'

# latex_elements = {
#     'preamble': r'''
#         \DeclareUnicodeCharacter{03A3}{$\Sigma$}
#         \DeclareUnicodeCharacter{03C6}{$\phi$}
#         \DeclareUnicodeCharacter{00B0}{$^\circ$}
#         \DeclareUnicodeCharacter{03BC}{$\mu$}
#         \DeclareUnicodeCharacter{2212}{$-$}
#     ''',
# }


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']