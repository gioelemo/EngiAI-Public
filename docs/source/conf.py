# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path("../..").resolve()))

# -- Project information -----------------------------------------------------

project = "Engineer Assistant"
copyright = "2025, Engineer Assistant Contributors"
author = "Gioele Molinari"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_github_changelog",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True

# MyST Parser configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "Engineer Assistant Documentation"
html_baseurl = ""
html_copy_source = False

# Theme options - matching EngiBench style
html_theme_options = {
    "repository_url": "https://github.com/yourusername/engineer-assistant",  # Update with your repo URL
    "repository_branch": "main",
    "path_to_docs": "docs/",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
}

html_static_path = ["_static"]
html_css_files: list[str] = []

# Add version information to the context
html_context = {
    "version_info": {
        "current": release,
        "versions": {
            "main": "/",
        },
    }
}

# Add version switcher to the left sidebar
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "search-field.html",
        "sbt-sidebar-nav.html",
    ]
}

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "langchain": ("https://api.python.langchain.com/en/latest/", None),
}

# GitHub Changelog
sphinx_github_changelog_token = os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN")
