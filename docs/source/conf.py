# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path("../..").resolve()))

# -- Project information -----------------------------------------------------

project = "EngiAI"
copyright = "2026, EngiAI Contributors"
author = "Gioele Molinari"
release = "1.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# MyST Parser configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "EngiAI Documentation"
html_baseurl = ""
html_copy_source = False
html_logo = "_static/logo.png"
html_favicon = "_static/logo_notext.png"

# Theme options - matching EngiBench style
html_theme_options = {
    "repository_url": "https://github.com/gioelemo/EngiAI",
    "repository_branch": "main",
    "path_to_docs": "docs/",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "logo": {
        "text": "EngiAI",
    },
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
}
