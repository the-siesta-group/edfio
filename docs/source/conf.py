import inspect
import subprocess
import sys
from pathlib import Path

import edfio

REPOSITORY_URL = "https://github.com/the-siesta-group/edfio"
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()

project = "edfio"
copyright = "2023, The Siesta Group"
author = "The Siesta Group"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "numpydoc",
    "myst_parser",
]

html_theme = "pydata_sphinx_theme"
html_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_theme_options = {
    "github_url": REPOSITORY_URL,
    "show_toc_level": 2,
    "pygment_dark_style": "github-dark",
    "footer_start": [
        "last-updated",
        "copyright",
    ],
    "footer_end": [
        "sphinx-version",
        "theme-version",
    ],
}

html_last_updated_fmt = "%Y-%m-%d"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/devdocs", None),
}

templates_path = ["_templates"]

default_role = "code"

autodoc_typehints = "none"
autodoc_default_options = {"members": True, "inherited-members": True}
autosummary_generate = True

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "Path": "pathlib.Path",
    "Sequence": "collections.abc.Sequence",
    "Iterable": "collections.abc.Sequence",
    "npt.NDArray": "numpy.ndarray",
}


def linkcode_resolve(domain, info):  # noqa: D103
    project_root = Path(edfio.__file__).parent.parent
    if domain != "py" or not info["module"]:
        return None
    obj = sys.modules[info["module"]]
    parent = obj
    for part in info["fullname"].split("."):
        parent = obj
        obj = getattr(obj, part)
    if isinstance(obj, property):
        obj = obj.fget
    try:
        inspect.getsourcefile(obj)
    except TypeError:
        obj = parent
    filename = Path(inspect.getsourcefile(obj)).relative_to(project_root)
    sourcelines, startline = inspect.getsourcelines(obj)
    endline = startline + len(sourcelines) - 1
    return f"{REPOSITORY_URL}/blob/{COMMIT}/{filename}#L{startline}-L{endline}"


def autodoc_process_docstring(app, what, name, obj, options, lines):  # noqa: D103
    def edf_to_bdf(line):
        line = line.replace("Edf", "Bdf")
        line = line.replace("EDF+", "BDF+")
        line = line.replace("See :class:`Bdf", "See :class:`Edf")
        return line.replace("BdfAnnotation", "EdfAnnotation")

    if name.startswith("edfio.Bdf"):
        lines[:] = [edf_to_bdf(l) for l in lines]


def setup(app):  # noqa: D103
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
