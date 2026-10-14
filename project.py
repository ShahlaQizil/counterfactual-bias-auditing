"""
Project layout, shared by every script.

In this folder the code is split by role:

    Counterfactual-Bias-Auditing/
        The paper/                                                  paper source, builds, previews
        Code that runs the experiments/
        Code that produces the paper's numbers, tables and figure/
        Build and release scripts/
        Results and data/                                           results_*/, data/, release/

Every script starts with

    import os, sys
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [_here, os.path.dirname(_here)]
    import project
    project.setup()

which puts all code folders on the import path and makes "Results and data/"
the working directory, so the scripts' relative paths (results_*/, data/)
resolve as before. In the flat public release (all files in one folder) the
same header works: there is no "Results and data/" folder, so everything is
relative to the release folder itself.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(ROOT, "Results and data")
PROJECT_LAYOUT = os.path.isdir(_RESULTS)

CODE_DIRS = [os.path.join(ROOT, d) for d in (
    "Code that runs the experiments",
    "Code that produces the paper's numbers, tables and figure",
    "Build and release scripts",
)] if PROJECT_LAYOUT else [ROOT]

RESULTS = _RESULTS if PROJECT_LAYOUT else ROOT
PAPER = os.path.join(ROOT, "The paper") if PROJECT_LAYOUT else ROOT
# table rows and the figure that the paper source loads
PAPER_TABLES = os.path.join(PAPER, "figures_tables") if PROJECT_LAYOUT else os.path.join(ROOT, "paper_tables")


def setup() -> None:
    """Make all code folders importable and run from the results folder."""
    for d in reversed(CODE_DIRS):
        if d not in sys.path:
            sys.path.insert(0, d)
    os.chdir(RESULTS)


def script(name: str) -> str:
    """Full path of a project script, wherever it lives."""
    for d in CODE_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)
