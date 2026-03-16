"""
core/paths.py
=============
Single source of truth for all output directories.

All figures and gephi files save directly into:
    <project_root>/outputs/          ← figures (.png, .gif)
    <project_root>/outputs/gephi/    ← .gexf files

Every experiment and visualization imports get_output_dir() from here.
Importing this module also sets the matplotlib backend to Agg so that
plt.show() never blocks and all figures save silently — in every script,
not just when called from run_all.py.
"""

from __future__ import annotations
import os

# ── Set non-interactive backend immediately on import ─────────────────────────
# This must happen before any other matplotlib import anywhere in the project.
# Effect: plt.show() becomes a no-op; plt.savefig() still works perfectly.
import matplotlib
matplotlib.use('Agg')

from datetime import datetime

# ── Project root and outputs dir ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUTS_ROOT = os.path.join(PROJECT_ROOT, 'outputs')

# Create on import so it always exists
os.makedirs(OUTPUTS_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_ROOT, 'gephi'), exist_ok=True)

# Module-level variable — kept for backward compat but now always points to OUTPUTS_ROOT
_run_folder: str = OUTPUTS_ROOT


def set_run_folder(name: str | None = None) -> str:
    """
    Now simply returns the fixed outputs/ folder.
    The `name` argument is accepted but ignored — kept so existing
    call sites in run_all.py don't break.
    """
    global _run_folder
    _run_folder = OUTPUTS_ROOT
    return _run_folder


def get_output_dir(subdir: str = '') -> str:
    """
    Return the output directory for saving figures.
    All files go into <project_root>/outputs/
    Gephi files go into <project_root>/outputs/gephi/
    """
    base = os.path.join(OUTPUTS_ROOT, subdir) if subdir else OUTPUTS_ROOT
    os.makedirs(base, exist_ok=True)
    return base


def get_run_folder() -> str:
    return _run_folder


def is_interactive() -> bool:
    """
    Always returns False — matplotlib.use('Agg') is set on import of this
    module, so plt.show() is always a no-op. Figures save silently via
    plt.savefig() in every script without any pop-up windows.
    """
    return False