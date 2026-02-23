"""
grdl-rt UI — Lightweight workflow runner GUI for component developers.

Provides a tkinter + matplotlib desktop application for executing GRDL
workflows and individual processor components against imagery with
real-time progress, parameter tuning, metrics visualisation, and
optional ground truth accuracy comparison.

Zero additional dependencies beyond the standard library (tkinter) and
matplotlib (already present in the runtime environment).

Usage::

    # Programmatic launch
    from grdl_rt.ui import launch
    launch()

    # CLI launch
    $ grdl-rt ui
    $ grdl-rt ui --workflow path/to/workflow.yaml --input path/to/image.nitf

    # Module invocation
    $ python -m grdl_rt.ui

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-13
"""

from __future__ import annotations

import argparse
import sys

__all__ = ["launch", "main"]


def launch(
    *,
    workflow: str | None = None,
    input_path: str | None = None,
) -> None:
    """Launch the grdl-rt runner GUI.

    Parameters
    ----------
    workflow : str, optional
        Path to a workflow YAML or component Python file to pre-load.
    input_path : str, optional
        Path to an input image to pre-load.
    """
    # Lazy imports so that tkinter/matplotlib are only pulled in when
    # the GUI is actually requested.
    from grdl_rt.ui._app import App

    app = App(workflow_path=workflow, input_path=input_path)
    app.mainloop()


def main(argv=None) -> int:
    """CLI entry point for the runner GUI.

    Parses optional ``--workflow`` and ``--input`` arguments, then
    launches the application.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (always 0 on normal exit).
    """
    parser = argparse.ArgumentParser(
        prog="grdl-rt-ui",
        description="grdl-rt Runner — lightweight workflow execution GUI",
    )
    parser.add_argument(
        "--workflow",
        "-w",
        type=str,
        default=None,
        help="Path to a workflow YAML or component .py file to pre-load.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to an input image file to pre-load.",
    )

    args = parser.parse_args(argv)
    launch(workflow=args.workflow, input_path=args.input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
