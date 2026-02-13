# -*- coding: utf-8 -*-
"""
Reusable Widgets — Tkinter building blocks for the grdl-rt runner GUI.

Provides file-picker rows, labeled entries, a scrollable log console,
a progress-bar wrapper, and a scrollable frame — all zero-dependency
beyond the standard library and ttk.

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

import tkinter as tk
from tkinter import filedialog, ttk
from typing import List, Optional

# ── File-type filter groups ──────────────────────────────────────────

IMAGE_FILETYPES = [
    ("All images", "*.nitf *.ntf *.tif *.tiff *.npy *.png *.jpg *.jpeg *.bmp"),
    ("NITF", "*.nitf *.ntf"),
    ("GeoTIFF / TIFF", "*.tif *.tiff"),
    ("NumPy array", "*.npy"),
    ("PNG", "*.png"),
    ("JPEG", "*.jpg *.jpeg"),
    ("All files", "*.*"),
]

WORKFLOW_FILETYPES = [
    ("Workflow / Component", "*.yaml *.yml *.py"),
    ("YAML workflow", "*.yaml *.yml"),
    ("Python component", "*.py"),
    ("All files", "*.*"),
]

GEOJSON_FILETYPES = [
    ("GeoJSON", "*.geojson *.json"),
    ("All files", "*.*"),
]

IMAGE_EXTENSIONS = {
    ".nitf", ".ntf", ".tif", ".tiff", ".npy",
    ".png", ".jpg", ".jpeg", ".bmp",
}


# ── FilePickerRow ────────────────────────────────────────────────────


class FilePickerRow(ttk.Frame):
    """Label + Entry + Browse button for selecting files or directories.

    Parameters
    ----------
    parent : tk widget
        Parent container.
    label : str
        Label text shown to the left of the entry.
    mode : str
        ``"file"`` for single file, ``"files"`` for multiple files,
        ``"directory"`` for folder selection.
    filetypes : list, optional
        File type filter tuples for the dialog.  Ignored when
        *mode* is ``"directory"``.
    """

    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        mode: str = "file",
        filetypes: Optional[list] = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._filetypes = filetypes or [("All files", "*.*")]

        self.var = tk.StringVar()

        lbl = ttk.Label(self, text=label, width=22, anchor="w")
        lbl.grid(row=0, column=0, sticky="w", padx=(0, 4))

        self._entry = ttk.Entry(self, textvariable=self.var, width=48)
        self._entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))

        btn = ttk.Button(self, text="Browse\u2026", width=9, command=self._browse)
        btn.grid(row=0, column=2)

        self.columnconfigure(1, weight=1)

    # ── public helpers ───────────────────────────────────────────

    def get(self) -> str:
        """Return the current path string."""
        return self.var.get().strip()

    def set(self, value: str) -> None:
        """Set the entry value programmatically."""
        self.var.set(value)

    def get_paths(self) -> List[str]:
        """Return list of paths (splits on ``';'`` for multi-file mode)."""
        raw = self.get()
        if not raw:
            return []
        return [p.strip() for p in raw.split(";") if p.strip()]

    # ── internals ────────────────────────────────────────────────

    def _browse(self) -> None:
        if self._mode == "directory":
            path = filedialog.askdirectory(title="Select Folder")
            if path:
                self.var.set(path)
        elif self._mode == "files":
            paths = filedialog.askopenfilenames(
                title="Select File(s)", filetypes=self._filetypes,
            )
            if paths:
                self.var.set("; ".join(paths))
        else:
            path = filedialog.askopenfilename(
                title="Select File", filetypes=self._filetypes,
            )
            if path:
                self.var.set(path)


# ── LabeledEntry ─────────────────────────────────────────────────────


class LabeledEntry(ttk.Frame):
    """Label + Entry pair with convenient get/set.

    Parameters
    ----------
    parent : tk widget
        Parent container.
    label : str
        Label text.
    default : str
        Initial entry value.
    width : int
        Entry widget width in characters.
    """

    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        default: str = "",
        width: int = 12,
    ) -> None:
        super().__init__(parent)
        self.var = tk.StringVar(value=default)

        lbl = ttk.Label(self, text=label, anchor="w")
        lbl.pack(side="left", padx=(0, 4))

        self._entry = ttk.Entry(self, textvariable=self.var, width=width)
        self._entry.pack(side="left", fill="x", expand=True)

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, value: str) -> None:
        self.var.set(value)


# ── LabeledCombobox ──────────────────────────────────────────────────


class LabeledCombobox(ttk.Frame):
    """Label + Combobox pair.

    Parameters
    ----------
    parent : tk widget
        Parent container.
    label : str
        Label text.
    values : list of str
        Combobox dropdown values.
    default : str, optional
        Initial selected value.
    """

    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        values: List[str],
        default: str = "",
    ) -> None:
        super().__init__(parent)
        self.var = tk.StringVar(value=default)

        lbl = ttk.Label(self, text=label, anchor="w")
        lbl.pack(side="left", padx=(0, 4))

        self._combo = ttk.Combobox(
            self, textvariable=self.var, values=values,
            state="readonly", width=18,
        )
        self._combo.pack(side="left", fill="x", expand=True)

    def get(self) -> str:
        return self.var.get()

    def set(self, value: str) -> None:
        self.var.set(value)


# ── LogConsole ───────────────────────────────────────────────────────


class LogConsole(ttk.Frame):
    """Scrolled read-only text widget for log output.

    Supports tagged messages with colour coding for ``info``,
    ``warn``, and ``error`` levels.
    """

    _TAG_COLOURS = {
        "info": "#c8c8c8",
        "warn": "#e6a800",
        "error": "#e64545",
        "success": "#4caf50",
    }

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)

        self._text = tk.Text(
            self, wrap="word", state="disabled",
            bg="#1e1e1e", fg="#d4d4d4",
            font=("Consolas", 9), relief="flat",
            borderwidth=0, highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=scrollbar.set)

        self._text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Configure tag colours
        for tag, colour in self._TAG_COLOURS.items():
            self._text.tag_configure(tag, foreground=colour)

    def append(self, message: str, tag: str = "info") -> None:
        """Append a line to the console.

        Parameters
        ----------
        message : str
            Log line text.
        tag : str
            Colour tag — ``"info"``, ``"warn"``, ``"error"``, ``"success"``.
        """
        self._text.configure(state="normal")
        self._text.insert("end", message + "\n", tag)
        self._text.see("end")
        self._text.configure(state="disabled")

    def clear(self) -> None:
        """Remove all text."""
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")


# ── ProgressBar ──────────────────────────────────────────────────────


class ProgressBar(ttk.Frame):
    """Thin wrapper around ttk.Progressbar with determinate/indeterminate modes."""

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self._bar = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self._bar.pack(fill="x", expand=True)

    def set_fraction(self, fraction: float) -> None:
        """Set progress to a value between 0.0 and 1.0."""
        self._bar.configure(mode="determinate")
        self._bar["value"] = max(0.0, min(1.0, fraction)) * 100

    def set_indeterminate(self) -> None:
        """Switch to indeterminate (pulsing) mode."""
        self._bar.configure(mode="indeterminate")
        self._bar.start(15)

    def reset(self) -> None:
        """Stop animation and reset to zero."""
        self._bar.stop()
        self._bar.configure(mode="determinate")
        self._bar["value"] = 0


# ── ScrollableFrame ──────────────────────────────────────────────────


class ScrollableFrame(ttk.Frame):
    """Canvas + Scrollbar + inner Frame for scrollable content.

    Add child widgets to ``self.interior`` rather than to the
    ``ScrollableFrame`` itself.
    """

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)

        self._canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.interior = ttk.Frame(self._canvas)
        self._window_id = self._canvas.create_window(
            (0, 0), window=self.interior, anchor="nw",
        )

        self.interior.bind("<Configure>", self._on_interior_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_interior_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window_id, width=event.width)
