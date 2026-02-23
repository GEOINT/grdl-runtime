"""
Config Dialog — Toplevel window for editing processor tunable parameters.

Renders appropriate widgets based on parameter type and constraints
(checkboxes for booleans, comboboxes for enumerations, sliders for
bounded numerics, text entries for everything else).  Multi-step
workflows use a tabbed notebook with one tab per processing step.

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
from tkinter import messagebox, ttk
from typing import Any

from grdl_rt.ui._importer import ParamInfo

# ── Type coercion ────────────────────────────────────────────────────


def _coerce_value(raw: str, param_type: str) -> Any:
    """Convert a string entry value to the declared parameter type.

    Raises ``ValueError`` on conversion failure.
    """
    raw = raw.strip()
    if param_type == "int":
        return int(raw)
    if param_type == "float":
        return float(raw)
    if param_type == "bool":
        return raw.lower() in ("true", "1", "yes", "on")
    return raw


# ── Single-step parameter form ───────────────────────────────────────


class _ParamForm(ttk.Frame):
    """Widget form for a single step's tunable parameters.

    Parameters
    ----------
    parent : tk widget
        Parent container.
    params : dict
        Mapping of parameter name → ``ParamInfo``.
    current_values : dict
        Current parameter values (overrides defaults).
    """

    def __init__(
        self,
        parent: tk.Widget,
        params: dict[str, ParamInfo],
        current_values: dict[str, Any],
    ) -> None:
        super().__init__(parent, padding=8)
        self._params = params
        self._widgets: dict[str, Any] = {}
        self._vars: dict[str, tk.Variable] = {}
        self._defaults: dict[str, Any] = {}

        if not params:
            lbl = ttk.Label(self, text="No tunable parameters.", foreground="grey")
            lbl.pack(anchor="w", pady=8)
            return

        for row, (name, info) in enumerate(params.items()):
            current = current_values.get(name, info.default)
            self._defaults[name] = info.default
            self._build_row(row, name, info, current)

    # ── row builders ─────────────────────────────────────────────

    def _build_row(
        self,
        row: int,
        name: str,
        info: ParamInfo,
        current: Any,
    ) -> None:
        # Label
        label_text = name
        if info.required:
            label_text += " *"
        lbl = ttk.Label(self, text=label_text, width=22, anchor="w")
        lbl.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)

        # Description tooltip (shown in a small label if present)
        if info.description:
            desc = ttk.Label(
                self,
                text=info.description,
                foreground="grey",
                wraplength=300,
            )
            desc.grid(row=row, column=2, sticky="w", padx=(8, 0), pady=2)

        # Widget selection
        if info.param_type == "bool":
            self._build_bool(row, name, current)
        elif info.choices:
            self._build_choices(row, name, info, current)
        elif (
            info.param_type in ("int", "float")
            and info.min_value is not None
            and info.max_value is not None
        ):
            self._build_slider(row, name, info, current)
        else:
            self._build_entry(row, name, info, current)

    def _build_bool(self, row: int, name: str, current: Any) -> None:
        var = tk.BooleanVar(value=bool(current) if current is not None else False)
        cb = ttk.Checkbutton(self, variable=var)
        cb.grid(row=row, column=1, sticky="w", pady=2)
        self._vars[name] = var
        self._widgets[name] = cb

    def _build_choices(
        self,
        row: int,
        name: str,
        info: ParamInfo,
        current: Any,
    ) -> None:
        var = tk.StringVar(value=str(current) if current is not None else "")
        combo = ttk.Combobox(
            self,
            textvariable=var,
            values=[str(c) for c in info.choices],
            state="readonly",
            width=20,
        )
        combo.grid(row=row, column=1, sticky="w", pady=2)
        self._vars[name] = var
        self._widgets[name] = combo

    def _build_slider(
        self,
        row: int,
        name: str,
        info: ParamInfo,
        current: Any,
    ) -> None:
        frame = ttk.Frame(self)
        frame.grid(row=row, column=1, sticky="ew", pady=2)

        val = current if current is not None else info.default
        if val is None:
            val = info.min_value

        var = tk.DoubleVar(value=float(val))

        scale = ttk.Scale(
            frame,
            from_=info.min_value,
            to=info.max_value,
            orient="horizontal",
            variable=var,
            length=160,
        )
        scale.pack(side="left", padx=(0, 4))

        # Linked entry showing numeric value
        entry_var = tk.StringVar(value=str(val))
        entry = ttk.Entry(frame, textvariable=entry_var, width=10)
        entry.pack(side="left")

        def _on_scale_change(*_args):
            v = var.get()
            if info.param_type == "int":
                v = int(round(v))
                var.set(float(v))
            entry_var.set(str(v))

        def _on_entry_change(*_args):
            try:
                v = float(entry_var.get())
                if info.min_value is not None:
                    v = max(v, info.min_value)
                if info.max_value is not None:
                    v = min(v, info.max_value)
                var.set(v)
            except ValueError:
                pass

        var.trace_add("write", _on_scale_change)
        entry.bind("<Return>", _on_entry_change)
        entry.bind("<FocusOut>", _on_entry_change)

        self._vars[name] = var
        self._widgets[name] = (scale, entry)

    def _build_entry(
        self,
        row: int,
        name: str,
        info: ParamInfo,
        current: Any,
    ) -> None:
        val = str(current) if current is not None else ""
        var = tk.StringVar(value=val)
        entry = ttk.Entry(self, textvariable=var, width=24)
        entry.grid(row=row, column=1, sticky="w", pady=2)
        self._vars[name] = var
        self._widgets[name] = entry

    # ── value extraction ─────────────────────────────────────────

    def get_values(self) -> dict[str, Any]:
        """Return a dict of parameter name → coerced value."""
        result: dict[str, Any] = {}
        for name, info in self._params.items():
            var = self._vars.get(name)
            if var is None:
                continue

            if info.param_type == "bool":
                result[name] = var.get()
            elif isinstance(var, tk.DoubleVar):
                v = var.get()
                result[name] = int(round(v)) if info.param_type == "int" else v
            else:
                raw = var.get()
                if raw == "" and not info.required:
                    if info.default is not None:
                        result[name] = info.default
                    continue
                result[name] = _coerce_value(raw, info.param_type)
        return result

    def reset_defaults(self) -> None:
        """Reset all widgets to their default values."""
        for name, info in self._params.items():
            var = self._vars.get(name)
            if var is None:
                continue
            default = self._defaults.get(name)
            if info.param_type == "bool":
                var.set(bool(default) if default is not None else False)
            elif isinstance(var, tk.DoubleVar):
                var.set(float(default) if default is not None else 0.0)
            else:
                var.set(str(default) if default is not None else "")


# ── ConfigDialog ─────────────────────────────────────────────────────


class ConfigDialog(tk.Toplevel):
    """Modal dialog for editing tunable parameters.

    For a single processor, shows the parameter form directly.  For
    multi-step workflows, uses a tabbed notebook with one tab per step.

    After the dialog is closed, ``self.result`` contains the collected
    parameter values (``dict`` of step_name → param_dict) or ``None``
    if the user cancelled.

    Parameters
    ----------
    parent : tk widget
        Owner window.
    steps : list of (step_name, params_dict, current_values_dict)
        One entry per workflow step.
    title : str, optional
        Window title.
    """

    def __init__(
        self,
        parent: tk.Widget,
        steps: list[tuple],
        title: str = "Configure Parameters",
    ) -> None:
        super().__init__(parent)
        self.title(title)
        self.resizable(True, True)
        self.minsize(480, 200)
        self.result: dict[str, dict[str, Any]] | None = None

        self._forms: dict[str, _ParamForm] = {}

        # Build content
        if len(steps) == 1:
            step_name, params, current = steps[0]
            form = _ParamForm(self, params, current)
            form.pack(fill="both", expand=True, padx=8, pady=4)
            self._forms[step_name] = form
        else:
            notebook = ttk.Notebook(self)
            notebook.pack(fill="both", expand=True, padx=8, pady=4)
            for step_name, params, current in steps:
                form = _ParamForm(notebook, params, current)
                notebook.add(form, text=step_name)
                self._forms[step_name] = form

        # Button bar
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=(4, 8))

        ttk.Button(
            btn_frame,
            text="Reset Defaults",
            command=self._reset,
        ).pack(side="left")

        ttk.Button(
            btn_frame,
            text="Cancel",
            command=self._cancel,
        ).pack(side="right", padx=(4, 0))

        ttk.Button(
            btn_frame,
            text="Apply",
            command=self._apply,
        ).pack(side="right")

        # Modal behaviour
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _apply(self) -> None:
        try:
            self.result = {}
            for step_name, form in self._forms.items():
                self.result[step_name] = form.get_values()
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Validation Error", str(e), parent=self)

    def _cancel(self) -> None:
        self.result = None
        self.destroy()

    def _reset(self) -> None:
        for form in self._forms.values():
            form.reset_defaults()
