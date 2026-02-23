"""
App — Main application window for the grdl-rt workflow runner GUI.

Assembles all UI components (file pickers, hardware options, parameter
config, metrics panel, log console, output preview) into a single Tk
window and wires them to the background :class:`WorkflowRunner`.

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

import json
import os
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
import contextlib

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from grdl_rt.ui._accuracy import AccuracyReport
from grdl_rt.ui._config_dialog import ConfigDialog
from grdl_rt.ui._importer import (
    ParamInfo,
    classify_py_file,
    discover_processors_in_module,
    discover_workflow_in_module,
    extract_tunable_params,
)
from grdl_rt.ui._metrics_panel import MetricsPanel
from grdl_rt.ui._runner import RunRequest, RunResult, WorkflowRunner
from grdl_rt.ui._widgets import (
    GEOJSON_FILETYPES,
    IMAGE_EXTENSIONS,
    IMAGE_FILETYPES,
    WORKFLOW_FILETYPES,
    FilePickerRow,
    LabeledEntry,
    LogConsole,
    ProgressBar,
)

# ── History persistence ──────────────────────────────────────────────

_HISTORY_DIR = Path.home() / ".grdl"
_HISTORY_FILE = _HISTORY_DIR / "ui_history.json"
_MAX_HISTORY = 20


def _load_history() -> list[dict]:
    """Load run history from disk."""
    try:
        if _HISTORY_FILE.exists():
            with open(_HISTORY_FILE, encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _save_history(history: list[dict]) -> None:
    """Persist run history to disk."""
    try:
        _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(_HISTORY_FILE, "w", encoding="utf-8") as fh:
            json.dump(history[-_MAX_HISTORY:], fh, indent=2, default=str)
    except Exception:
        pass


# ── File-change watcher ──────────────────────────────────────────────


class _FileWatcher:
    """Polls a file's mtime and fires a callback when it changes."""

    def __init__(self, root: tk.Tk, callback) -> None:
        self._root = root
        self._callback = callback
        self._path: Path | None = None
        self._mtime: float | None = None
        self._after_id: str | None = None

    def watch(self, path: Path) -> None:
        self.stop()
        self._path = path
        try:
            self._mtime = os.path.getmtime(str(path))
        except OSError:
            self._mtime = None
        self._poll()

    def stop(self) -> None:
        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
            self._after_id = None
        self._path = None

    def _poll(self) -> None:
        if self._path is None:
            return
        try:
            mtime = os.path.getmtime(str(self._path))
            if self._mtime is not None and mtime != self._mtime:
                self._mtime = mtime
                self._callback(self._path)
            else:
                self._mtime = mtime
        except OSError:
            pass
        self._after_id = self._root.after(2000, self._poll)


# ── Main Application ─────────────────────────────────────────────────


class App(tk.Tk):
    """Main grdl-rt runner window.

    Parameters
    ----------
    workflow_path : str, optional
        Pre-fill the workflow file picker.
    input_path : str, optional
        Pre-fill the input image picker.
    """

    def __init__(
        self,
        workflow_path: str | None = None,
        input_path: str | None = None,
    ) -> None:
        super().__init__()

        self.title("grdl-rt Runner")
        self.minsize(1050, 720)
        self.configure(bg="#1e1e1e")

        # Apply ttk theme
        style = ttk.Style(self)
        with contextlib.suppress(tk.TclError):
            style.theme_use("clam")

        # Dark-ish overrides for clam
        style.configure(
            ".",
            background="#2b2b2b",
            foreground="#d4d4d4",
            fieldbackground="#3c3c3c",
            bordercolor="#555555",
        )
        style.configure("TLabel", background="#2b2b2b", foreground="#d4d4d4")
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TButton", background="#3c3c3c", foreground="#d4d4d4")
        style.configure("TNotebook", background="#2b2b2b")
        style.configure("TNotebook.Tab", background="#3c3c3c", foreground="#d4d4d4", padding=[8, 2])
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#4a4a4a")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure("Accent.TButton", background="#0078d4", foreground="#ffffff")

        # State
        self._current_params: dict[str, dict[str, Any]] = {}
        self._step_param_info: dict[str, dict[str, ParamInfo]] = {}
        self._last_results: list = []
        self._last_accuracy: AccuracyReport | None = None
        self._selected_component: str | None = None
        self._history: list[dict] = _load_history()

        # Runner
        self._runner = WorkflowRunner(
            on_progress=self._on_progress,
            on_log=self._on_log,
            on_complete=self._on_complete,
        )

        # File watcher
        self._watcher = _FileWatcher(self, self._on_file_changed)

        # Build UI
        self._build_menu()
        self._build_layout()
        self._build_status_bar()

        # Keyboard shortcuts
        self.bind_all("<Control-o>", lambda _: self._on_browse_workflow())
        self.bind_all("<Control-i>", lambda _: self._on_browse_input())
        self.bind_all("<Control-r>", lambda _: self._on_run())
        self.bind_all("<Escape>", lambda _: self._on_cancel())

        # Pre-fill if paths provided
        if workflow_path:
            self._wf_picker.set(workflow_path)
            self._on_workflow_selected(workflow_path)
        if input_path:
            self._input_picker.set(input_path)

    # ── Menu bar ─────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menubar = tk.Menu(
            self, bg="#2b2b2b", fg="#d4d4d4", activebackground="#4a4a4a", activeforeground="#ffffff"
        )

        # File menu
        file_menu = tk.Menu(
            menubar, tearoff=0, bg="#2b2b2b", fg="#d4d4d4", activebackground="#4a4a4a"
        )
        file_menu.add_command(label="Open Workflow\u2026  Ctrl+O", command=self._on_browse_workflow)
        file_menu.add_command(label="Open Image(s)\u2026  Ctrl+I", command=self._on_browse_input)
        file_menu.add_command(label="Open Folder\u2026", command=self._on_browse_folder)
        file_menu.add_command(label="Set Output Dir\u2026", command=self._on_browse_output)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results\u2026", command=self._on_export)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        # Run menu
        run_menu = tk.Menu(
            menubar, tearoff=0, bg="#2b2b2b", fg="#d4d4d4", activebackground="#4a4a4a"
        )
        run_menu.add_command(label="Execute  Ctrl+R", command=self._on_run)
        run_menu.add_command(label="Cancel  Esc", command=self._on_cancel)
        run_menu.add_separator()
        run_menu.add_command(label="Re-run with Same Params", command=self._on_rerun)
        menubar.add_cascade(label="Run", menu=run_menu)

        # View menu
        view_menu = tk.Menu(
            menubar, tearoff=0, bg="#2b2b2b", fg="#d4d4d4", activebackground="#4a4a4a"
        )
        view_menu.add_command(label="Clear Results", command=self._on_clear)
        menubar.add_cascade(label="View", menu=view_menu)

        # Help menu
        help_menu = tk.Menu(
            menubar, tearoff=0, bg="#2b2b2b", fg="#d4d4d4", activebackground="#4a4a4a"
        )
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.configure(menu=menubar)

    # ── Main layout ──────────────────────────────────────────────

    def _build_layout(self) -> None:
        # PanedWindow for resizable left/right split
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Left panel: input configuration ──────────────────────
        left = ttk.Frame(paned, padding=4)
        paned.add(left, weight=1)

        # Workflow file picker
        self._wf_picker = FilePickerRow(
            left,
            "Workflow / Component:",
            mode="file",
            filetypes=WORKFLOW_FILETYPES,
        )
        self._wf_picker.pack(fill="x", pady=(0, 4))
        self._wf_picker.var.trace_add(
            "write",
            lambda *_: self._on_workflow_selected(self._wf_picker.get()),
        )

        # Input images (multi-file)
        self._input_picker = FilePickerRow(
            left,
            "Input Image(s):",
            mode="files",
            filetypes=IMAGE_FILETYPES,
        )
        self._input_picker.pack(fill="x", pady=(0, 4))

        # Input folder
        self._folder_picker = FilePickerRow(
            left,
            "Input Folder:",
            mode="directory",
        )
        self._folder_picker.pack(fill="x", pady=(0, 4))

        # Output directory
        self._output_picker = FilePickerRow(
            left,
            "Output Directory:",
            mode="directory",
        )
        self._output_picker.pack(fill="x", pady=(0, 4))

        # Ground truth GeoJSON
        self._gt_picker = FilePickerRow(
            left,
            "Ground Truth (GeoJSON):",
            mode="file",
            filetypes=GEOJSON_FILETYPES,
        )
        self._gt_picker.pack(fill="x", pady=(0, 8))

        # Hardware / concurrency section
        hw_frame = ttk.LabelFrame(left, text="Hardware / Concurrency", padding=6)
        hw_frame.pack(fill="x", pady=(0, 8))

        self._gpu_var = tk.BooleanVar(value=False)
        gpu_cb = ttk.Checkbutton(hw_frame, text="Prefer GPU", variable=self._gpu_var)
        gpu_cb.grid(row=0, column=0, sticky="w", pady=2)

        self._workers_entry = LabeledEntry(hw_frame, "Max Workers:", default="1", width=6)
        self._workers_entry.grid(row=1, column=0, sticky="w", pady=2)

        self._memory_entry = LabeledEntry(hw_frame, "Memory Limit (MB):", default="", width=8)
        self._memory_entry.grid(row=2, column=0, sticky="w", pady=2)

        # Component info label
        self._component_label = ttk.Label(left, text="", foreground="#888888")
        self._component_label.pack(fill="x", pady=(0, 4))

        # Configure Params button
        self._config_btn = ttk.Button(
            left,
            text="Configure Parameters\u2026",
            command=self._on_configure_params,
        )
        self._config_btn.pack(fill="x", pady=(0, 4))

        # Run / Cancel buttons
        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=(0, 4))

        self._run_btn = ttk.Button(
            btn_frame,
            text="  Run  ",
            command=self._on_run,
            style="Accent.TButton",
        )
        self._run_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self._cancel_btn = ttk.Button(
            btn_frame,
            text="Cancel",
            command=self._on_cancel,
            state="disabled",
        )
        self._cancel_btn.pack(side="left")

        # Progress bar
        self._progress = ProgressBar(left)
        self._progress.pack(fill="x", pady=(0, 4))

        # ── Right panel: results notebook ────────────────────────
        right = ttk.Frame(paned, padding=4)
        paned.add(right, weight=2)

        self._notebook = ttk.Notebook(right)
        self._notebook.pack(fill="both", expand=True)

        # Output preview tab
        preview_frame = ttk.Frame(self._notebook)
        self._notebook.add(preview_frame, text="Output Preview")
        self._build_preview_tab(preview_frame)

        # Metrics tab
        self._metrics_panel = MetricsPanel(self._notebook)
        self._notebook.add(self._metrics_panel, text="Metrics")

        # Log tab
        self._log = LogConsole(self._notebook)
        self._notebook.add(self._log, text="Log")

    def _build_preview_tab(self, parent: ttk.Frame) -> None:
        """Build the output image preview tab with matplotlib canvas."""
        # Image selector (for batch mode)
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=4, pady=2)

        ttk.Label(top, text="Image:").pack(side="left")
        self._preview_selector = ttk.Combobox(
            top,
            state="readonly",
            width=30,
        )
        self._preview_selector.pack(side="left", padx=4)
        self._preview_selector.bind("<<ComboboxSelected>>", self._on_preview_select)

        # Matplotlib canvas
        self._preview_fig = Figure(figsize=(6, 4), dpi=96, tight_layout=True)
        self._preview_fig.patch.set_facecolor("#2b2b2b")
        self._preview_canvas = FigureCanvasTkAgg(self._preview_fig, master=parent)
        self._preview_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Navigation toolbar
        toolbar = NavigationToolbar2Tk(self._preview_canvas, parent)
        toolbar.update()

    def _build_status_bar(self) -> None:
        """Status bar at the bottom of the window."""
        self._status_var = tk.StringVar(value="Ready")
        status = ttk.Label(
            self,
            textvariable=self._status_var,
            relief="sunken",
            anchor="w",
            padding=(4, 2),
        )
        status.pack(side="bottom", fill="x")

    # ── Workflow loading ─────────────────────────────────────────

    def _on_workflow_selected(self, path_str: str) -> None:
        """Handle a new workflow/component file being selected."""
        if not path_str:
            return
        path = Path(path_str)
        if not path.exists():
            return

        self._watcher.watch(path)
        self._current_params.clear()
        self._step_param_info.clear()
        self._selected_component = None

        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            self._load_yaml_workflow(path)
        elif suffix == ".py":
            self._load_py_file(path)

    def _load_yaml_workflow(self, path: Path) -> None:
        try:
            from grdl_rt.api import load_workflow

            wf_def = load_workflow(path)
            self._component_label.configure(
                text=f"Workflow: {wf_def.name} v{wf_def.version} " f"({len(wf_def.steps)} steps)",
            )
            self._log.append(f"Loaded workflow: {wf_def.name}", "info")

            # Validate
            errors = wf_def.validate()
            for err in errors:
                self._log.append(f"Validation: [{err.code}] {err.message}", "warn")

            # Extract params for each step
            from grdl_rt.execution.discovery import resolve_processor_class

            for step in wf_def.steps:
                proc_name = getattr(step, "processor_name", None)
                if proc_name is None:
                    continue
                try:
                    cls = resolve_processor_class(proc_name)
                    params = extract_tunable_params(cls)
                    step_key = proc_name
                    self._step_param_info[step_key] = params
                    self._current_params[step_key] = dict(step.params)  # type: ignore[union-attr]
                except Exception:
                    pass

        except Exception as exc:
            self._log.append(f"Failed to load workflow: {exc}", "error")

    def _load_py_file(self, path: Path) -> None:
        try:
            kind = classify_py_file(path)
            if kind == "workflow":
                wf = discover_workflow_in_module(path)
                from grdl_rt.execution.builder import Workflow
                from grdl_rt.execution.workflow import WorkflowDefinition

                if isinstance(wf, Workflow):
                    self._component_label.configure(
                        text=f"Python workflow: {wf._name}",
                    )
                    self._log.append(f"Loaded Python workflow: {wf._name}", "info")
                elif isinstance(wf, WorkflowDefinition):
                    self._component_label.configure(
                        text=f"Workflow: {wf.name} v{wf.version}",
                    )
                    self._log.append(f"Loaded workflow def: {wf.name}", "info")
            else:
                # Bare component
                processors = discover_processors_in_module(path)
                if not processors:
                    self._component_label.configure(
                        text="No processors found in file.",
                    )
                    self._log.append(
                        f"No processors found in {path.name}",
                        "warn",
                    )
                    return

                if len(processors) == 1:
                    name, cls = processors[0]
                    self._selected_component = name
                else:
                    # Selection dialog
                    names = [n for n, _ in processors]
                    name = self._show_processor_selection(names)  # type: ignore[assignment]
                    if name is None:
                        return
                    cls = dict(processors)[name]
                    self._selected_component = name

                params = extract_tunable_params(cls)
                self._step_param_info[self._selected_component] = params
                self._current_params[self._selected_component] = {
                    p.name: p.default for p in params.values() if p.default is not None
                }
                self._component_label.configure(
                    text=f"Component: {self._selected_component}",
                )
                self._log.append(
                    f"Loaded component: {self._selected_component} "
                    f"({len(params)} tunable params)",
                    "info",
                )
        except Exception as exc:
            self._log.append(f"Failed to load file: {exc}", "error")

    def _show_processor_selection(self, names: list[str]) -> str | None:
        """Show a simple dialog to select one processor from a list."""
        dialog = tk.Toplevel(self)
        dialog.title("Select Processor")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(dialog, text="Multiple processors found. Select one:").pack(
            padx=12,
            pady=(12, 4),
        )

        tk.StringVar(value=names[0])
        listbox = tk.Listbox(
            dialog,
            listvariable=tk.StringVar(value=names),  # type: ignore[arg-type]
            height=min(len(names), 10),
            width=40,
            bg="#3c3c3c",
            fg="#d4d4d4",
            selectbackground="#0078d4",
        )
        listbox.pack(padx=12, pady=4)
        listbox.selection_set(0)

        result: list[str | None] = [None]

        def _ok():
            sel = listbox.curselection()
            if sel:
                result[0] = names[sel[0]]
            dialog.destroy()

        def _cancel():
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(padx=12, pady=(4, 12))
        ttk.Button(btn_frame, text="Select", command=_ok).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=_cancel).pack(side="left", padx=4)

        dialog.wait_window()
        return result[0]

    # ── File change notification ─────────────────────────────────

    def _on_file_changed(self, path: Path) -> None:
        answer = messagebox.askyesno(
            "File Changed",
            f"{path.name} has been modified on disk.\nReload?",
            parent=self,
        )
        if answer:
            self._wf_picker.set(str(path))
            self._on_workflow_selected(str(path))

    # ── Browse actions ───────────────────────────────────────────

    def _on_browse_workflow(self) -> None:
        self._wf_picker._browse()

    def _on_browse_input(self) -> None:
        self._input_picker._browse()

    def _on_browse_folder(self) -> None:
        self._folder_picker._browse()

    def _on_browse_output(self) -> None:
        self._output_picker._browse()

    # ── Parameter configuration ──────────────────────────────────

    def _on_configure_params(self) -> None:
        if not self._step_param_info:
            messagebox.showinfo(
                "No Parameters",
                "Load a workflow or component first.",
                parent=self,
            )
            return

        steps = []
        for step_name, params in self._step_param_info.items():
            current = self._current_params.get(step_name, {})
            steps.append((step_name, params, current))

        dialog = ConfigDialog(self, steps)  # type: ignore[arg-type]
        self.wait_window(dialog)

        if dialog.result is not None:
            self._current_params.update(dialog.result)
            self._log.append("Parameters updated.", "info")

    # ── Run / Cancel ─────────────────────────────────────────────

    def _collect_input_paths(self) -> list[Path]:
        """Gather input paths from both file and folder pickers."""
        paths: list[Path] = []

        # From multi-file picker
        for p in self._input_picker.get_paths():
            pp = Path(p)
            if pp.exists():
                paths.append(pp)

        # From folder picker
        folder_str = self._folder_picker.get()
        if folder_str:
            folder = Path(folder_str)
            if folder.is_dir():
                for child in sorted(folder.iterdir()):
                    if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                        paths.append(child)

        return paths

    def _on_run(self) -> None:
        wf_path = self._wf_picker.get()
        if not wf_path:
            messagebox.showwarning(
                "Missing Input", "Select a workflow or component file.", parent=self
            )
            return

        input_paths = self._collect_input_paths()
        if not input_paths:
            messagebox.showwarning("Missing Input", "Select at least one input image.", parent=self)
            return

        # Parse worker count
        try:
            max_workers = int(self._workers_entry.get() or "1")
        except ValueError:
            max_workers = 1

        wf_source = Path(wf_path)
        suffix = wf_source.suffix.lower()
        is_component = suffix == ".py" and classify_py_file(wf_source) == "component"

        output_dir = None
        out_str = self._output_picker.get()
        if out_str:
            output_dir = Path(out_str)
            output_dir.mkdir(parents=True, exist_ok=True)

        gt_path = None
        gt_str = self._gt_picker.get()
        if gt_str:
            gt_path = Path(gt_str)

        request = RunRequest(
            input_paths=input_paths,
            workflow_source=wf_source,
            is_component=is_component,
            component_class=self._selected_component,
            params=dict(self._current_params),
            prefer_gpu=self._gpu_var.get(),
            max_workers=max_workers,
            output_dir=output_dir,
            ground_truth_path=gt_path,
        )

        self._log.clear()
        self._log.append(
            f"Starting run: {len(input_paths)} image(s), " f"workflow={wf_source.name}",
            "info",
        )
        self._progress.reset()
        self._run_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")
        self._status_var.set("Running\u2026")

        self._runner.start(request, self)

    def _on_cancel(self) -> None:
        if self._runner.is_running:
            self._runner.cancel()
            self._status_var.set("Cancelling\u2026")

    def _on_rerun(self) -> None:
        """Re-execute the last run with same parameters."""
        self._on_run()

    # ── Runner callbacks ─────────────────────────────────────────

    def _on_progress(self, fraction: float, message: str) -> None:
        self._progress.set_fraction(fraction)
        self._status_var.set(message)

    def _on_log(self, message: str, level: str) -> None:
        self._log.append(message, level)

    def _on_complete(self, result: RunResult) -> None:
        self._run_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")
        self._progress.set_fraction(1.0)

        if result.error:
            self._status_var.set(f"Failed: {result.error}")
            self._log.append(f"Run failed: {result.error}", "error")
        else:
            n = len(result.workflow_results)
            self._status_var.set(f"Complete \u2014 {n} image(s) in {result.elapsed_seconds:.2f}s")
            self._log.append(
                f"Run complete: {n} results in {result.elapsed_seconds:.2f}s",
                "success",
            )

        self._last_results = result.workflow_results
        self._last_accuracy = result.accuracy

        # Update metrics panel
        self._metrics_panel.update(result.workflow_results, result.accuracy)

        # Update output preview
        self._update_preview_selector(result)
        if result.workflow_results:
            self._show_output_preview(0)

        # Switch to preview tab
        if result.workflow_results:
            self._notebook.select(0)

        # Save to history
        self._save_run_to_history(result)

    # ── Output preview ───────────────────────────────────────────

    def _update_preview_selector(self, result: RunResult) -> None:
        """Populate the image selector combo for batch results."""
        n = len(result.workflow_results)
        if n == 0:
            self._preview_selector.configure(values=[])
            return
        labels = []
        for i, p in enumerate(result.input_paths[:n]):
            labels.append(f"{i + 1}. {p.name}")
        # Pad if more results than input paths recorded
        while len(labels) < n:
            labels.append(f"{len(labels) + 1}. (unknown)")
        self._preview_selector.configure(values=labels)
        self._preview_selector.current(0)

    def _on_preview_select(self, _event=None) -> None:
        idx = self._preview_selector.current()
        if 0 <= idx < len(self._last_results):
            self._show_output_preview(idx)

    def _show_output_preview(self, idx: int) -> None:
        """Render a result array in the preview tab."""
        if idx >= len(self._last_results):
            return

        wr = self._last_results[idx]
        data = wr.result

        self._preview_fig.clear()
        ax = self._preview_fig.add_subplot(111)
        ax.set_facecolor("#2b2b2b")

        if isinstance(data, np.ndarray):
            img = _prepare_display_array(data)
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", aspect="equal")
            else:
                ax.imshow(img, aspect="equal")
            ax.set_title(
                f"shape={data.shape}  dtype={data.dtype}",
                fontsize=9,
                color="#cccccc",
            )
        else:
            ax.text(
                0.5,
                0.5,
                f"Result type: {type(data).__name__}\n(not displayable as image)",
                ha="center",
                va="center",
                fontsize=10,
                color="#888888",
                transform=ax.transAxes,
            )

        ax.axis("off")
        self._preview_canvas.draw()

    # ── History ──────────────────────────────────────────────────

    def _save_run_to_history(self, result: RunResult) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "workflow": self._wf_picker.get(),
            "n_images": len(result.input_paths),
            "n_results": len(result.workflow_results),
            "elapsed_s": result.elapsed_seconds,
            "error": result.error,
        }
        if result.accuracy:
            entry["accuracy"] = {
                "precision": result.accuracy.precision,
                "recall": result.accuracy.recall,
                "f1": result.accuracy.f1,
            }
        self._history.append(entry)
        _save_history(self._history)

    # ── Export ────────────────────────────────────────────────────

    def _on_export(self) -> None:
        if not self._last_results:
            messagebox.showinfo("Nothing to Export", "Run a workflow first.", parent=self)
            return

        out_dir = filedialog.askdirectory(title="Export Results To", parent=self)
        if not out_dir:
            return
        out_path = Path(out_dir)

        # Save result arrays
        for i, wr in enumerate(self._last_results):
            if isinstance(wr.result, np.ndarray):
                np.save(str(out_path / f"result_{i:03d}.npy"), wr.result)

        # Save metrics
        metrics_list = [wr.metrics.to_dict() for wr in self._last_results]
        with open(out_path / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(metrics_list, fh, indent=2, default=str)

        # Save accuracy
        if self._last_accuracy:
            with open(out_path / "accuracy.json", "w", encoding="utf-8") as fh:
                json.dump(self._last_accuracy.to_dict(), fh, indent=2)

        # Save preview image
        self._preview_fig.savefig(
            str(out_path / "preview.png"),
            dpi=150,
            facecolor="#2b2b2b",
            bbox_inches="tight",
        )

        self._log.append(f"Results exported to {out_path}", "success")
        messagebox.showinfo("Export Complete", f"Saved to {out_path}", parent=self)

    # ── Clear ────────────────────────────────────────────────────

    def _on_clear(self) -> None:
        self._last_results.clear()
        self._last_accuracy = None
        self._metrics_panel.clear()
        self._preview_fig.clear()
        self._preview_canvas.draw()
        self._preview_selector.configure(values=[])
        self._log.clear()
        self._progress.reset()
        self._status_var.set("Ready")

    # ── Help dialogs ─────────────────────────────────────────────

    def _show_about(self) -> None:
        try:
            from grdl_rt import __version__

            ver = __version__
        except ImportError:
            ver = "unknown"
        messagebox.showinfo(
            "About grdl-rt Runner",
            f"grdl-rt Runner GUI\nVersion: {ver}\n\n"
            "Lightweight workflow execution tool for GRDL component developers.\n"
            "Uses tkinter + matplotlib (zero additional dependencies).",
            parent=self,
        )

    def _show_shortcuts(self) -> None:
        messagebox.showinfo(
            "Keyboard Shortcuts",
            "Ctrl+O   Open workflow/component\n"
            "Ctrl+I   Open input image(s)\n"
            "Ctrl+R   Run workflow\n"
            "Escape   Cancel running workflow",
            parent=self,
        )


# ── Display helpers ──────────────────────────────────────────────────


def _prepare_display_array(arr: np.ndarray) -> np.ndarray:
    """Prepare a numpy array for matplotlib ``imshow``.

    Handles complex arrays (magnitude), multi-band (first 3 or first
    band), and normalisation to [0, 1] float.
    """
    # Complex → magnitude
    if np.iscomplexobj(arr):
        arr = np.abs(arr)

    # Squeeze singleton dims
    arr = np.squeeze(arr)

    # Multi-band: pick first 3 bands for RGB or first band for grayscale
    if arr.ndim == 3:
        if arr.shape[0] <= 4:
            # Band-first → (H, W, C)
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] >= 3:
            arr = arr[..., :3]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
    elif arr.ndim > 3:
        # Take first 2D slice
        arr = arr.reshape(-1, arr.shape[-1])

    # Normalise to [0, 1]
    arr = arr.astype(np.float64)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    arr = (arr - vmin) / (vmax - vmin) if vmax - vmin > 0 else np.zeros_like(arr)

    return arr
