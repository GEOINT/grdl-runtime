"""
Metrics Panel — Matplotlib-embedded execution metrics visualisation.

Renders per-step timing bar charts, memory usage, per-image timing
(batch mode), and accuracy metrics (when ground truth is provided)
inside a tkinter frame using ``FigureCanvasTkAgg``.

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
from tkinter import ttk

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from grdl_rt.ui._accuracy import AccuracyReport


class MetricsPanel(ttk.Frame):
    """Frame containing matplotlib charts for execution metrics.

    Embeds a 2x2 subplot grid that adapts content based on whether
    batch results and/or accuracy data are available.
    """

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)

        self._fig = Figure(figsize=(8, 5), dpi=96, tight_layout=True)
        self._fig.patch.set_facecolor("#2b2b2b")
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # Summary text label below the figure
        self._summary_var = tk.StringVar(value="No results yet.")
        self._summary_label = ttk.Label(
            self,
            textvariable=self._summary_var,
            font=("Consolas", 9),
            foreground="#aaaaaa",
        )
        self._summary_label.pack(fill="x", padx=4, pady=(2, 4))

    # ── public API ───────────────────────────────────────────────

    def update(  # type: ignore[override]
        self,
        results: list,
        accuracy: AccuracyReport | None = None,
    ) -> None:
        """Redraw all charts with new results.

        Parameters
        ----------
        results : list of WorkflowResult
            Execution results (may be empty).
        accuracy : AccuracyReport, optional
            Aggregate accuracy data (if ground truth was provided).
        """
        self._fig.clear()

        if not results:
            ax = self._fig.add_subplot(111)
            _style_axes(ax)
            ax.text(
                0.5,
                0.5,
                "No results to display.",
                ha="center",
                va="center",
                fontsize=12,
                color="#888888",
                transform=ax.transAxes,
            )
            self._canvas.draw()
            return

        has_accuracy = accuracy is not None
        n_rows = 2
        n_cols = 2

        # ── Top-left: per-step wall-clock time ───────────────────
        ax1 = self._fig.add_subplot(n_rows, n_cols, 1)
        self._plot_step_timing(ax1, results)

        # ── Top-right: per-step peak memory ──────────────────────
        ax2 = self._fig.add_subplot(n_rows, n_cols, 2)
        self._plot_step_memory(ax2, results)

        # ── Bottom-left: per-image timing (batch) ────────────────
        ax3 = self._fig.add_subplot(n_rows, n_cols, 3)
        self._plot_image_timing(ax3, results)

        # ── Bottom-right: accuracy or aggregate stats ────────────
        ax4 = self._fig.add_subplot(n_rows, n_cols, 4)
        if has_accuracy and accuracy is not None:
            self._plot_accuracy(ax4, accuracy)
        else:
            self._plot_aggregate_stats(ax4, results)

        self._update_summary(results, accuracy)
        self._canvas.draw()

    def clear(self) -> None:
        """Reset the panel to blank state."""
        self._fig.clear()
        self._summary_var.set("No results yet.")
        self._canvas.draw()

    # ── plot helpers ─────────────────────────────────────────────

    def _plot_step_timing(self, ax, results: list) -> None:
        """Horizontal bar chart of per-step wall-clock time."""
        _style_axes(ax)
        ax.set_title("Step Wall-Clock Time", fontsize=10, color="#dddddd")

        # Use the first result's step metrics as representative
        wr = results[0]
        metrics = wr.metrics
        if not metrics.step_metrics:
            ax.text(
                0.5,
                0.5,
                "No step metrics.",
                ha="center",
                va="center",
                color="#888888",
                transform=ax.transAxes,
            )
            return

        names = [sm.processor_name for sm in metrics.step_metrics]
        times = [sm.wall_time_s for sm in metrics.step_metrics]

        # Truncate long names
        names = [n[:20] + "\u2026" if len(n) > 20 else n for n in names]

        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, times, color="#4fc3f7", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8, color="#cccccc")
        ax.set_xlabel("Seconds", fontsize=8, color="#aaaaaa")
        ax.invert_yaxis()

        # Value labels
        for bar, t in zip(bars, times, strict=False):
            ax.text(
                bar.get_width() + max(times) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{t:.3f}s",
                va="center",
                fontsize=7,
                color="#cccccc",
            )

    def _plot_step_memory(self, ax, results: list) -> None:
        """Horizontal bar chart of per-step peak RSS delta."""
        _style_axes(ax)
        ax.set_title("Step Peak Memory", fontsize=10, color="#dddddd")

        wr = results[0]
        metrics = wr.metrics
        if not metrics.step_metrics:
            ax.text(
                0.5,
                0.5,
                "No step metrics.",
                ha="center",
                va="center",
                color="#888888",
                transform=ax.transAxes,
            )
            return

        names = [sm.processor_name for sm in metrics.step_metrics]
        mem_mb = [sm.peak_rss_bytes / (1024 * 1024) for sm in metrics.step_metrics]

        names = [n[:20] + "\u2026" if len(n) > 20 else n for n in names]
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, mem_mb, color="#81c784", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8, color="#cccccc")
        ax.set_xlabel("MB", fontsize=8, color="#aaaaaa")
        ax.invert_yaxis()

        for bar, m in zip(bars, mem_mb, strict=False):
            ax.text(
                bar.get_width() + max(mem_mb) * 0.02 if max(mem_mb) > 0 else 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{m:.1f}",
                va="center",
                fontsize=7,
                color="#cccccc",
            )

    def _plot_image_timing(self, ax, results: list) -> None:
        """Bar chart of per-image total execution time."""
        _style_axes(ax)
        ax.set_title("Per-Image Execution Time", fontsize=10, color="#dddddd")

        n = len(results)
        if n == 0:
            ax.text(
                0.5,
                0.5,
                "No results.",
                ha="center",
                va="center",
                color="#888888",
                transform=ax.transAxes,
            )
            return

        times = [wr.metrics.total_wall_time_s for wr in results]
        x = np.arange(n)
        labels = [f"Img {i + 1}" for i in range(n)]

        bars = ax.bar(x, times, color="#ffb74d", width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, color="#cccccc", rotation=45 if n > 6 else 0)
        ax.set_ylabel("Seconds", fontsize=8, color="#aaaaaa")

        for bar, t in zip(bars, times, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{t:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#cccccc",
            )

    def _plot_accuracy(self, ax, accuracy: AccuracyReport) -> None:
        """Bar chart of P/R/F1 + IoU histogram."""
        _style_axes(ax)
        ax.set_title("Accuracy Metrics", fontsize=10, color="#dddddd")

        metrics = ["Precision", "Recall", "F1"]
        values = [accuracy.precision, accuracy.recall, accuracy.f1]
        colours = ["#4fc3f7", "#81c784", "#ffb74d"]

        x = np.arange(len(metrics))
        bars = ax.bar(x, values, color=colours, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9, color="#cccccc")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontsize=8, color="#aaaaaa")

        for bar, v in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#cccccc",
            )

        # Annotate with counts
        ax.text(
            0.98,
            0.02,
            f"TP={accuracy.true_positives}  FP={accuracy.false_positives}  "
            f"FN={accuracy.false_negatives}  mIoU={accuracy.mean_iou:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#aaaaaa",
        )

    def _plot_aggregate_stats(self, ax, results: list) -> None:
        """Text summary of aggregate execution stats."""
        _style_axes(ax)
        ax.set_title("Run Summary", fontsize=10, color="#dddddd")
        ax.axis("off")

        n = len(results)
        total_wall = sum(wr.metrics.total_wall_time_s for wr in results)
        total_cpu = sum(wr.metrics.total_cpu_time_s for wr in results)
        peak_mem = max(wr.metrics.peak_rss_bytes for wr in results) if results else 0
        peak_mem_mb = peak_mem / (1024 * 1024)

        gpu_steps = sum(1 for wr in results for sm in wr.metrics.step_metrics if sm.gpu_used)
        total_steps = sum(len(wr.metrics.step_metrics) for wr in results)

        lines = [
            f"Images processed:  {n}",
            f"Total wall time:   {total_wall:.2f}s",
            f"Total CPU time:    {total_cpu:.2f}s",
            f"Peak memory:       {peak_mem_mb:.1f} MB",
            f"GPU steps:         {gpu_steps}/{total_steps}",
        ]

        text = "\n".join(lines)
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="#cccccc",
            family="monospace",
        )

    def _update_summary(
        self,
        results: list,
        accuracy: AccuracyReport | None,
    ) -> None:
        """Update the text summary below the figure."""
        n = len(results)
        total_wall = sum(wr.metrics.total_wall_time_s for wr in results)
        parts = [f"{n} image(s)", f"{total_wall:.2f}s total"]
        if accuracy is not None:
            parts.append(f"F1={accuracy.f1:.3f}")
        self._summary_var.set("  |  ".join(parts))


# ── Helpers ──────────────────────────────────────────────────────────


def _style_axes(ax) -> None:
    """Apply dark-theme styling to a matplotlib axes."""
    ax.set_facecolor("#2b2b2b")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#555555")
