# -*- coding: utf-8 -*-
"""
CLI — Command-line interface for grdl-runtime.

Provides the ``grdl-rt`` console entry point with subcommands for
headless workflow execution, validation, and processor discovery.

Author
------
Steven Siebert

License
-------
MIT License
Copyright (c) 2024 geoint.org
See LICENSE file for full text.

Created
-------
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
import argparse
import json as json_mod
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Third-party
import numpy as np

from grdl_rt.execution.context import configure_logging, get_logger

logger = get_logger(__name__)

# Exit codes
EXIT_SUCCESS = 0
EXIT_VALIDATION_FAILURE = 1
EXIT_EXECUTION_FAILURE = 2
EXIT_INPUT_ERROR = 3


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog='grdl-rt',
        description='GRDL Runtime — headless workflow execution engine',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {_get_version()}',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (-v for INFO, -vv for DEBUG).',
    )
    parser.add_argument(
        '--json-log',
        action='store_true',
        default=False,
        help='Output log lines as JSON (for machine consumption).',
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ── run subcommand ───────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        'run',
        help='Execute a workflow on an input image.',
    )
    run_parser.add_argument(
        'workflow',
        type=str,
        help='Path to a YAML workflow definition file.',
    )
    run_parser.add_argument(
        'input',
        type=str,
        help='Path to the input image file (.tif, .nitf, .npy, etc.).',
    )
    run_parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path.  If omitted, prints summary to stdout.',
    )
    run_parser.add_argument(
        '--output-format',
        type=str,
        default=None,
        help='Output format override (e.g., geotiff, numpy, png).',
    )
    run_parser.add_argument(
        '--auto-tap-out',
        action='store_true',
        default=False,
        help='Write all intermediates to a timestamped directory.',
    )
    run_parser.add_argument(
        '--tap-out-format',
        type=str,
        default='npy',
        help='Format for auto tap-out intermediate files (default: npy).',
    )
    run_parser.add_argument(
        '--tap-out-dir',
        type=str,
        default=None,
        help='Override the auto tap-out output directory.',
    )
    run_parser.add_argument(
        '--prefer-gpu',
        action='store_true',
        default=False,
        help='Attempt GPU acceleration for compatible steps.',
    )

    # ── validate subcommand ──────────────────────────────────────────
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate a workflow YAML without executing.',
    )
    validate_parser.add_argument(
        'workflow',
        type=str,
        help='Path to a YAML workflow definition file.',
    )

    # ── resume subcommand ─────────────────────────────────────────────
    resume_parser = subparsers.add_parser(
        'resume',
        help='Resume a workflow from a checkpoint file.',
    )
    resume_parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to checkpoint.json or its parent directory.',
    )
    resume_parser.add_argument(
        'workflow',
        type=str,
        help='Path to the YAML workflow definition file.',
    )
    resume_parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path.  If omitted, prints summary to stdout.',
    )
    resume_parser.add_argument(
        '--output-format',
        type=str,
        default=None,
        help='Output format override (e.g., geotiff, numpy, png).',
    )
    resume_parser.add_argument(
        '--prefer-gpu',
        action='store_true',
        default=False,
        help='Attempt GPU acceleration for compatible steps.',
    )

    # ── history subcommand ──────────────────────────────────────────
    history_parser = subparsers.add_parser(
        'history',
        help='List recent workflow executions.',
    )
    history_parser.add_argument(
        '--status',
        type=str,
        default=None,
        choices=['running', 'success', 'failed', 'cancelled'],
        help='Filter by execution status.',
    )
    history_parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Show details for a specific run_id.',
    )
    history_parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Maximum number of records to show (default: 20).',
    )
    history_parser.add_argument(
        '--json',
        action='store_true',
        default=False,
        dest='json_output',
        help='Output in JSON format.',
    )
    history_parser.add_argument(
        '--db',
        type=str,
        default=None,
        help='Path to history database (default: ~/.grdl_rt/history.db).',
    )

    # ── list-processors subcommand ───────────────────────────────────
    list_parser = subparsers.add_parser(
        'list-processors',
        help='List available GRDL processors from the catalog.',
    )
    list_parser.add_argument(
        '--modality',
        type=str,
        default=None,
        help='Filter by image modality (e.g., SAR, EO).',
    )
    list_parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Filter by processor category.',
    )
    list_parser.add_argument(
        '--json',
        action='store_true',
        default=False,
        dest='json_output',
        help='Output in JSON format.',
    )

    return parser


def _get_version() -> str:
    """Return the package version."""
    try:
        from grdl_rt import __version__
        return __version__
    except ImportError:
        return "unknown"


def _read_input(input_path: Path) -> np.ndarray:
    """Read input image from a file path.

    Supports .npy files directly and delegates to grdl.IO.open_image
    for raster formats.

    Parameters
    ----------
    input_path : Path
        Path to the input file.

    Returns
    -------
    np.ndarray
    """
    ext = input_path.suffix.lower()

    if ext == '.npy':
        return np.load(str(input_path))

    # Use grdl.IO for raster formats
    try:
        from grdl.IO import open_image
        with open_image(input_path) as reader:
            return reader.read_full()
    except ImportError:
        raise ImportError(
            "grdl.IO is required for reading raster images.  "
            "Install grdl or use a .npy input file."
        )


def _progress_bar(fraction: float) -> None:
    """Simple console progress bar callback."""
    bar_width = 40
    filled = int(bar_width * fraction)
    bar = '#' * filled + '-' * (bar_width - filled)
    pct = int(fraction * 100)
    sys.stderr.write(f'\r  [{bar}] {pct:3d}%')
    if fraction >= 1.0:
        sys.stderr.write('\n')
    sys.stderr.flush()


def _run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand.

    Returns
    -------
    int
        Exit code.
    """
    from grdl_rt.api import load_workflow
    from grdl_rt.execution.executor import WorkflowExecutor
    from grdl_rt.execution.gpu import GpuBackend

    workflow_path = Path(args.workflow)
    input_path = Path(args.input)

    if not workflow_path.exists():
        logger.error("workflow_not_found", path=str(workflow_path))
        return EXIT_INPUT_ERROR

    if not input_path.exists():
        logger.error("input_not_found", path=str(input_path))
        return EXIT_INPUT_ERROR

    # Load workflow
    logger.info("workflow_loading", path=str(workflow_path))
    wf = load_workflow(workflow_path)

    # Validate before execution
    errors = wf.validate()
    if errors:
        for err in errors:
            step_str = f"step {err.step_index}" if err.step_index is not None else "workflow"
            logger.error(
                "validation_error", code=err.code,
                step=step_str, message=err.message,
            )
        return EXIT_VALIDATION_FAILURE

    # Read input
    logger.info("input_reading", path=str(input_path))
    try:
        source = _read_input(input_path)
    except Exception as e:
        logger.error("input_read_failed", path=str(input_path), error=str(e))
        return EXIT_INPUT_ERROR
    logger.info("input_loaded", shape=list(source.shape), dtype=str(source.dtype))

    # Create executor
    gpu = GpuBackend(prefer_gpu=args.prefer_gpu)
    executor = WorkflowExecutor(wf, gpu=gpu)

    # Execute
    logger.info(
        "workflow_executing", workflow_name=wf.name,
        step_count=len(wf.steps),
    )
    try:
        wr = executor.execute(
            source,
            progress_callback=_progress_bar,
            auto_tap_out=args.auto_tap_out,
            tap_out_format=args.tap_out_format,
            tap_out_dir=args.tap_out_dir,
        )
    except Exception as e:
        logger.error("execution_failed", error=str(e))
        return EXIT_EXECUTION_FAILURE

    # Print metrics summary to stderr
    sys.stderr.write(wr.metrics.to_json() + '\n')

    # Write output
    if args.output:
        output_path = Path(args.output)
        logger.info("output_writing", path=str(output_path))
        try:
            from grdl.IO import write as io_write
            io_write(wr.result, output_path, format=args.output_format)
        except ImportError:
            # Fallback to numpy if grdl.IO unavailable
            np.save(str(output_path), wr.result)
        print(f"Output written to {output_path}")
    else:
        r = wr.result
        print(f"Result: shape={r.shape}, dtype={r.dtype}, "
              f"min={r.min():.4f}, max={r.max():.4f}")
        print(f"Timing: {wr.metrics.total_wall_time_s:.3f}s wall, "
              f"{wr.metrics.total_cpu_time_s:.3f}s CPU")

    return EXIT_SUCCESS


def _validate(args: argparse.Namespace) -> int:
    """Execute the 'validate' subcommand.

    Returns
    -------
    int
        Exit code.
    """
    from grdl_rt.api import load_workflow

    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        logger.error("workflow_not_found", path=str(workflow_path))
        return EXIT_INPUT_ERROR

    wf = load_workflow(workflow_path)
    errors = wf.validate()

    if not errors:
        print(f"Workflow '{wf.name}' is valid ({len(wf.steps)} steps).")
        return EXIT_SUCCESS

    print(f"Workflow '{wf.name}' has {len(errors)} validation error(s):")
    for err in errors:
        step_str = f"step {err.step_index}" if err.step_index is not None else "workflow"
        print(f"  [{err.code}] {step_str}: {err.message}")
    return EXIT_VALIDATION_FAILURE


def _list_processors(args: argparse.Namespace) -> int:
    """Execute the 'list-processors' subcommand.

    Returns
    -------
    int
        Exit code.
    """
    from grdl_rt.execution.discovery import (
        discover_processors,
        filter_processors,
        get_processor_tags,
    )

    if args.modality or args.category:
        procs = filter_processors(
            modality=args.modality,
            category=args.category,
        )
    else:
        procs = discover_processors()

    if args.json_output:
        entries = []
        for name, cls in sorted(procs.items()):
            tags = get_processor_tags(cls)
            entries.append({
                "name": name,
                "class": f"{cls.__module__}.{cls.__qualname__}",
                "tags": tags,
            })
        print(json_mod.dumps(entries, indent=2, default=str))
    else:
        if not procs:
            print("No processors found.")
        else:
            for name in sorted(procs):
                print(f"  {name}")
            print(f"\n{len(procs)} processor(s) found.")

    return EXIT_SUCCESS


def _resume(args: argparse.Namespace) -> int:
    """Execute the 'resume' subcommand.

    Returns
    -------
    int
        Exit code.
    """
    from grdl_rt.api import load_workflow
    from grdl_rt.execution.checkpoint import CheckpointManager
    from grdl_rt.execution.errors import CheckpointError, ResumeError
    from grdl_rt.execution.executor import WorkflowExecutor
    from grdl_rt.execution.gpu import GpuBackend

    checkpoint_path = Path(args.checkpoint)
    workflow_path = Path(args.workflow)

    if not checkpoint_path.exists():
        logger.error("checkpoint_not_found", path=str(checkpoint_path))
        return EXIT_INPUT_ERROR

    if not workflow_path.exists():
        logger.error("workflow_not_found", path=str(workflow_path))
        return EXIT_INPUT_ERROR

    # Load workflow
    wf = load_workflow(workflow_path)

    # Create checkpoint manager (use parent of checkpoint as dir)
    if checkpoint_path.is_dir():
        ckpt_dir = checkpoint_path.parent
    else:
        ckpt_dir = checkpoint_path.parent.parent
    ckpt_mgr = CheckpointManager(ckpt_dir)

    # Create executor
    gpu = GpuBackend(prefer_gpu=args.prefer_gpu)
    executor = WorkflowExecutor(wf, gpu=gpu, checkpoint_manager=ckpt_mgr)

    # Resume
    logger.info("workflow_resuming", checkpoint=str(checkpoint_path))
    try:
        wr = executor.resume(
            checkpoint_path,
            progress_callback=_progress_bar,
            enable_checkpointing=True,
        )
    except (ResumeError, CheckpointError) as e:
        print(f"Resume failed: {e}", file=sys.stderr)
        return EXIT_EXECUTION_FAILURE
    except Exception as e:
        logger.error("execution_failed", error=str(e))
        return EXIT_EXECUTION_FAILURE

    # Print metrics
    sys.stderr.write(wr.metrics.to_json() + '\n')

    # Write output
    if args.output:
        output_path = Path(args.output)
        try:
            from grdl.IO import write as io_write
            io_write(wr.result, output_path, format=args.output_format)
        except ImportError:
            np.save(str(output_path), wr.result)
        print(f"Output written to {output_path}")
    else:
        r = wr.result
        print(f"Result: shape={r.shape}, dtype={r.dtype}, "
              f"min={r.min():.4f}, max={r.max():.4f}")
        print(f"Timing: {wr.metrics.total_wall_time_s:.3f}s wall, "
              f"{wr.metrics.total_cpu_time_s:.3f}s CPU")

    return EXIT_SUCCESS


def _history(args: argparse.Namespace) -> int:
    """Execute the 'history' subcommand.

    Returns
    -------
    int
        Exit code.
    """
    from grdl_rt.execution.history import ExecutionHistoryDB

    db_path = Path(args.db) if args.db else None
    try:
        db = ExecutionHistoryDB(db_path=db_path)
    except Exception as e:
        print(f"Failed to open history database: {e}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    try:
        if args.run_id:
            record = db.get_execution(args.run_id)
            if record is None:
                print(f"No execution found with run_id: {args.run_id}")
                return EXIT_SUCCESS

            if args.json_output:
                detail = {
                    "workflow_id": record.workflow_id,
                    "run_id": record.run_id,
                    "workflow_hash": record.workflow_hash,
                    "start_time": record.start_time,
                    "end_time": record.end_time,
                    "status": record.status,
                    "step_count": record.step_count,
                    "error_message": record.error_message,
                    "checkpoint_path": record.checkpoint_path,
                }
                if record.metrics_json:
                    detail["metrics"] = json_mod.loads(record.metrics_json)
                if record.parameters_json:
                    detail["parameters"] = json_mod.loads(record.parameters_json)
                print(json_mod.dumps(detail, indent=2, default=str))
            else:
                print(f"Run ID:    {record.run_id}")
                print(f"Workflow:  {record.workflow_id}")
                print(f"Status:    {record.status}")
                print(f"Started:   {record.start_time}")
                print(f"Ended:     {record.end_time or '(still running)'}")
                print(f"Steps:     {record.step_count}")
                if record.error_message:
                    print(f"Error:     {record.error_message}")
                if record.checkpoint_path:
                    print(f"Checkpoint: {record.checkpoint_path}")
            return EXIT_SUCCESS

        records = db.list_executions(status=args.status, limit=args.limit)

        if args.json_output:
            entries = [
                {
                    "run_id": r.run_id,
                    "workflow_id": r.workflow_id,
                    "status": r.status,
                    "start_time": r.start_time,
                    "end_time": r.end_time,
                    "step_count": r.step_count,
                    "error_message": r.error_message,
                }
                for r in records
            ]
            print(json_mod.dumps(entries, indent=2, default=str))
        else:
            if not records:
                print("No executions found.")
            else:
                for r in records:
                    status_icon = {
                        "success": "+",
                        "failed": "!",
                        "running": "~",
                        "cancelled": "x",
                    }.get(r.status, "?")
                    end = r.end_time or "(running)"
                    print(
                        f"  [{status_icon}] {r.run_id[:8]}  "
                        f"{r.workflow_id:<30s}  {r.status:<10s}  "
                        f"{r.start_time}"
                    )
                print(f"\n{len(records)} execution(s) shown.")

        return EXIT_SUCCESS
    finally:
        db.close()


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose >= 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    configure_logging(json_output=args.json_log, level=level)

    if args.command is None:
        parser.print_help()
        return EXIT_SUCCESS

    if args.command == 'run':
        return _run(args)
    elif args.command == 'validate':
        return _validate(args)
    elif args.command == 'resume':
        return _resume(args)
    elif args.command == 'history':
        return _history(args)
    elif args.command == 'list-processors':
        return _list_processors(args)

    parser.print_help()
    return EXIT_SUCCESS


if __name__ == '__main__':
    sys.exit(main())
