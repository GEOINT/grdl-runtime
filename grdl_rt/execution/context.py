"""
Execution Context — Structured logging and execution identity.

Provides ``ExecutionContext`` (immutable context bound to every log line
during a workflow run), ``configure_logging`` (one-time structlog setup),
and ``get_logger`` (module-level logger factory).

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
2026-02-11

Modified
--------
2026-02-11
"""

# Standard library
import logging
import threading
import uuid
from dataclasses import dataclass, field

# Third-party
import structlog


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable context bound to every log line during a workflow run.

    Attributes
    ----------
    workflow_id : str
        Identifier for the workflow definition (e.g., name:version or UUID).
    workflow_name : str
        Human-readable workflow name.
    run_id : str
        Unique run identifier (UUID4), auto-generated if not provided.
    """

    workflow_id: str
    workflow_name: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def as_log_dict(self) -> dict:
        """Return a dict suitable for structlog context binding.

        Returns
        -------
        dict
            Keys: ``workflow_id``, ``workflow_name``, ``run_id``.
        """
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "run_id": self.run_id,
        }


_configured = False
_configure_lock = threading.Lock()


def configure_logging(
    json_output: bool = True,
    level: int = logging.INFO,
) -> None:
    """Configure structlog for JSON or console output.

    Must be called once at process startup (CLI entry point, test conftest,
    or embedding application).  Safe to call multiple times; subsequent
    calls reconfigure.

    Parameters
    ----------
    json_output : bool
        If ``True`` (default), render as JSON lines.  If ``False``, render
        colored console output for development.
    level : int
        Logging level threshold (default ``logging.INFO``).
    """
    global _configured

    with _configure_lock:
        _configure_logging_inner(json_output, level)
        _configured = True


def _configure_logging_inner(
    json_output: bool,
    level: int,
) -> None:
    """Internal implementation of logging configuration (called under lock)."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound to the given module name.

    Parameters
    ----------
    name : str
        Module name, typically ``__name__``.

    Returns
    -------
    structlog.stdlib.BoundLogger
    """
    return structlog.get_logger(name)
