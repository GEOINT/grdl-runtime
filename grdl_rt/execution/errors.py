# -*- coding: utf-8 -*-
"""
Custom exceptions for resilient workflow execution.

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
"""

from typing import List


class StepRetryExhaustedError(Exception):
    """All retry attempts for a workflow step have been exhausted.

    Attributes
    ----------
    step_name : str
        Name of the step that failed.
    attempts : List[Exception]
        Exception from each attempt, ordered chronologically.
    """

    def __init__(self, step_name: str, attempts: List[Exception]) -> None:
        self.step_name = step_name
        self.attempts = list(attempts)
        n = len(self.attempts)
        last = self.attempts[-1] if self.attempts else "unknown"
        super().__init__(
            f"Step '{step_name}' failed after {n} attempt(s). "
            f"Last error: {last}"
        )


class StepTimeoutError(Exception):
    """A workflow step exceeded its configured timeout.

    Attributes
    ----------
    step_name : str
        Name of the step that timed out.
    timeout_seconds : float
        The configured timeout that was exceeded.
    """

    def __init__(self, step_name: str, timeout_seconds: float) -> None:
        self.step_name = step_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Step '{step_name}' timed out after {timeout_seconds}s"
        )


class CheckpointError(Exception):
    """An error occurred while writing or reading a checkpoint.

    Attributes
    ----------
    path : str
        Path to the checkpoint file or directory involved.
    """

    def __init__(self, message: str, path: str = "") -> None:
        self.path = path
        super().__init__(message)


class ResumeError(Exception):
    """Resume from checkpoint failed validation.

    Attributes
    ----------
    reason : str
        Why the resume was rejected.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Cannot resume: {reason}")


class DAGCycleError(ValueError):
    """Workflow DAG contains a cycle.

    Attributes
    ----------
    message : str
        Description of the cycle detected.
    """

    pass


class ConditionError(ValueError):
    """A condition expression is invalid or evaluation failed.

    Attributes
    ----------
    expression : str
        The condition expression that caused the error.
    """

    def __init__(self, expression: str, reason: str) -> None:
        self.expression = expression
        super().__init__(
            f"Condition '{expression}' failed: {reason}"
        )


class ResolutionError(Exception):
    """A workflow step could not be resolved to a runnable processor.

    Attributes
    ----------
    step_id : str
        ID of the step that failed resolution.
    processor_name : str
        Name of the processor that could not be resolved.
    """

    def __init__(self, step_id: str, processor_name: str, reason: str) -> None:
        self.step_id = step_id
        self.processor_name = processor_name
        super().__init__(
            f"Cannot resolve step '{step_id}' processor "
            f"'{processor_name}': {reason}"
        )


class FallbackExhaustedError(Exception):
    """Primary processor and all alternatives failed at runtime.

    Attributes
    ----------
    step_id : str
        ID of the step that failed.
    original_processor : str
        Name of the primary processor.
    tried_alternatives : List[str]
        Names of alternatives that were attempted.
    """

    def __init__(
        self, step_id: str, original: str, tried: List[str],
    ) -> None:
        self.step_id = step_id
        self.original_processor = original
        self.tried_alternatives = tried
        super().__init__(
            f"Step '{step_id}': processor '{original}' failed and "
            f"all alternatives exhausted: {tried}"
        )


class MemoryThresholdError(Exception):
    """Estimated memory usage exceeds the abort threshold.

    Attributes
    ----------
    estimated_bytes : int
        Estimated memory requirement in bytes.
    available_bytes : int
        Available system memory in bytes.
    threshold : float
        The abort threshold ratio that was exceeded (e.g. 0.95).
    """

    def __init__(
        self,
        estimated_bytes: int,
        available_bytes: int,
        threshold: float,
    ) -> None:
        self.estimated_bytes = estimated_bytes
        self.available_bytes = available_bytes
        self.threshold = threshold
        ratio = estimated_bytes / available_bytes if available_bytes > 0 else float('inf')
        super().__init__(
            f"Estimated memory {estimated_bytes / 1e9:.2f} GB is "
            f"{ratio:.0%} of available {available_bytes / 1e9:.2f} GB "
            f"(abort threshold: {threshold:.0%})"
        )
