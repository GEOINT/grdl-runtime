"""
Checkpoint / Resume — Versioned checkpoint files with workflow hash
validation and atomic writes.

Enables long workflows to survive failures.  After each step (when
checkpointing is enabled), a ``CheckpointState`` is written to disk
atomically (write-to-temp then rename).  Resume validates that the
workflow definition has not changed (hash match) and that all referenced
intermediate files still exist on disk.

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

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from grdl_rt.execution.context import get_logger
from grdl_rt.execution.errors import CheckpointError, ResumeError

logger = get_logger(__name__)

CHECKPOINT_SCHEMA_VERSION = "2.0"


def compute_workflow_hash(workflow_dict: dict[str, Any]) -> str:
    """Compute a stable SHA-256 hash of a workflow definition.

    The hash covers the workflow name, version, and the full step
    definitions (processor names, versions, params).  It deliberately
    excludes mutable metadata like ``state`` and ``description`` so
    that documentation-only changes do not invalidate checkpoints.

    Parameters
    ----------
    workflow_dict : dict
        Serialised workflow definition (``WorkflowDefinition.to_dict()``).

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    # Extract only the structurally significant fields
    canonical = {
        "name": workflow_dict.get("name", ""),
        "version": workflow_dict.get("version", ""),
        "steps": workflow_dict.get("steps", []),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class CheckpointState:
    """Versioned checkpoint persisted after each completed step.

    Attributes
    ----------
    schema_version : str
        Checkpoint format version (for forward-compatibility).
    workflow_definition_hash : str
        SHA-256 of the canonical workflow definition.
    step_index : int
        Index of the last completed step (-1 if none).
    intermediate_files : List[str]
        Paths to serialised intermediate arrays, one per completed
        processing step.
    metrics_so_far : List[Dict[str, Any]]
        Per-step metrics collected so far.
    execution_context : Dict[str, Any]
        Context dict (workflow_id, run_id, workflow_name, etc.).
    timestamp : str
        ISO 8601 UTC timestamp of this checkpoint write.
    workflow_dict : Dict[str, Any]
        Full serialised workflow definition for reference.
    """

    schema_version: str = CHECKPOINT_SCHEMA_VERSION
    workflow_definition_hash: str = ""
    step_index: int = -1
    intermediate_files: list[str] = field(default_factory=list)
    metrics_so_far: list[dict[str, Any]] = field(default_factory=list)
    execution_context: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    workflow_dict: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "schema_version": self.schema_version,
            "workflow_definition_hash": self.workflow_definition_hash,
            "step_index": self.step_index,
            "intermediate_files": self.intermediate_files,
            "metrics_so_far": self.metrics_so_far,
            "execution_context": self.execution_context,
            "timestamp": self.timestamp,
            "workflow_dict": self.workflow_dict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        """Deserialise from a dictionary."""
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            workflow_definition_hash=data.get("workflow_definition_hash", ""),
            step_index=data.get("step_index", -1),
            intermediate_files=data.get("intermediate_files", []),
            metrics_so_far=data.get("metrics_so_far", []),
            execution_context=data.get("execution_context", {}),
            timestamp=data.get("timestamp", ""),
            workflow_dict=data.get("workflow_dict", {}),
        )


class CheckpointManager:
    """Write and read checkpoint files with atomic-write semantics.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory where checkpoint folders are created.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir

    def run_dir(self, run_id: str) -> Path:
        """Return the checkpoint sub-directory for a given run."""
        return self._checkpoint_dir / f"grdl_checkpoint_{run_id}"

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write_step_intermediate(
        self,
        run_id: str,
        step_index: int,
        array: np.ndarray,
    ) -> str:
        """Save an intermediate array for one completed step.

        Returns
        -------
        str
            Path (relative to checkpoint dir) of the saved ``.npy`` file.
        """
        rd = self.run_dir(run_id)
        rd.mkdir(parents=True, exist_ok=True)
        filename = f"intermediate_step_{step_index:04d}.npy"
        dest = rd / filename
        # Atomic: write to temp then rename
        fd, tmp = tempfile.mkstemp(suffix=".npy", dir=str(rd), prefix=".tmp_")
        try:
            os.close(fd)
            np.save(tmp, array)
            os.replace(tmp, str(dest))
        except Exception:
            # Clean up temp file on failure
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return str(dest)

    def write_checkpoint(
        self,
        state: CheckpointState,
        run_id: str,
    ) -> Path:
        """Atomically write a checkpoint JSON file.

        Parameters
        ----------
        state : CheckpointState
        run_id : str

        Returns
        -------
        Path
            Path to ``checkpoint.json``.
        """
        rd = self.run_dir(run_id)
        rd.mkdir(parents=True, exist_ok=True)
        dest = rd / "checkpoint.json"
        payload = json.dumps(state.to_dict(), indent=2, default=str)

        fd, tmp = tempfile.mkstemp(suffix=".json", dir=str(rd), prefix=".tmp_")
        try:
            os.close(fd)
            Path(tmp).write_text(payload, encoding="utf-8")
            os.replace(tmp, str(dest))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

        logger.info(
            "checkpoint_written",
            checkpoint_path=str(dest),
            step_index=state.step_index,
        )
        return dest

    # ------------------------------------------------------------------
    # Reading / validation
    # ------------------------------------------------------------------

    @staticmethod
    def load_checkpoint(checkpoint_path: Path) -> CheckpointState:
        """Load and parse a checkpoint JSON file.

        Parameters
        ----------
        checkpoint_path : Path
            Path to ``checkpoint.json`` (or its parent directory).

        Returns
        -------
        CheckpointState

        Raises
        ------
        CheckpointError
            If the file cannot be read or parsed.
        """
        p = Path(checkpoint_path)
        if p.is_dir():
            p = p / "checkpoint.json"
        if not p.exists():
            raise CheckpointError(f"Checkpoint file not found: {p}", path=str(p))
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CheckpointError(f"Failed to read checkpoint: {exc}", path=str(p)) from exc

        return CheckpointState.from_dict(data)

    @staticmethod
    def validate_for_resume(
        state: CheckpointState,
        workflow_dict: dict[str, Any],
    ) -> None:
        """Validate that a checkpoint is compatible with the given workflow.

        Raises
        ------
        ResumeError
            If any validation check fails.
        """
        # 1. Schema version compatibility
        major = state.schema_version.split(".")[0]
        if major not in ("1", "2"):
            raise ResumeError(f"Unsupported checkpoint schema version: " f"{state.schema_version}")

        # 2. Workflow hash match
        current_hash = compute_workflow_hash(workflow_dict)
        if state.workflow_definition_hash != current_hash:
            raise ResumeError(
                "Workflow definition has changed since checkpoint was created "
                f"(checkpoint hash: {state.workflow_definition_hash[:12]}..., "
                f"current hash: {current_hash[:12]}...)"
            )

        # 3. All intermediate files still exist
        missing = [f for f in state.intermediate_files if not Path(f).exists()]
        if missing:
            raise ResumeError(
                f"{len(missing)} intermediate file(s) missing: "
                f"{', '.join(missing[:3])}" + ("..." if len(missing) > 3 else "")
            )

    @staticmethod
    def load_last_intermediate(state: CheckpointState) -> np.ndarray:
        """Load the most recent intermediate array from a checkpoint.

        Returns
        -------
        np.ndarray

        Raises
        ------
        CheckpointError
            If no intermediates exist or the file cannot be loaded.
        """
        if not state.intermediate_files:
            raise CheckpointError("Checkpoint has no intermediate files to resume from")
        last = state.intermediate_files[-1]
        if not Path(last).exists():
            raise CheckpointError(f"Intermediate file not found: {last}", path=last)
        return np.load(last)
