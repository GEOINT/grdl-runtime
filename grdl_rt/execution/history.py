"""
Execution History Journal — SQLite-backed audit trail for workflow runs.

Records one row per execution with timing, status, parameters, and metrics.
Provides query helpers for the ``grdl-rt history`` CLI command.

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

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grdl_rt.execution.context import get_logger

logger = get_logger(__name__)

_DEFAULT_HISTORY_DIR = Path.home() / ".grdl_rt"
_DEFAULT_HISTORY_DB = _DEFAULT_HISTORY_DIR / "history.db"

_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    run_id TEXT NOT NULL UNIQUE,
    workflow_hash TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK(status IN ('running', 'success', 'failed', 'cancelled')),
    step_count INTEGER DEFAULT 0,
    error_message TEXT,
    parameters_json TEXT,
    output_hash TEXT,
    metrics_json TEXT,
    checkpoint_path TEXT
);

CREATE TABLE IF NOT EXISTS history_schema_version (
    version INTEGER NOT NULL
);
"""

_HISTORY_SCHEMA_VERSION = 1


@dataclass
class ExecutionRecord:
    """A single execution history entry.

    Attributes
    ----------
    id : int or None
        Database row ID (None for unsaved records).
    workflow_id : str
        Workflow identifier (name:version).
    run_id : str
        Unique run UUID.
    workflow_hash : str
        SHA-256 of the workflow definition.
    start_time : str
        ISO 8601 UTC timestamp.
    end_time : str or None
        ISO 8601 UTC timestamp, None if still running.
    status : str
        One of: running, success, failed, cancelled.
    step_count : int
        Number of steps executed (may be partial on failure).
    error_message : str or None
        Error details if status is failed.
    parameters_json : str or None
        JSON-encoded workflow parameters.
    output_hash : str or None
        Hash of the final output (if computed).
    metrics_json : str or None
        JSON-encoded WorkflowMetrics.
    checkpoint_path : str or None
        Path to the checkpoint directory, if one exists.
    """

    id: int | None = None
    workflow_id: str = ""
    run_id: str = ""
    workflow_hash: str = ""
    start_time: str = ""
    end_time: str | None = None
    status: str = "running"
    step_count: int = 0
    error_message: str | None = None
    parameters_json: str | None = None
    output_hash: str | None = None
    metrics_json: str | None = None
    checkpoint_path: str | None = None


class ExecutionHistoryDB:
    """SQLite-backed execution history journal.

    Parameters
    ----------
    db_path : Path or None
        Path to the SQLite database file.  Defaults to
        ``~/.grdl_rt/history.db``.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            db_path = _DEFAULT_HISTORY_DB
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they do not exist."""
        self._conn.executescript(_HISTORY_SCHEMA)
        self._conn.commit()

        row = self._conn.execute("SELECT version FROM history_schema_version").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO history_schema_version (version) VALUES (?)",
                (_HISTORY_SCHEMA_VERSION,),
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> ExecutionHistoryDB:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def record_start(
        self,
        workflow_id: str,
        run_id: str,
        workflow_hash: str,
        parameters_json: str | None = None,
    ) -> int:
        """Record the start of a workflow execution.

        Parameters
        ----------
        workflow_id : str
            Workflow identifier (name:version).
        run_id : str
            Unique run UUID.
        workflow_hash : str
            SHA-256 of the workflow definition.
        parameters_json : str, optional
            JSON-encoded workflow parameters.

        Returns
        -------
        int
            Database row ID.
        """
        now = datetime.now(UTC).isoformat()
        cursor = self._conn.execute(
            """INSERT INTO executions
            (workflow_id, run_id, workflow_hash, start_time, status,
             parameters_json)
            VALUES (?, ?, ?, ?, 'running', ?)""",
            (workflow_id, run_id, workflow_hash, now, parameters_json),
        )
        self._conn.commit()
        logger.debug(
            "history_record_start",
            run_id=run_id,
            workflow_id=workflow_id,
        )
        return cursor.lastrowid  # type: ignore[return-value]

    def record_completion(
        self,
        run_id: str,
        status: str,
        step_count: int,
        error_message: str | None = None,
        metrics_json: str | None = None,
        output_hash: str | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        """Update a running execution with its final status.

        Parameters
        ----------
        run_id : str
        status : str
            One of: success, failed, cancelled.
        step_count : int
        error_message : str, optional
        metrics_json : str, optional
        output_hash : str, optional
        checkpoint_path : str, optional
        """
        now = datetime.now(UTC).isoformat()
        self._conn.execute(
            """UPDATE executions SET
                end_time = ?,
                status = ?,
                step_count = ?,
                error_message = ?,
                metrics_json = ?,
                output_hash = ?,
                checkpoint_path = ?
            WHERE run_id = ?""",
            (
                now,
                status,
                step_count,
                error_message,
                metrics_json,
                output_hash,
                checkpoint_path,
                run_id,
            ),
        )
        self._conn.commit()
        logger.debug(
            "history_record_completion",
            run_id=run_id,
            status=status,
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_executions(
        self,
        status: str | None = None,
        limit: int = 20,
    ) -> list[ExecutionRecord]:
        """List recent executions, optionally filtered by status.

        Parameters
        ----------
        status : str, optional
            Filter to this status (running, success, failed, cancelled).
        limit : int
            Maximum number of records to return (default 20).

        Returns
        -------
        List[ExecutionRecord]
        """
        if status:
            rows = self._conn.execute(
                "SELECT * FROM executions WHERE status = ? " "ORDER BY start_time DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM executions ORDER BY start_time DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_execution(self, run_id: str) -> ExecutionRecord | None:
        """Get a single execution record by run_id.

        Parameters
        ----------
        run_id : str

        Returns
        -------
        ExecutionRecord or None
        """
        row = self._conn.execute(
            "SELECT * FROM executions WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def count_by_status(self) -> dict[str, int]:
        """Return counts of executions grouped by status.

        Returns
        -------
        Dict[str, int]
        """
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM executions GROUP BY status"
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Retention / cleanup
    # ------------------------------------------------------------------

    def purge_old_records(
        self,
        *,
        max_age_days: int | None = None,
        max_records: int | None = None,
    ) -> int:
        """Remove old execution records based on age or count limits.

        Parameters
        ----------
        max_age_days : int, optional
            Delete records older than this many days.
        max_records : int, optional
            Keep at most this many records (most recent first).

        Returns
        -------
        int
            Number of records deleted.
        """
        deleted = 0

        if max_age_days is not None:
            cursor = self._conn.execute(
                "DELETE FROM executions WHERE start_time < datetime('now', ?)",
                (f"-{max_age_days} days",),
            )
            deleted += cursor.rowcount

        if max_records is not None:
            cursor = self._conn.execute(
                "DELETE FROM executions WHERE id NOT IN "
                "(SELECT id FROM executions ORDER BY start_time DESC LIMIT ?)",
                (max_records,),
            )
            deleted += cursor.rowcount

        if deleted > 0:
            self._conn.commit()
            logger.info("history_purged", deleted=deleted)

        return deleted

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ExecutionRecord:
        return ExecutionRecord(
            id=row["id"],
            workflow_id=row["workflow_id"],
            run_id=row["run_id"],
            workflow_hash=row["workflow_hash"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            status=row["status"],
            step_count=row["step_count"],
            error_message=row["error_message"],
            parameters_json=row["parameters_json"],
            output_hash=row["output_hash"],
            metrics_json=row["metrics_json"],
            checkpoint_path=row["checkpoint_path"],
        )
