# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.context — ExecutionContext and structured logging.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-11
"""

import io
import json
import logging
import uuid

import pytest
import structlog

from grdl_rt.execution.context import (
    ExecutionContext,
    configure_logging,
    get_logger,
)


# ---------------------------------------------------------------------------
# ExecutionContext
# ---------------------------------------------------------------------------

class TestExecutionContext:
    def test_fields_assigned(self):
        ctx = ExecutionContext(
            workflow_id="wf:1.0",
            workflow_name="My Workflow",
            run_id="abc-123",
        )
        assert ctx.workflow_id == "wf:1.0"
        assert ctx.workflow_name == "My Workflow"
        assert ctx.run_id == "abc-123"

    def test_auto_generates_run_id(self):
        ctx = ExecutionContext(workflow_id="wf:1.0", workflow_name="Test")
        # Should be a valid UUID4
        parsed = uuid.UUID(ctx.run_id, version=4)
        assert str(parsed) == ctx.run_id

    def test_frozen(self):
        ctx = ExecutionContext(workflow_id="wf:1.0", workflow_name="Test")
        with pytest.raises(AttributeError):
            ctx.workflow_id = "changed"  # type: ignore[misc]

    def test_as_log_dict(self):
        ctx = ExecutionContext(
            workflow_id="wf:1.0",
            workflow_name="Test",
            run_id="run-42",
        )
        d = ctx.as_log_dict()
        assert d == {
            "workflow_id": "wf:1.0",
            "workflow_name": "Test",
            "run_id": "run-42",
        }

    def test_two_contexts_have_different_run_ids(self):
        ctx1 = ExecutionContext(workflow_id="wf:1.0", workflow_name="A")
        ctx2 = ExecutionContext(workflow_id="wf:1.0", workflow_name="A")
        assert ctx1.run_id != ctx2.run_id


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def test_json_output_produces_parseable_json(self):
        configure_logging(json_output=True, level=logging.DEBUG)
        stream = io.StringIO()

        handler = logging.StreamHandler(stream)
        formatter = logging.getLogger().handlers[0].formatter
        handler.setFormatter(formatter)

        root = logging.getLogger()
        root.addHandler(handler)
        try:
            log = structlog.get_logger("test_json")
            log.info("test_event", key="value")

            output = stream.getvalue()
            # Should be valid JSON
            for line in output.strip().splitlines():
                parsed = json.loads(line)
                assert parsed["event"] == "test_event"
                assert parsed["key"] == "value"
        finally:
            root.removeHandler(handler)

    def test_console_output_does_not_raise(self):
        configure_logging(json_output=False, level=logging.WARNING)

    def test_reconfigure_is_safe(self):
        configure_logging(json_output=True, level=logging.INFO)
        configure_logging(json_output=False, level=logging.DEBUG)


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_bound_logger(self):
        log = get_logger("test_module")
        assert hasattr(log, "info")
        assert hasattr(log, "debug")
        assert hasattr(log, "error")

    def test_logger_accepts_kwargs(self):
        log = get_logger("test_module")
        # Should not raise
        log.debug("event_name", step_index=0, processor_name="test")
