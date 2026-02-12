# -*- coding: utf-8 -*-
"""
Workflow Result — Return type for all execution paths.

Wraps the processed image array together with structured execution metrics,
ensuring timing and resource data are always available after a workflow run.

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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

# Third-party
import numpy as np

# grdl-runtime internal
from grdl_rt.execution.metrics import WorkflowMetrics

if TYPE_CHECKING:
    from grdl_rt.execution.lineage import DataLineage


@dataclass
class WorkflowResult:
    """Return type from all execution paths.

    Every ``execute()`` call on ``WorkflowExecutor``, ``Workflow``, or
    ``DAGExecutor`` returns a ``WorkflowResult`` containing both the
    processed image and structured metrics.

    Attributes
    ----------
    result : np.ndarray
        The processed image array.  For DAG workflows with a single
        terminal node, this is that node's output.  For multiple
        terminals, this is the last terminal in topological order.
    metrics : WorkflowMetrics
        Timing and resource usage metrics for the run.
    step_results : Optional[Dict[str, np.ndarray]]
        For DAG executions, the full results map keyed by step ID.
        ``None`` for linear executions.
    lineage : Optional[DataLineage]
        Data lineage record mapping input to output through transforms.
        ``None`` if lineage tracking was not enabled.
    """

    result: np.ndarray
    metrics: WorkflowMetrics
    step_results: Optional[Dict[str, np.ndarray]] = None
    lineage: Optional[DataLineage] = None
