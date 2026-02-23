"""
Tests for grdl_rt.execution.lineage — data lineage provenance tracking.

Verifies array hashing, LineageTransform / DataLineage serialization,
build_lineage construction from execution artifacts, and GeoTIFF
lineage embedding.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-11
"""

import json
import re
from unittest.mock import MagicMock, patch

import numpy as np

from grdl_rt.execution.lineage import (
    DataLineage,
    LineageTransform,
    build_lineage,
    compute_array_hash,
    embed_lineage_geotiff,
)
from grdl_rt.execution.workflow import ProcessingStep

# ---------------------------------------------------------------------------
# compute_array_hash
# ---------------------------------------------------------------------------


class TestComputeArrayHash:
    def test_deterministic(self, grayscale_8x8):
        """Same array always produces the same hash."""
        h1 = compute_array_hash(grayscale_8x8)
        h2 = compute_array_hash(grayscale_8x8)
        assert h1 == h2

    def test_different_data(self, grayscale_8x8):
        """Different array contents produce different hashes."""
        other = np.ones((8, 8), dtype=np.float32)
        h1 = compute_array_hash(grayscale_8x8)
        h2 = compute_array_hash(other)
        assert h1 != h2

    def test_hex_format(self, grayscale_8x8):
        """Result is a 64-character lowercase hex string (SHA-256)."""
        h = compute_array_hash(grayscale_8x8)
        assert len(h) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", h) is not None


# ---------------------------------------------------------------------------
# LineageTransform
# ---------------------------------------------------------------------------


class TestLineageTransform:
    def test_to_dict(self):
        """Serialization includes step_index, processor_name, and params."""
        lt = LineageTransform(
            step_index=0,
            processor_name="LeeFilter",
            params={"window_size": 7},
        )
        d = lt.to_dict()
        assert d["step_index"] == 0
        assert d["processor_name"] == "LeeFilter"
        assert d["params"] == {"window_size": 7}

    def test_to_dict_no_version(self):
        """When processor_version is None the key is omitted from dict."""
        lt = LineageTransform(
            step_index=1,
            processor_name="Threshold",
            processor_version=None,
            params={"t": 0.5},
        )
        d = lt.to_dict()
        assert "processor_version" not in d

    def test_to_dict_with_version(self):
        """When processor_version is set, it appears in the dict."""
        lt = LineageTransform(
            step_index=2,
            processor_name="Orthorectifier",
            processor_version="1.2.0",
            params={"method": "bilinear"},
        )
        d = lt.to_dict()
        assert d["processor_version"] == "1.2.0"


# ---------------------------------------------------------------------------
# DataLineage round-trip
# ---------------------------------------------------------------------------


class TestDataLineage:
    def test_roundtrip(self):
        """to_dict -> from_dict produces an equivalent DataLineage."""
        original = DataLineage(
            input_hash="a" * 64,
            input_shape=[8, 8],
            input_dtype="float32",
            transforms=[
                LineageTransform(
                    step_index=0,
                    processor_name="LeeFilter",
                    processor_version="1.0.0",
                    params={"window_size": 7},
                ),
                LineageTransform(
                    step_index=1,
                    processor_name="Threshold",
                    processor_version=None,
                    params={"t": 0.5},
                ),
            ],
            output_hash="b" * 64,
            output_shape=[8, 8],
            output_dtype="float32",
        )
        d = original.to_dict()
        restored = DataLineage.from_dict(d)

        assert restored.input_hash == original.input_hash
        assert restored.input_shape == original.input_shape
        assert restored.input_dtype == original.input_dtype
        assert restored.output_hash == original.output_hash
        assert restored.output_shape == original.output_shape
        assert restored.output_dtype == original.output_dtype

        assert len(restored.transforms) == 2
        assert restored.transforms[0].processor_name == "LeeFilter"
        assert restored.transforms[0].processor_version == "1.0.0"
        assert restored.transforms[0].params == {"window_size": 7}
        assert restored.transforms[1].processor_name == "Threshold"
        assert restored.transforms[1].processor_version is None
        assert restored.transforms[1].params == {"t": 0.5}


# ---------------------------------------------------------------------------
# build_lineage
# ---------------------------------------------------------------------------


class TestBuildLineage:
    def test_basic(self):
        """Build lineage from source, result, mock steps and metrics."""
        source = np.ones((4, 4), dtype=np.float32)
        result = np.ones((4, 4), dtype=np.float32) * 2.0

        # Real ProcessingStep objects (build_lineage uses isinstance check)
        step_a = ProcessingStep(
            processor_name="ScaleTransform",
            processor_version="1.0.0",
            params={"scale": 2.0},
        )

        # Mock step metrics (StepMetrics-like objects)
        metric_a = MagicMock()
        metric_a.step_index = 0
        metric_a.processor_name = "ScaleTransform"

        lineage = build_lineage(
            source=source,
            result=result,
            workflow_steps=[step_a],
            step_metrics=[metric_a],
        )

        assert lineage.input_hash == compute_array_hash(source)
        assert lineage.output_hash == compute_array_hash(result)
        assert lineage.input_shape == [4, 4]
        assert lineage.output_shape == [4, 4]
        assert lineage.input_dtype == "float32"
        assert lineage.output_dtype == "float32"

        assert len(lineage.transforms) == 1
        t = lineage.transforms[0]
        assert t.step_index == 0
        assert t.processor_name == "ScaleTransform"
        assert t.params == {"scale": 2.0}
        # Discovery won't resolve test mock processors
        assert t.processor_version is None


# ---------------------------------------------------------------------------
# embed_lineage_geotiff
# ---------------------------------------------------------------------------


class TestEmbedLineageGeotiff:
    def _make_lineage(self):
        """Create a minimal DataLineage for embedding tests."""
        return DataLineage(
            input_hash="a" * 64,
            input_shape=[4, 4],
            input_dtype="float32",
            transforms=[],
            output_hash="b" * 64,
            output_shape=[4, 4],
            output_dtype="float32",
        )

    @patch.dict("sys.modules", {"rasterio": None})
    def test_no_rasterio(self):
        """When rasterio import fails, no error is raised."""
        lineage = self._make_lineage()
        # Should return silently without raising
        embed_lineage_geotiff("/tmp/fake.tif", lineage)

    def test_writes_tag(self):
        """Mock rasterio and verify update_tags is called with GRDL_LINEAGE."""
        lineage = self._make_lineage()

        mock_ds = MagicMock()
        mock_ds.__enter__ = MagicMock(return_value=mock_ds)
        mock_ds.__exit__ = MagicMock(return_value=False)

        mock_rasterio = MagicMock()
        mock_rasterio.open.return_value = mock_ds

        with patch.dict("sys.modules", {"rasterio": mock_rasterio}):
            embed_lineage_geotiff("/tmp/output.tif", lineage)

        mock_rasterio.open.assert_called_once_with("/tmp/output.tif", "r+")
        mock_ds.update_tags.assert_called_once()

        call_kwargs = mock_ds.update_tags.call_args
        tag_value = call_kwargs.kwargs.get("GRDL_LINEAGE") or call_kwargs[1]["GRDL_LINEAGE"]
        parsed = json.loads(tag_value)
        assert parsed["input_hash"] == "a" * 64
        assert parsed["output_hash"] == "b" * 64
