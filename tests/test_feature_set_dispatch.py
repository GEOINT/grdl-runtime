"""
Tests for FeatureSet-aware dispatch and FeatureSetAggregator.

Covers:
- GPU transfer skipped for non-array data
- FeatureSetAggregator union strategy

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-03-25
"""

import pytest

from grdl_rt.execution.dispatch import supports_gpu_transfer
from grdl_rt.execution.operators import FeatureSetAggregator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeVectorProcessor:
    """Mock processor that should not receive GPU data."""

    __gpu_compatible__ = False


class _FakeGpuProcessor:
    """Mock processor that is GPU compatible."""

    __gpu_compatible__ = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDispatchGpuSkip:
    """Test that GPU transfer is skipped for non-array processors."""

    def test_non_gpu_processor(self):
        proc = _FakeVectorProcessor()
        assert supports_gpu_transfer(proc) is False

    def test_gpu_processor(self):
        proc = _FakeGpuProcessor()
        assert supports_gpu_transfer(proc) is True

    def test_plain_object(self):
        """Objects without __gpu_compatible__ default to False."""
        assert supports_gpu_transfer(object()) is False


class TestFeatureSetAggregatorUnion:
    """Test FeatureSetAggregator with union strategy."""

    def test_union_merge(self):
        agg = FeatureSetAggregator(strategy="union")

        # Simulate ImageMetadata
        class _Meta:
            pass

        meta = _Meta()

        inputs = {
            "step_a": {
                "features": [{"id": 1, "geom": "A"}, {"id": 2, "geom": "B"}],
                "type": "FeatureSet",
            },
            "step_b": {"features": [{"id": 3, "geom": "C"}], "type": "FeatureSet"},
        }

        result, out_meta = agg.execute(meta, inputs)
        assert isinstance(result, dict)
        assert result["type"] == "FeatureSet"
        assert len(result["features"]) == 3

    def test_union_empty_inputs(self):
        agg = FeatureSetAggregator(strategy="union")

        class _Meta:
            pass

        result, _ = agg.execute(_Meta(), {})
        assert result["features"] == []

    def test_non_dict_raises(self):
        agg = FeatureSetAggregator(strategy="union")

        class _Meta:
            pass

        with pytest.raises(TypeError, match="expects a dict"):
            agg.execute(_Meta(), "not a dict")

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy must be"):
            FeatureSetAggregator(strategy="invalid")


class TestFeatureSetAggregatorIntersection:
    """Test FeatureSetAggregator with intersection strategy."""

    def test_intersection_merge(self):
        agg = FeatureSetAggregator(strategy="intersection")

        class _Meta:
            pass

        inputs = {
            "step_a": {"features": [{"id": 1}, {"id": 2}, {"id": 3}], "type": "FeatureSet"},
            "step_b": {"features": [{"id": 2}, {"id": 3}, {"id": 4}], "type": "FeatureSet"},
        }

        result, _ = agg.execute(_Meta(), inputs)
        assert isinstance(result, dict)
        ids = {f["id"] for f in result["features"]}
        assert ids == {2, 3}

    def test_intersection_empty(self):
        agg = FeatureSetAggregator(strategy="intersection")

        class _Meta:
            pass

        result, _ = agg.execute(_Meta(), {})
        assert result["features"] == []
