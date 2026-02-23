# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.catalog.updater — ArtifactUpdateWorker.

Author
------
Claude Code (Anthropic)

Contributor
-----------
Steven Siebert

Created
-------
2026-02-06
"""

from unittest.mock import MagicMock, patch
import json

import pytest

from grdl_rt.catalog.models import Artifact, UpdateResult
from grdl_rt.catalog.updater import ArtifactUpdateWorker


@pytest.fixture
def mock_catalog():
    """Mock ArtifactCatalog with a list of artifacts."""
    catalog = MagicMock()
    catalog.list_artifacts.return_value = []
    return catalog


@pytest.fixture
def worker(mock_catalog):
    return ArtifactUpdateWorker(catalog=mock_catalog, timeout=5.0)


def _make_artifact(**kwargs):
    defaults = dict(
        name="test-processor",
        version="1.0.0",
        artifact_type="grdl_processor",
    )
    defaults.update(kwargs)
    return Artifact(**defaults)


# ---------------------------------------------------------------------------
# check_pypi
# ---------------------------------------------------------------------------


class TestCheckPypi:
    @patch("grdl_rt.catalog.updater.requests.get")
    def test_success(self, mock_get, worker):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"info": {"version": "2.0.0"}},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        result = worker.check_pypi("test-package")
        assert result == "2.0.0"

    @patch("grdl_rt.catalog.updater.requests.get")
    def test_http_error(self, mock_get, worker):
        import requests

        mock_get.side_effect = requests.RequestException("timeout")
        result = worker.check_pypi("test-package")
        assert result is None

    @patch("grdl_rt.catalog.updater.requests.get")
    def test_bad_json(self, mock_get, worker):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()
        mock_get.return_value.json.side_effect = json.JSONDecodeError("", "", 0)
        result = worker.check_pypi("test-package")
        assert result is None


# ---------------------------------------------------------------------------
# check_conda
# ---------------------------------------------------------------------------


class TestCheckConda:
    @patch("grdl_rt.catalog.updater.requests.get")
    def test_success_noarch(self, mock_get, worker):
        repodata = {
            "packages": {
                "test-pkg-1.0.0.tar.bz2": {
                    "name": "test-pkg",
                    "version": "1.0.0",
                },
                "test-pkg-2.0.0.tar.bz2": {
                    "name": "test-pkg",
                    "version": "2.0.0",
                },
            },
            "packages.conda": {},
        }
        resp = MagicMock(status_code=200, json=lambda: repodata)
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = worker.check_conda("test-pkg", "conda-forge")
        assert result == "2.0.0"

    @patch("grdl_rt.catalog.updater.requests.get")
    def test_not_found(self, mock_get, worker):
        repodata = {"packages": {}, "packages.conda": {}}
        resp = MagicMock(status_code=200, json=lambda: repodata)
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = worker.check_conda("nonexistent-pkg", "conda-forge")
        assert result is None

    @patch("grdl_rt.catalog.updater.requests.get")
    def test_network_error(self, mock_get, worker):
        import requests

        mock_get.side_effect = requests.RequestException("network down")
        result = worker.check_conda("test-pkg")
        assert result is None

    @patch("grdl_rt.catalog.updater.requests.get")
    def test_invalid_version_skipped(self, mock_get, worker):
        repodata = {
            "packages": {
                "pkg-bad.tar.bz2": {
                    "name": "test-pkg",
                    "version": "not_a_version",
                },
                "pkg-good.tar.bz2": {
                    "name": "test-pkg",
                    "version": "1.0.0",
                },
            },
            "packages.conda": {},
        }
        resp = MagicMock(status_code=200, json=lambda: repodata)
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = worker.check_conda("test-pkg", "conda-forge")
        assert result == "1.0.0"


# ---------------------------------------------------------------------------
# _is_newer
# ---------------------------------------------------------------------------


class TestIsNewer:
    def test_newer(self, worker):
        assert worker._is_newer("1.0.0", "2.0.0") is True

    def test_same(self, worker):
        assert worker._is_newer("1.0.0", "1.0.0") is False

    def test_older(self, worker):
        assert worker._is_newer("2.0.0", "1.0.0") is False

    def test_invalid_version(self, worker):
        assert worker._is_newer("1.0.0", "not.a.version") is False


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRun:
    @patch("grdl_rt.catalog.updater.requests.get")
    def test_run_with_pypi_update(self, mock_get, mock_catalog):
        artifact = _make_artifact(
            name="my-filter",
            version="1.0.0",
            pypi_package="my-filter",
            id=1,
        )
        mock_catalog.list_artifacts.return_value = [artifact]

        resp = MagicMock(status_code=200, json=lambda: {"info": {"version": "2.0.0"}})
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        worker = ArtifactUpdateWorker(catalog=mock_catalog)
        results = worker.run()

        assert len(results) == 1
        r = results[0]
        assert r.update_available is True
        assert r.latest_version == "2.0.0"
        assert r.source == "pypi"
        mock_catalog.update_remote_version.assert_called_once_with(1, "pypi", "2.0.0")

    @patch("grdl_rt.catalog.updater.requests.get")
    def test_run_no_updates(self, mock_get, mock_catalog):
        artifact = _make_artifact(
            name="my-filter",
            version="2.0.0",
            pypi_package="my-filter",
            id=1,
        )
        mock_catalog.list_artifacts.return_value = [artifact]

        resp = MagicMock(status_code=200, json=lambda: {"info": {"version": "2.0.0"}})
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        worker = ArtifactUpdateWorker(catalog=mock_catalog)
        results = worker.run()

        assert len(results) == 1
        assert results[0].update_available is False

    def test_run_empty_catalog(self, mock_catalog):
        mock_catalog.list_artifacts.return_value = []
        worker = ArtifactUpdateWorker(catalog=mock_catalog)
        results = worker.run()
        assert results == []

    @patch("grdl_rt.catalog.updater.requests.get")
    def test_run_with_conda_update(self, mock_get, mock_catalog):
        artifact = _make_artifact(
            name="my-filter",
            version="1.0.0",
            conda_package="my-filter",
            conda_channel="conda-forge",
            id=2,
        )
        mock_catalog.list_artifacts.return_value = [artifact]

        repodata = {
            "packages": {
                "my-filter-2.0.0.tar.bz2": {
                    "name": "my-filter",
                    "version": "2.0.0",
                },
            },
            "packages.conda": {},
        }
        resp = MagicMock(status_code=200, json=lambda: repodata)
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        worker = ArtifactUpdateWorker(catalog=mock_catalog)
        results = worker.run()

        assert len(results) == 1
        r = results[0]
        assert r.source == "conda"
        assert r.update_available is True
        assert r.latest_version == "2.0.0"
        mock_catalog.update_remote_version.assert_called_once_with(2, "conda", "2.0.0")

    @patch("grdl_rt.catalog.updater.requests.get")
    def test_run_conda_query_failed(self, mock_get, mock_catalog):
        import requests

        artifact = _make_artifact(
            name="my-filter",
            version="1.0.0",
            conda_package="my-filter",
            id=3,
        )
        mock_catalog.list_artifacts.return_value = [artifact]
        mock_get.side_effect = requests.RequestException("network down")

        worker = ArtifactUpdateWorker(catalog=mock_catalog)
        results = worker.run()

        assert len(results) == 1
        assert results[0].error == "Conda query failed"
        assert results[0].update_available is False

    @patch("grdl_rt.catalog.updater.requests.get")
    def test_run_conda_default_channel(self, mock_get, mock_catalog):
        """Artifact with conda_package but no conda_channel uses conda-forge."""
        artifact = _make_artifact(
            name="my-filter",
            version="1.0.0",
            conda_package="my-filter",
            conda_channel=None,
            id=4,
        )
        mock_catalog.list_artifacts.return_value = [artifact]

        repodata = {"packages": {}, "packages.conda": {}}
        resp = MagicMock(status_code=200, json=lambda: repodata)
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        worker = ArtifactUpdateWorker(catalog=mock_catalog)
        results = worker.run()

        assert len(results) == 1
        assert results[0].source == "conda"
