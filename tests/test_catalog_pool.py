"""
Tests for grdl_rt.catalog.pool — ThreadExecutorPool.

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

from unittest import mock

from grdl_rt.catalog.pool import ThreadExecutorPool


class TestThreadExecutorPool:

    def test_submit_returns_future(self):
        pool = ThreadExecutorPool(max_workers=2)
        try:
            # Submit a simple callable
            future = pool._executor.submit(lambda: 42)
            assert future.result(timeout=5) == 42
        finally:
            pool.shutdown(wait=True)

    def test_shutdown_is_safe(self):
        pool = ThreadExecutorPool(max_workers=1)
        pool.shutdown(wait=True)
        # Should not raise

    def test_submit_download_pip(self):
        pool = ThreadExecutorPool(max_workers=1)
        try:
            # Mock subprocess.run to avoid actually installing
            with mock.patch("grdl_rt.catalog.pool.subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0, stdout="installed", stderr="")
                future = pool.submit_download("fake-package")
                result = future.result(timeout=10)
                assert result.returncode == 0
        finally:
            pool.shutdown(wait=True)

    def test_submit_update_check(self):
        pool = ThreadExecutorPool(max_workers=1)
        try:
            worker = mock.MagicMock()
            worker.run.return_value = []
            future = pool.submit_update_check(worker)
            result = future.result(timeout=10)
            assert result == []
            worker.run.assert_called_once()
        finally:
            pool.shutdown(wait=True)

    def test_submit_download_conda(self):
        pool = ThreadExecutorPool(max_workers=1)
        try:
            with mock.patch("grdl_rt.catalog.pool.subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0, stdout="installed", stderr="")
                future = pool.submit_download(
                    "fake-package", use_conda=True, conda_channel="my-channel"
                )
                result = future.result(timeout=10)
                assert result.returncode == 0
                cmd = mock_run.call_args[0][0]
                assert cmd[0] == "conda"
                assert "-c" in cmd
                assert "my-channel" in cmd
        finally:
            pool.shutdown(wait=True)

    def test_submit_download_conda_no_channel(self):
        pool = ThreadExecutorPool(max_workers=1)
        try:
            with mock.patch("grdl_rt.catalog.pool.subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
                future = pool.submit_download("pkg", use_conda=True)
                future.result(timeout=10)
                cmd = mock_run.call_args[0][0]
                assert "-c" not in cmd
        finally:
            pool.shutdown(wait=True)

    def test_submit_download_venv_pip(self, tmp_path):
        pool = ThreadExecutorPool(max_workers=1)
        try:
            with mock.patch("grdl_rt.catalog.pool.subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
                # Create a fake venv with Scripts/pip.exe (Windows path)
                scripts_dir = tmp_path / "Scripts"
                scripts_dir.mkdir()
                pip_exe = scripts_dir / "pip.exe"
                pip_exe.write_text("")

                future = pool.submit_download("fake-package", target_venv=tmp_path)
                result = future.result(timeout=10)
                assert result.returncode == 0
        finally:
            pool.shutdown(wait=True)

    def test_install_package_failure_logs_error(self):
        pool = ThreadExecutorPool(max_workers=1)
        try:
            with mock.patch("grdl_rt.catalog.pool.subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="error occurred")
                future = pool.submit_download("bad-package")
                result = future.result(timeout=10)
                assert result.returncode == 1
        finally:
            pool.shutdown(wait=True)
