"""
ThreadExecutorPool - Thread pool for background catalog operations.

Provides a managed thread pool for running update checks and package
downloads in the background without blocking the UI.

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
2026-02-06

Modified
--------
2026-02-06
"""

# Standard library
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

# grdl-runtime internal
from grdl_rt.execution.context import get_logger

logger = get_logger(__name__)

# grdl-runtime internal
from grdl_rt.catalog.updater import ArtifactUpdateWorker


class ThreadExecutorPool:
    """Manages a pool of worker threads for background catalog operations.

    Parameters
    ----------
    max_workers : int
        Maximum number of concurrent threads. Default 4.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit_update_check(
        self,
        worker: ArtifactUpdateWorker,
    ) -> Future:
        """Submit an update check job to run in the background.

        Parameters
        ----------
        worker : ArtifactUpdateWorker
            The update worker to run.

        Returns
        -------
        Future
            Future resolving to List[UpdateResult].
        """
        return self._executor.submit(worker.run)

    def submit_download(
        self,
        package: str,
        target_venv: Path | None = None,
        use_conda: bool = False,
        conda_channel: str | None = None,
    ) -> Future:
        """Submit a package download/install job.

        Parameters
        ----------
        package : str
            Package name to install.
        target_venv : Optional[Path]
            Path to a virtualenv to install into. If None, installs
            to the current environment.
        use_conda : bool
            If True, use conda instead of pip.
        conda_channel : Optional[str]
            Conda channel to install from.

        Returns
        -------
        Future
            Future resolving to subprocess.CompletedProcess.
        """
        return self._executor.submit(
            self._install_package,
            package,
            target_venv,
            use_conda,
            conda_channel,
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the thread pool.

        Parameters
        ----------
        wait : bool
            If True, wait for running tasks to complete.
        """
        self._executor.shutdown(wait=wait)

    @staticmethod
    def _install_package(
        package: str,
        target_venv: Path | None,
        use_conda: bool,
        conda_channel: str | None,
    ) -> subprocess.CompletedProcess:
        """Install a package using pip or conda.

        Parameters
        ----------
        package : str
        target_venv : Optional[Path]
        use_conda : bool
        conda_channel : Optional[str]

        Returns
        -------
        subprocess.CompletedProcess
        """
        if use_conda:
            cmd = ["conda", "install", "-y"]
            if conda_channel:
                cmd.extend(["-c", conda_channel])
            cmd.append(package)
        else:
            if target_venv:
                pip_path = target_venv / "bin" / "pip"
                if not pip_path.exists():
                    pip_path = target_venv / "Scripts" / "pip.exe"
                cmd = [str(pip_path), "install", package]
            else:
                cmd = [sys.executable, "-m", "pip", "install", package]

        logger.info("Installing package", package=package, cmd=" ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error("Install failed", package=package, stderr=result.stderr)
        else:
            logger.info("Successfully installed package", package=package)
        return result
