# -*- coding: utf-8 -*-
"""
Artifact Update Worker - Check PyPI/conda for updates to known artifacts.

Provides a runnable worker class that queries PyPI and conda for the
latest versions of known GRDL artifacts and reports available updates.

Dependencies
------------
requests
packaging

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
import json
from typing import List, Optional

# Third-party
import requests
from packaging.version import Version, InvalidVersion

# grdl-runtime internal
from grdl_rt.execution.context import get_logger

logger = get_logger(__name__)

# grdl-runtime internal
from grdl_rt.catalog.base import ArtifactCatalogBase
from grdl_rt.catalog.models import Artifact, UpdateResult


_PYPI_URL = "https://pypi.org/pypi/{package}/json"
_CONDA_URL = "https://conda.anaconda.org/{channel}/{platform}/repodata.json"


class ArtifactUpdateWorker:
    """Runnable worker that checks PyPI/conda for artifact updates.

    Parameters
    ----------
    catalog : ArtifactCatalogBase
        The catalog backend to check.
    timeout : float
        HTTP request timeout in seconds. Default 10.0.
    """

    def __init__(
        self,
        catalog: ArtifactCatalogBase,
        timeout: float = 10.0,
    ) -> None:
        self._catalog = catalog
        self._timeout = timeout

    def check_pypi(self, package_name: str) -> Optional[str]:
        """Query PyPI JSON API for the latest version of a package.

        Parameters
        ----------
        package_name : str
            PyPI package name.

        Returns
        -------
        Optional[str]
            Latest version string, or None if the query fails.
        """
        url = _PYPI_URL.format(package=package_name)
        try:
            resp = requests.get(url, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            return data['info']['version']
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            logger.warning("PyPI check failed", package=package_name, error=str(e))
            return None

    def check_conda(
        self,
        package_name: str,
        channel: str = "conda-forge",
    ) -> Optional[str]:
        """Query conda repodata for the latest version of a package.

        Parameters
        ----------
        package_name : str
            Conda package name.
        channel : str
            Conda channel. Default 'conda-forge'.

        Returns
        -------
        Optional[str]
            Latest version string, or None if the query fails.
        """
        # Query the noarch repodata first, then platform-specific
        for platform in ('noarch', 'win-64', 'linux-64'):
            url = _CONDA_URL.format(channel=channel, platform=platform)
            try:
                resp = requests.get(url, timeout=self._timeout)
                resp.raise_for_status()
                data = resp.json()
                packages = data.get('packages', {})
                packages.update(data.get('packages.conda', {}))

                latest: Optional[Version] = None
                for pkg_info in packages.values():
                    if pkg_info.get('name') == package_name:
                        try:
                            v = Version(pkg_info['version'])
                            if latest is None or v > latest:
                                latest = v
                        except InvalidVersion:
                            continue

                if latest is not None:
                    return str(latest)
            except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
                logger.warning(
                    "Conda check failed",
                    package=package_name,
                    channel=channel,
                    platform=platform,
                    error=str(e),
                )
                continue

        return None

    def _is_newer(self, current: str, latest: str) -> bool:
        """Compare version strings.

        Parameters
        ----------
        current : str
        latest : str

        Returns
        -------
        bool
            True if latest is newer than current.
        """
        try:
            return Version(latest) > Version(current)
        except InvalidVersion:
            return False

    def run(self) -> List[UpdateResult]:
        """Check all known artifacts for updates.

        Queries PyPI and/or conda for each artifact that has a package
        location configured.

        Returns
        -------
        List[UpdateResult]
            Results for each artifact checked.
        """
        results: List[UpdateResult] = []
        artifacts = self._catalog.list_artifacts()

        for artifact in artifacts:
            if artifact.pypi_package:
                latest = self.check_pypi(artifact.pypi_package)
                update_available = False
                if latest:
                    update_available = self._is_newer(
                        artifact.version, latest
                    )
                    if artifact.id is not None:
                        self._catalog.update_remote_version(
                            artifact.id, 'pypi', latest
                        )
                results.append(UpdateResult(
                    artifact=artifact,
                    source='pypi',
                    current_version=artifact.version,
                    latest_version=latest,
                    update_available=update_available,
                    error=None if latest else 'PyPI query failed',
                ))

            if artifact.conda_package:
                channel = artifact.conda_channel or 'conda-forge'
                latest = self.check_conda(
                    artifact.conda_package, channel
                )
                update_available = False
                if latest:
                    update_available = self._is_newer(
                        artifact.version, latest
                    )
                    if artifact.id is not None:
                        self._catalog.update_remote_version(
                            artifact.id, 'conda', latest
                        )
                results.append(UpdateResult(
                    artifact=artifact,
                    source='conda',
                    current_version=artifact.version,
                    latest_version=latest,
                    update_available=update_available,
                    error=None if latest else 'Conda query failed',
                ))

        return results
