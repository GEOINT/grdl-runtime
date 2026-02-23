"""
Project Model - GRDK project management (create, load, save).

A GRDK project is a directory containing imagery references, extracted
chips with labels, workflow definitions, and metadata. Projects support
flexible entry points -- a user can open a project at any stage.

Dependencies
------------
pyyaml

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
import os
from datetime import UTC, datetime
from pathlib import Path

# grdl-runtime internal
from grdl_rt.execution.context import get_logger

logger = get_logger(__name__)

# Third-party
import numpy as np

# grdl-runtime internal
from grdl_rt.execution.tags import ProjectTags
from grdl_rt.execution.workflow import WorkflowDefinition


class GrdkProject:
    """A GRDK project directory containing all project artifacts.

    Project directory layout::

        my_project/
        ├── project.json      # Manifest
        ├── images/           # Symlinks or copies of source imagery
        ├── chips/            # Extracted chip arrays (.npy) + labels.json
        ├── workflows/        # definition.yaml + definition.py per workflow
        └── cache/            # GPU preview cache, thumbnails

    Parameters
    ----------
    project_dir : Path
        Root directory of the project.
    name : str
        Human-readable project name.
    tags : Optional[ProjectTags]
        Project-level tags.
    """

    MANIFEST_FILENAME = "project.json"

    def __init__(
        self,
        project_dir: Path,
        name: str = "",
        tags: ProjectTags | None = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.name = name
        self.tags = tags or ProjectTags()
        self.image_paths: list[str] = []
        self.workflows: dict[str, WorkflowDefinition] = {}
        self._created_at: str = datetime.now(UTC).isoformat()
        self._modified_at: str = self._created_at

    @classmethod
    def create(
        cls,
        project_dir: Path,
        name: str,
        tags: ProjectTags | None = None,
    ) -> "GrdkProject":
        """Create a new GRDK project directory and manifest.

        Parameters
        ----------
        project_dir : Path
            Directory to create the project in. Will be created if it
            does not exist.
        name : str
            Human-readable project name.
        tags : Optional[ProjectTags]
            Project-level tags.

        Returns
        -------
        GrdkProject
            The newly created project.

        Raises
        ------
        FileExistsError
            If a project.json already exists in the directory.
        """
        project_dir = Path(project_dir)
        manifest_path = project_dir / cls.MANIFEST_FILENAME

        if manifest_path.exists():
            raise FileExistsError(f"Project already exists at {project_dir}")

        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "images").mkdir(exist_ok=True)
        (project_dir / "chips").mkdir(exist_ok=True)
        (project_dir / "workflows").mkdir(exist_ok=True)
        (project_dir / "cache").mkdir(exist_ok=True)

        project = cls(project_dir, name=name, tags=tags)
        project.save()
        return project

    @classmethod
    def load(cls, project_dir: Path) -> "GrdkProject":
        """Load an existing GRDK project from disk.

        Parameters
        ----------
        project_dir : Path
            Root directory of the project.

        Returns
        -------
        GrdkProject

        Raises
        ------
        FileNotFoundError
            If project.json does not exist.
        """
        project_dir = Path(project_dir)
        manifest_path = project_dir / cls.MANIFEST_FILENAME

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No project found at {project_dir} " f"(missing {cls.MANIFEST_FILENAME})"
            )

        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)

        project = cls(
            project_dir,
            name=data.get("name", ""),
            tags=ProjectTags.from_dict(data.get("tags", {})),
        )
        project.image_paths = data.get("image_paths", [])
        project._created_at = data.get("created_at", project._created_at)
        project._modified_at = data.get("modified_at", project._modified_at)

        # Load workflow definitions
        for wf_name, wf_data in data.get("workflows", {}).items():
            project.workflows[wf_name] = WorkflowDefinition.from_dict(wf_data)

        return project

    def save(self) -> None:
        """Save the project manifest to disk.

        Writes project.json with current state. Chip arrays and
        workflow files are saved separately.
        """
        self._modified_at = datetime.now(UTC).isoformat()

        manifest = {
            "name": self.name,
            "version": "1.0",
            "created_at": self._created_at,
            "modified_at": self._modified_at,
            "tags": self.tags.to_dict(),
            "image_paths": self.image_paths,
            "workflows": {name: wf.to_dict() for name, wf in self.workflows.items()},
        }

        manifest_path = self.project_dir / self.MANIFEST_FILENAME
        tmp_path = manifest_path.with_suffix(".json.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            os.replace(str(tmp_path), str(manifest_path))
        except Exception:
            logger.error("Failed to save project manifest", path=str(manifest_path))
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def add_image(self, image_path: str) -> None:
        """Add an image path reference to the project.

        Parameters
        ----------
        image_path : str
            Path to the image file (absolute or relative).
        """
        if image_path not in self.image_paths:
            self.image_paths.append(image_path)

    def add_workflow(self, workflow: WorkflowDefinition) -> None:
        """Add or update a workflow definition in the project.

        Parameters
        ----------
        workflow : WorkflowDefinition
        """
        self.workflows[workflow.name] = workflow

    def save_chips(
        self,
        chip_name: str,
        chips: list[np.ndarray],
        labels: list[str],
    ) -> None:
        """Save a set of chips and labels to the chips/ directory.

        Parameters
        ----------
        chip_name : str
            Name for this chip collection (used as subdirectory).
        chips : List[np.ndarray]
            Chip image arrays.
        labels : List[str]
            Labels for each chip ("positive", "negative", "unknown").
        """
        chip_dir = self.project_dir / "chips" / chip_name
        chip_dir.mkdir(parents=True, exist_ok=True)

        for i, chip in enumerate(chips):
            np.save(chip_dir / f"chip_{i:04d}.npy", chip)

        labels_data = {
            "count": len(labels),
            "labels": labels,
        }
        labels_path = chip_dir / "labels.json"
        labels_tmp = labels_path.with_suffix(".json.tmp")
        try:
            with open(labels_tmp, "w", encoding="utf-8") as f:
                json.dump(labels_data, f, indent=2)
            os.replace(str(labels_tmp), str(labels_path))
        except Exception:
            logger.error("Failed to save chip labels", path=str(labels_path))
            if labels_tmp.exists():
                labels_tmp.unlink()
            raise

    def load_chips(
        self,
        chip_name: str,
    ) -> tuple:
        """Load chips and labels from the chips/ directory.

        Parameters
        ----------
        chip_name : str
            Name of the chip collection.

        Returns
        -------
        tuple
            (chips: List[np.ndarray], labels: List[str])
        """
        chip_dir = self.project_dir / "chips" / chip_name

        with open(chip_dir / "labels.json", encoding="utf-8") as f:
            labels_data = json.load(f)

        chips = []
        for i in range(labels_data["count"]):
            chip_path = chip_dir / f"chip_{i:04d}.npy"
            chips.append(np.load(chip_path))

        return chips, labels_data["labels"]

    @property
    def created_at(self) -> str:
        """ISO 8601 creation timestamp."""
        return self._created_at

    @property
    def modified_at(self) -> str:
        """ISO 8601 last modification timestamp."""
        return self._modified_at
