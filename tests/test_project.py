# -*- coding: utf-8 -*-
"""
Tests for grdl_rt.execution.project — GrdkProject model.

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

import json
import numpy as np
import pytest

from grdl_rt.execution.project import GrdkProject
from grdl_rt.execution.tags import ProjectTags
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition


@pytest.fixture
def tmp_project_dir(tmp_path):
    """Provide a temporary directory for project creation."""
    return tmp_path / "test_project"


class TestGrdkProjectCreate:

    def test_create_new_project(self, tmp_project_dir):
        project = GrdkProject.create(tmp_project_dir, name="Test Project")
        assert project.name == "Test Project"
        assert tmp_project_dir.exists()
        assert (tmp_project_dir / "project.json").exists()
        assert (tmp_project_dir / "images").is_dir()
        assert (tmp_project_dir / "chips").is_dir()
        assert (tmp_project_dir / "workflows").is_dir()
        assert (tmp_project_dir / "cache").is_dir()

    def test_create_with_tags(self, tmp_project_dir):
        tags = ProjectTags(intended_target="vehicle")
        project = GrdkProject.create(
            tmp_project_dir, name="Tagged", tags=tags
        )
        assert project.tags.intended_target == "vehicle"

    def test_create_existing_raises(self, tmp_project_dir):
        GrdkProject.create(tmp_project_dir, name="First")
        with pytest.raises(FileExistsError):
            GrdkProject.create(tmp_project_dir, name="Second")


class TestGrdkProjectLoadSave:

    def test_save_and_load_roundtrip(self, tmp_project_dir):
        project = GrdkProject.create(
            tmp_project_dir, name="Roundtrip",
            tags=ProjectTags(intended_target="ship"),
        )
        project.add_image("/path/to/image1.tif")
        project.add_image("/path/to/image2.ntf")

        wf = WorkflowDefinition(name="MyWorkflow", version="1.0.0")
        wf.add_step(ProcessingStep("Filter", "0.1.0", params={'k': 3}))
        project.add_workflow(wf)
        project.save()

        loaded = GrdkProject.load(tmp_project_dir)
        assert loaded.name == "Roundtrip"
        assert loaded.tags.intended_target == "ship"
        assert len(loaded.image_paths) == 2
        assert "/path/to/image1.tif" in loaded.image_paths
        assert "MyWorkflow" in loaded.workflows
        assert loaded.workflows["MyWorkflow"].steps[0].params == {'k': 3}

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GrdkProject.load(tmp_path / "nonexistent")


class TestGrdkProjectImages:

    def test_add_image(self, tmp_project_dir):
        project = GrdkProject.create(tmp_project_dir, name="Images")
        project.add_image("/data/img1.tif")
        project.add_image("/data/img2.tif")
        assert len(project.image_paths) == 2

    def test_add_duplicate_image_ignored(self, tmp_project_dir):
        project = GrdkProject.create(tmp_project_dir, name="Dupes")
        project.add_image("/data/img1.tif")
        project.add_image("/data/img1.tif")
        assert len(project.image_paths) == 1


class TestGrdkProjectChips:

    def test_save_and_load_chips(self, tmp_project_dir):
        project = GrdkProject.create(tmp_project_dir, name="Chips")
        chips = [
            np.random.default_rng(0).random((32, 32)),
            np.random.default_rng(1).random((32, 32)),
            np.random.default_rng(2).random((32, 32)),
        ]
        labels = ["positive", "negative", "unknown"]

        project.save_chips("region_01", chips, labels)
        loaded_chips, loaded_labels = project.load_chips("region_01")

        assert len(loaded_chips) == 3
        assert loaded_labels == labels
        np.testing.assert_array_equal(loaded_chips[0], chips[0])
        np.testing.assert_array_equal(loaded_chips[2], chips[2])


class TestGrdkProjectTimestamps:

    def test_created_at_set(self, tmp_project_dir):
        project = GrdkProject.create(tmp_project_dir, name="Timestamps")
        assert project.created_at is not None
        assert 'T' in project.created_at  # ISO 8601

    def test_modified_at_updates_on_save(self, tmp_project_dir):
        project = GrdkProject.create(tmp_project_dir, name="Modified")
        first_modified = project.modified_at
        project.add_image("/data/new.tif")
        project.save()
        assert project.modified_at >= first_modified
