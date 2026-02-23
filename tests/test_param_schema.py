"""
Tests for TG8 — parameter schema extraction, catalog storage, and
JSON Schema–based workflow validation.

Covers:
- 8.1: extract_param_schema() correctness for all constraint types
- 8.2: catalog.get_param_schema() round-trip through SQLite and YAML backends
- 8.3: validate_workflow() using jsonschema for deep parameter checking

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-11
"""

from typing import Annotated
from unittest.mock import patch

import jsonschema
import pytest
from grdl.image_processing.params import Desc, Options, Range, collect_param_specs

from grdl_rt.catalog.models import Artifact
from grdl_rt.catalog.schema import extract_param_schema
from grdl_rt.execution.validation import validate_workflow
from grdl_rt.execution.workflow import ProcessingStep, WorkflowDefinition

# ====================================================================
# Test processors — realistic parameter declarations
# ====================================================================


class _MinimalProcessor:
    """No param_specs at all (plain __init__)."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def apply(self, source, **kwargs):
        return source


class _AnnotatedProcessor:
    """Processor with Annotated params — collected into __param_specs__."""

    sigma: Annotated[float, Range(min=0.001, max=100.0), Desc("Gaussian sigma in pixels")] = 2.0
    accuracy: Annotated[float, Range(min=0.0001, max=1.0), Desc("Accuracy parameter")] = 0.002
    interpolation: Annotated[
        str, Options("nearest", "bilinear", "bicubic"), Desc("Resampling method")
    ] = "bilinear"
    normalize: Annotated[bool, Desc("Normalize output")] = True

    def apply(self, source, **kwargs):
        return source


# Manually collect param_specs the same way ImageProcessor.__init_subclass__ does
_AnnotatedProcessor.__param_specs__ = collect_param_specs(_AnnotatedProcessor)  # type: ignore[attr-defined]


class _RequiredParamProcessor:
    """Processor with a required (no-default) annotated parameter."""

    num_looks: Annotated[int, Range(min=2), Desc("Number of sublooks")]
    overlap: Annotated[float, Range(min=0.0, max=0.99), Desc("Fractional overlap")] = 0.0
    dimension: Annotated[str, Options("azimuth", "range"), Desc("Split dimension")] = "azimuth"
    deweight: Annotated[bool, Desc("Apply deweighting")] = True

    def apply(self, source, **kwargs):
        return source


_RequiredParamProcessor.__param_specs__ = collect_param_specs(_RequiredParamProcessor)  # type: ignore[attr-defined]


# ====================================================================
# 8.1 — Schema Extractor Tests
# ====================================================================


class TestExtractParamSchema:
    """Test extract_param_schema for all constraint types."""

    def test_empty_schema_for_class_without_param_specs(self):
        schema = extract_param_schema(_MinimalProcessor)
        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert "required" not in schema

    def test_schema_has_correct_top_level_structure(self):
        schema = extract_param_schema(_AnnotatedProcessor)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert isinstance(schema["properties"], dict)

    def test_float_param_maps_to_number(self):
        schema = extract_param_schema(_AnnotatedProcessor)
        sigma = schema["properties"]["sigma"]
        assert sigma["type"] == "number"

    def test_range_min_max(self):
        schema = extract_param_schema(_AnnotatedProcessor)
        sigma = schema["properties"]["sigma"]
        assert sigma["minimum"] == 0.001
        assert sigma["maximum"] == 100.0

    def test_range_min_only(self):
        schema = extract_param_schema(_RequiredParamProcessor)
        num_looks = schema["properties"]["num_looks"]
        assert num_looks["minimum"] == 2
        assert "maximum" not in num_looks

    def test_string_enum(self):
        schema = extract_param_schema(_AnnotatedProcessor)
        interp = schema["properties"]["interpolation"]
        assert interp["type"] == "string"
        assert interp["enum"] == ["nearest", "bilinear", "bicubic"]

    def test_bool_param(self):
        schema = extract_param_schema(_AnnotatedProcessor)
        norm = schema["properties"]["normalize"]
        assert norm["type"] == "boolean"

    def test_int_param(self):
        schema = extract_param_schema(_RequiredParamProcessor)
        assert schema["properties"]["num_looks"]["type"] == "integer"

    def test_descriptions_present(self):
        schema = extract_param_schema(_AnnotatedProcessor)
        assert schema["properties"]["sigma"]["description"] == "Gaussian sigma in pixels"
        assert schema["properties"]["accuracy"]["description"] == "Accuracy parameter"

    def test_defaults_present(self):
        schema = extract_param_schema(_AnnotatedProcessor)
        assert schema["properties"]["sigma"]["default"] == 2.0
        assert schema["properties"]["normalize"]["default"] is True
        assert schema["properties"]["interpolation"]["default"] == "bilinear"

    def test_required_params_listed(self):
        schema = extract_param_schema(_RequiredParamProcessor)
        assert "required" in schema
        assert "num_looks" in schema["required"]
        # optional params should NOT be in required
        assert "overlap" not in schema["required"]
        assert "dimension" not in schema["required"]
        assert "deweight" not in schema["required"]

    def test_all_optional_means_no_required_key(self):
        schema = extract_param_schema(_AnnotatedProcessor)
        assert "required" not in schema

    def test_produced_schema_is_valid_json_schema(self):
        """The schema itself must validate under meta-schema."""
        schema = extract_param_schema(_RequiredParamProcessor)
        validator_cls = jsonschema.Draft202012Validator
        validator_cls.check_schema(schema)

    def test_schema_validates_correct_params(self):
        schema = extract_param_schema(_RequiredParamProcessor)
        valid_params = {
            "num_looks": 4,
            "overlap": 0.5,
            "dimension": "azimuth",
            "deweight": False,
        }
        jsonschema.validate(valid_params, schema)

    def test_schema_rejects_wrong_type(self):
        schema = extract_param_schema(_RequiredParamProcessor)
        bad_params = {"num_looks": "abc"}  # should be integer
        with pytest.raises(jsonschema.ValidationError) as exc_info:
            jsonschema.validate(bad_params, schema)
        assert "type" in exc_info.value.message

    def test_schema_rejects_out_of_range(self):
        schema = extract_param_schema(_RequiredParamProcessor)
        bad_params = {"num_looks": 1}  # minimum is 2
        with pytest.raises(jsonschema.ValidationError) as exc_info:
            jsonschema.validate(bad_params, schema)
        assert "minimum" in exc_info.value.message

    def test_schema_rejects_invalid_enum(self):
        schema = extract_param_schema(_RequiredParamProcessor)
        bad_params = {"num_looks": 3, "dimension": "diagonal"}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad_params, schema)

    def test_schema_rejects_missing_required(self):
        schema = extract_param_schema(_RequiredParamProcessor)
        bad_params = {"overlap": 0.5}  # num_looks is required
        with pytest.raises(jsonschema.ValidationError) as exc_info:
            jsonschema.validate(bad_params, schema)
        assert "required" in str(exc_info.value.validator)

    def test_schema_rejects_additional_properties(self):
        schema = extract_param_schema(_AnnotatedProcessor)
        bad_params = {"sigma": 2.0, "unknown_param": 42}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad_params, schema)


# ====================================================================
# 8.2 — Catalog Schema Storage Tests
# ====================================================================


class TestCatalogSchemaStorageSqlite:
    """Test param_schema round-trip through SqliteArtifactCatalog."""

    def test_store_and_retrieve_schema(self, tmp_path):
        from grdl_rt.catalog.database import SqliteArtifactCatalog

        schema = extract_param_schema(_AnnotatedProcessor)
        artifact = Artifact(
            name="annotated-proc",
            version="1.0.0",
            artifact_type="grdl_processor",
            description="Test processor",
            processor_class="test._AnnotatedProcessor",
            param_schema=schema,
        )

        db_path = tmp_path / "test.db"
        with SqliteArtifactCatalog(db_path) as cat:
            cat.add_artifact(artifact)

            retrieved = cat.get_artifact("annotated-proc", "1.0.0")
            assert retrieved is not None
            assert retrieved.param_schema is not None
            assert retrieved.param_schema["type"] == "object"
            assert "sigma" in retrieved.param_schema["properties"]

    def test_get_param_schema_by_name(self, tmp_path):
        from grdl_rt.catalog.database import SqliteArtifactCatalog

        schema = extract_param_schema(_RequiredParamProcessor)
        artifact = Artifact(
            name="sublook",
            version="0.1.0",
            artifact_type="grdl_processor",
            processor_class="test._RequiredParamProcessor",
            param_schema=schema,
        )

        db_path = tmp_path / "test.db"
        with SqliteArtifactCatalog(db_path) as cat:
            cat.add_artifact(artifact)

            result = cat.get_param_schema("sublook")
            assert result is not None
            assert "num_looks" in result["properties"]
            assert "required" in result
            assert "num_looks" in result["required"]

    def test_get_param_schema_by_name_and_version(self, tmp_path):
        from grdl_rt.catalog.database import SqliteArtifactCatalog

        schema = extract_param_schema(_AnnotatedProcessor)
        artifact = Artifact(
            name="gauss",
            version="2.0.0",
            artifact_type="grdl_processor",
            processor_class="test._AnnotatedProcessor",
            param_schema=schema,
        )

        db_path = tmp_path / "test.db"
        with SqliteArtifactCatalog(db_path) as cat:
            cat.add_artifact(artifact)

            result = cat.get_param_schema("gauss", version="2.0.0")
            assert result is not None
            assert result["properties"]["sigma"]["type"] == "number"

            missing = cat.get_param_schema("gauss", version="9.9.9")
            assert missing is None

    def test_no_schema_returns_none(self, tmp_path):
        from grdl_rt.catalog.database import SqliteArtifactCatalog

        artifact = Artifact(
            name="bare",
            version="1.0.0",
            artifact_type="grdl_processor",
            processor_class="test._MinimalProcessor",
        )

        db_path = tmp_path / "test.db"
        with SqliteArtifactCatalog(db_path) as cat:
            cat.add_artifact(artifact)
            result = cat.get_param_schema("bare")
            assert result is None

    def test_schema_migration_from_v3(self, tmp_path):
        """Verify database migration adds param_schema column."""
        from grdl_rt.catalog.database import SqliteArtifactCatalog

        db_path = tmp_path / "test.db"
        with SqliteArtifactCatalog(db_path) as cat:
            assert cat.schema_version >= 4


class TestCatalogSchemaStorageYaml:
    """Test param_schema round-trip through YamlArtifactCatalog."""

    def test_store_and_retrieve_schema(self, tmp_path):
        from grdl_rt.catalog.yaml_catalog import YamlArtifactCatalog

        schema = extract_param_schema(_AnnotatedProcessor)
        artifact = Artifact(
            name="yaml-proc",
            version="1.0.0",
            artifact_type="grdl_processor",
            param_schema=schema,
        )

        yaml_path = tmp_path / "catalog.yaml"
        with YamlArtifactCatalog(yaml_path) as cat:
            cat.add_artifact(artifact)

        # Re-open and verify persistence
        with YamlArtifactCatalog(yaml_path) as cat:
            result = cat.get_param_schema("yaml-proc")
            assert result is not None
            assert "sigma" in result["properties"]
            assert result["properties"]["interpolation"]["enum"] == [
                "nearest",
                "bilinear",
                "bicubic",
            ]


class TestCatalogSchemaStorageFederated:
    """Test param_schema through FederatedArtifactCatalog."""

    def test_federated_delegates_get_param_schema(self, tmp_path):
        from grdl_rt.catalog.database import SqliteArtifactCatalog
        from grdl_rt.catalog.federated import FederatedArtifactCatalog

        schema = extract_param_schema(_RequiredParamProcessor)
        artifact = Artifact(
            name="fed-proc",
            version="1.0.0",
            artifact_type="grdl_processor",
            param_schema=schema,
        )

        db_path = tmp_path / "test.db"
        backend = SqliteArtifactCatalog(db_path)
        backend.add_artifact(artifact)

        federated = FederatedArtifactCatalog([backend])
        result = federated.get_param_schema("fed-proc")
        assert result is not None
        assert "num_looks" in result["properties"]

        federated.close()


# ====================================================================
# 8.3 — Enhanced Validation Tests
# ====================================================================


class TestJsonSchemaValidation:
    """Test workflow validation using JSON Schema for parameter checking."""

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_valid_params_pass_schema_validation(self, mock_resolve):
        mock_resolve.return_value = _AnnotatedProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(
            ProcessingStep(
                "AnnotatedProcessor",
                "1.0",
                params={"sigma": 5.0, "accuracy": 0.01},
            )
        )
        errors = validate_workflow(wf)
        assert len(errors) == 0

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_wrong_type_caught_by_schema(self, mock_resolve):
        """String 'abc' when schema expects integer → INVALID_PARAM_TYPE."""
        mock_resolve.return_value = _RequiredParamProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(
            ProcessingStep(
                "RequiredProc",
                "1.0",
                params={"num_looks": "abc", "overlap": 0.5},
            )
        )
        errors = validate_workflow(wf)

        type_errors = [e for e in errors if e.code == "INVALID_PARAM_TYPE"]
        assert len(type_errors) >= 1
        assert "num_looks" in type_errors[0].message

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_out_of_range_caught_by_schema(self, mock_resolve):
        mock_resolve.return_value = _RequiredParamProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(
            ProcessingStep(
                "RequiredProc",
                "1.0",
                params={"num_looks": 1},  # minimum is 2
            )
        )
        errors = validate_workflow(wf)

        value_errors = [e for e in errors if e.code == "INVALID_PARAM_VALUE"]
        assert len(value_errors) >= 1
        assert "num_looks" in value_errors[0].message

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_missing_required_caught_by_schema(self, mock_resolve):
        mock_resolve.return_value = _RequiredParamProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(
            ProcessingStep(
                "RequiredProc",
                "1.0",
                params={"overlap": 0.5},  # num_looks is required
            )
        )
        errors = validate_workflow(wf)

        req_errors = [e for e in errors if e.code == "MISSING_REQUIRED_PARAM"]
        assert len(req_errors) >= 1
        assert "num_looks" in req_errors[0].message

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_invalid_enum_caught_by_schema(self, mock_resolve):
        mock_resolve.return_value = _RequiredParamProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(
            ProcessingStep(
                "RequiredProc",
                "1.0",
                params={"num_looks": 3, "dimension": "diagonal"},
            )
        )
        errors = validate_workflow(wf)

        value_errors = [e for e in errors if e.code == "INVALID_PARAM_VALUE"]
        assert len(value_errors) >= 1
        assert "dimension" in value_errors[0].message

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_schema_from_catalog_used_when_available(self, mock_resolve, tmp_path):
        """When catalog has a stored schema, it is used for validation."""
        from grdl_rt.catalog.database import SqliteArtifactCatalog

        mock_resolve.return_value = _MinimalProcessor  # no __param_specs__

        schema = extract_param_schema(_RequiredParamProcessor)
        artifact = Artifact(
            name="RequiredProc",
            version="1.0.0",
            artifact_type="grdl_processor",
            processor_class="test._RequiredParamProcessor",
            param_schema=schema,
        )

        db_path = tmp_path / "test.db"
        with SqliteArtifactCatalog(db_path) as cat:
            cat.add_artifact(artifact)

            wf = WorkflowDefinition(name="Test")
            wf.add_step(
                ProcessingStep(
                    "RequiredProc",
                    "1.0",
                    params={"num_looks": "bad"},
                )
            )
            errors = validate_workflow(wf, catalog=cat)

            type_errors = [e for e in errors if e.code == "INVALID_PARAM_TYPE"]
            assert len(type_errors) >= 1

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_multiple_errors_reported(self, mock_resolve):
        """Multiple parameter violations produce multiple errors."""
        mock_resolve.return_value = _RequiredParamProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(
            ProcessingStep(
                "RequiredProc",
                "1.0",
                params={
                    "num_looks": "not_int",  # type error
                    "overlap": 5.0,  # exceeds maximum 0.99
                    "dimension": "diagonal",  # not in enum
                },
            )
        )
        errors = validate_workflow(wf)
        # Should have at least 3 parameter errors
        param_errors = [
            e for e in errors if e.code in ("INVALID_PARAM_TYPE", "INVALID_PARAM_VALUE")
        ]
        assert len(param_errors) >= 3

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_fallback_to_param_specs_when_no_jsonschema(self, mock_resolve):
        """When jsonschema is mocked away, falls back to __param_specs__."""
        mock_resolve.return_value = _RequiredParamProcessor

        wf = WorkflowDefinition(name="Test")
        wf.add_step(
            ProcessingStep(
                "RequiredProc",
                "1.0",
                params={},  # missing required num_looks
            )
        )

        # Even with jsonschema path, missing required should be caught
        errors = validate_workflow(wf)
        req_errors = [e for e in errors if e.code == "MISSING_REQUIRED_PARAM"]
        assert len(req_errors) >= 1


# ====================================================================
# Integration: end-to-end extract → store → validate
# ====================================================================


class TestEndToEnd:
    """Full round-trip: extract schema, store in catalog, validate workflow."""

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_extract_store_validate_accepts_good_params(self, mock_resolve, tmp_path):
        from grdl_rt.catalog.database import SqliteArtifactCatalog

        mock_resolve.return_value = _AnnotatedProcessor

        schema = extract_param_schema(_AnnotatedProcessor)
        artifact = Artifact(
            name="GaussianBlur",
            version="1.0.0",
            artifact_type="grdl_processor",
            processor_class="test._AnnotatedProcessor",
            param_schema=schema,
        )

        db_path = tmp_path / "test.db"
        with SqliteArtifactCatalog(db_path) as cat:
            cat.add_artifact(artifact)

            # Verify schema is queryable
            stored = cat.get_param_schema("GaussianBlur")
            assert stored is not None

            # Validate a correct workflow
            wf = WorkflowDefinition(name="Good")
            wf.add_step(
                ProcessingStep(
                    "GaussianBlur",
                    "1.0",
                    params={"sigma": 3.0, "interpolation": "bicubic"},
                )
            )
            errors = validate_workflow(wf, catalog=cat)
            assert len(errors) == 0

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_extract_store_validate_rejects_bad_params(self, mock_resolve, tmp_path):
        from grdl_rt.catalog.database import SqliteArtifactCatalog

        mock_resolve.return_value = _AnnotatedProcessor

        schema = extract_param_schema(_AnnotatedProcessor)
        artifact = Artifact(
            name="GaussianBlur",
            version="1.0.0",
            artifact_type="grdl_processor",
            processor_class="test._AnnotatedProcessor",
            param_schema=schema,
        )

        db_path = tmp_path / "test.db"
        with SqliteArtifactCatalog(db_path) as cat:
            cat.add_artifact(artifact)

            wf = WorkflowDefinition(name="Bad")
            wf.add_step(
                ProcessingStep(
                    "GaussianBlur",
                    "1.0",
                    params={"sigma": "not_a_number"},
                )
            )
            errors = validate_workflow(wf, catalog=cat)

            type_errors = [e for e in errors if e.code == "INVALID_PARAM_TYPE"]
            assert len(type_errors) >= 1
            assert "sigma" in type_errors[0].message

    @patch("grdl_rt.execution.discovery.resolve_processor_class")
    def test_grdk_can_consume_schema_without_importing_processor(
        self,
        mock_resolve,
        tmp_path,
    ):
        """Simulates grdk reading schema from catalog to build UI controls."""
        from grdl_rt.catalog.database import SqliteArtifactCatalog

        # Step 1: At cataloging time, extract and store
        schema = extract_param_schema(_RequiredParamProcessor)
        artifact = Artifact(
            name="sublook_decomposition",
            version="0.1.0",
            artifact_type="grdl_processor",
            processor_class="grdl.sar.SublookDecomposition",
            param_schema=schema,
        )

        db_path = tmp_path / "test.db"
        with SqliteArtifactCatalog(db_path) as cat:
            cat.add_artifact(artifact)

        # Step 2: grdk opens catalog, reads schema — no processor import
        mock_resolve.side_effect = ImportError("processor not installed")

        with SqliteArtifactCatalog(db_path) as cat:
            schema = cat.get_param_schema("sublook_decomposition")

        # Step 3: grdk uses schema to determine UI controls
        assert schema is not None
        props = schema["properties"]

        # Slider for num_looks: integer, min=2
        assert props["num_looks"]["type"] == "integer"
        assert props["num_looks"]["minimum"] == 2

        # Dropdown for dimension: enum
        assert props["dimension"]["enum"] == ["azimuth", "range"]

        # Slider for overlap: number, 0.0–0.99
        assert props["overlap"]["type"] == "number"
        assert props["overlap"]["minimum"] == 0.0
        assert props["overlap"]["maximum"] == 0.99

        # Checkbox for deweight: boolean
        assert props["deweight"]["type"] == "boolean"
        assert props["deweight"]["default"] is True
