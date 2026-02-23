"""
Tests for grdl-runtime package scaffold.

Validates that the package can be imported, exposes a version attribute,
and that both subpackages (execution, catalog) are importable.
"""


class TestPackageImport:
    """Verify the top-level package is importable."""

    def test_import_grdl_rt(self):
        import grdl_rt

        assert grdl_rt is not None

    def test_version_attribute(self):
        import grdl_rt

        assert hasattr(grdl_rt, "__version__")
        assert isinstance(grdl_rt.__version__, str)
        assert grdl_rt.__version__ == "0.1.1"


class TestSubpackageImport:
    """Verify subpackages are importable."""

    def test_import_execution(self):
        import grdl_rt.execution

        assert grdl_rt.execution is not None

    def test_import_catalog(self):
        import grdl_rt.catalog

        assert grdl_rt.catalog is not None


class TestNoGuiDependencies:
    """Verify the package has no GUI framework references."""

    def test_no_qt_in_init(self):
        import inspect

        import grdl_rt

        source = inspect.getsource(grdl_rt)
        for forbidden in ("PyQt6", "PySide6", "orange", "napari", "QWidget", "QApplication"):
            assert forbidden not in source, f"Found forbidden GUI reference: {forbidden}"
