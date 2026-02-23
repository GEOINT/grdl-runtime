"""
Tests for headless guarantee — no GUI framework imports in grdl_rt/.

Ensures that the grdl-runtime package can be used in headless server
environments without requiring PyQt6, Orange, napari, or any other
GUI framework.

Author
------
Claude Code (Anthropic)

Created
-------
2026-02-09
"""

from pathlib import Path

# Forbidden GUI framework tokens
FORBIDDEN_GUI_TOKENS = [
    "PyQt6",
    "PySide6",
    "from orange",
    "import orange",
    "napari",
    "QWidget",
    "QApplication",
]

# Root of the grdl_rt package
PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "grdl_rt"


class TestHeadlessConstraint:
    """Verify zero GUI framework references in grdl_rt/ source tree."""

    def _scan_files(self):
        """Yield (path, content) for every .py file in the package."""
        for py_file in sorted(PACKAGE_ROOT.rglob("*.py")):
            yield py_file, py_file.read_text(encoding="utf-8")

    def test_no_gui_imports_in_package(self):
        """No .py file in grdl_rt/ should reference GUI frameworks."""
        violations = []
        for py_file, content in self._scan_files():
            for token in FORBIDDEN_GUI_TOKENS:
                if token in content:
                    rel = py_file.relative_to(PACKAGE_ROOT.parent)
                    violations.append(f"{rel}: contains '{token}'")
        assert (
            violations == []
        ), "GUI framework references found in headless package:\n" + "\n".join(violations)

    def test_package_root_exists(self):
        """Sanity check: the package root exists and has Python files."""
        assert PACKAGE_ROOT.is_dir()
        py_files = list(PACKAGE_ROOT.rglob("*.py"))
        assert len(py_files) > 0
