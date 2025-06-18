"""
Basic tests that don't require problematic dependencies.
"""

import pytest
import sys
from pathlib import Path


def test_python_version():
    """Test that we're using a compatible Python version."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"


def test_import_basic_modules():
    """Test importing basic Python modules."""
    import json
    import logging
    import tempfile

    assert True


def test_project_structure():
    """Test that the project has the expected structure."""
    project_root = Path(__file__).parent.parent

    # Check for essential files
    assert (project_root / "requirements.txt").exists()
    assert (project_root / "README.md").exists()
    assert (project_root / "config.yaml").exists()

    # Check for directories
    assert (project_root / "tests").exists()
    assert (project_root / "data").exists() or (project_root / "output").exists()


def test_config_file():
    """Test that config.yaml is valid YAML."""
    try:
        import yaml

        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)
        assert "data" in config
        assert "model" in config
    except ImportError:
        pytest.skip("PyYAML not available")


def test_requirements_file():
    """Test that requirements.txt exists and is readable."""
    req_path = Path(__file__).parent.parent / "requirements.txt"
    assert req_path.exists()

    with open(req_path, "r") as f:
        requirements = f.read()

    assert len(requirements) > 0
    assert "numpy" in requirements
    assert "pandas" in requirements


def test_makefile():
    """Test that Makefile exists and has expected targets."""
    makefile_path = Path(__file__).parent.parent / "Makefile"
    assert makefile_path.exists()

    with open(makefile_path, "r") as f:
        makefile_content = f.read()

    assert "test:" in makefile_content
    assert "install:" in makefile_content


if __name__ == "__main__":
    pytest.main([__file__])
