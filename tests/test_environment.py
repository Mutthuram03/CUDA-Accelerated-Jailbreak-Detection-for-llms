"""
Tests for environment sanity and dependency availability.
Compatible with both unittest and pytest.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.system_info import get_system_telemetry


class TestEnvironment(unittest.TestCase):
    """Verifies Python runtime environment and core packages."""

    def test_python_version(self):
        """Verify supported Python version is 3.10+."""
        self.assertGreaterEqual(
            sys.version_info[:2],
            (3, 10),
            f"Python 3.10+ required. Found {sys.version}"
        )

    def test_core_dependencies(self):
        """Verify presence of core scientific dependencies."""
        import numpy as np
        import pandas as pd
        import sklearn
        import yaml

        self.assertIsNotNone(np.__version__)
        self.assertIsNotNone(pd.__version__)
        self.assertIsNotNone(sklearn.__version__)
        self.assertIsNotNone(yaml.__version__)

    def test_system_telemetry(self):
        """Verify telemetry collection returns well-formed data."""
        telemetry = get_system_telemetry()
        self.assertIn("os", telemetry)
        self.assertIn("cpu_count_logical", telemetry)
        self.assertIn("host_ram_total_gb", telemetry)
        self.assertIn("gpu_available", telemetry)
        self.assertIsInstance(telemetry["host_ram_total_gb"], float)


if __name__ == "__main__":
    unittest.main()
