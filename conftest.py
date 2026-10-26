"""
Pytest configuration shared across the test suite.

- Puts the workspace root on sys.path so `import common` works no matter how
  pytest is invoked.
- Registers the `carla_server` marker used for tests that require a live CARLA
  simulator (skipped automatically when no server / carla package is available).
"""

import os
import sys

# Ensure the workspace root (this directory) is importable.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def pytest_configure(config):
    """Register custom markers to avoid unknown-marker warnings."""
    config.addinivalue_line(
        "markers",
        "carla_server: requires a running CARLA simulator and the carla python package",
    )


def _carla_is_available() -> bool:
    """Return True iff we can import carla AND connect to a running simulator."""
    try:
        from common.carla_env import CarlaDrivingEnv  # local import (needs sys.path)
    except Exception:
        return False
    try:
        env = CarlaDrivingEnv(connect_on_init=True)
    except Exception:
        return False
    try:
        env.close()
    except Exception:
        pass
    return True


import pytest


@pytest.fixture
def carla_env_live():
    """Yield a live (connected) CarlaDrivingEnv, skipping if unavailable.

    This fixture is the single place that decides whether a 'carla_server'
    marked test runs. Tests that need the real simulator should depend on this
    fixture instead of constructing the env themselves.
    """
    if not _carla_is_available():
        pytest.skip("CARLA simulator / carla package not available")
    from common.carla_env import CarlaDrivingEnv
    env = CarlaDrivingEnv(connect_on_init=True)
    try:
        yield env
    finally:
        env.close()
