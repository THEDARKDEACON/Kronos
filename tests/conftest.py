"""Pytest configuration."""

import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: model/API integration tests")
    config.addinivalue_line("markers", "slow: long-running tests")


def pytest_collection_modifyitems(config, items):
    if os.getenv("RUN_MODEL_TESTS") == "1":
        return
    skip_integration = pytest.mark.skip(
        reason="Set RUN_MODEL_TESTS=1 to run model integration tests"
    )
    for item in items:
        if "integration" in item.keywords or "test_kronos_regression" in item.nodeid:
            item.add_marker(skip_integration)
