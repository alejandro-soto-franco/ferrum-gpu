"""pytest fixtures for ferrum-gpu integration tests."""

import pytest

import ferrum_gpu as fgpu


@pytest.fixture(scope="session")
def fgpu_module():
    return fgpu
