"""pytest fixtures for ferrum-gpu integration tests."""

import pytest

import ferrum_gpu as fg


@pytest.fixture(scope="session")
def fg_module():
    return fg
