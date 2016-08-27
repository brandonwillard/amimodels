"""
Example from http://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
"""
import sys

import pytest


def pytest_configure(config):
    import sys
    sys._called_from_test = True


def pytest_unconfigure(config):
    del sys._called_from_test


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
