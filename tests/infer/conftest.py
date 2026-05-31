"""Shared pytest options for inference tests."""


def pytest_addoption(parser):
    parser.addoption("--checkpoint_path_14b", default="")
    parser.addoption("--checkpoint_path_1_3b", default="")
    parser.addoption("--port", default=43020, type=int)
    parser.addoption("--server_timeout", default=300, type=float)
