"""
Pytest configuration.
"""

import os
import shutil

import pytest

from depthmap_generator import utils

tmp_folder = os.path.join(os.path.dirname(__file__), "tmp")


@pytest.fixture(scope="session", autouse=True)
def setup_folders():
    os.makedirs(tmp_folder, exist_ok=True)

    yield

    shutil.rmtree(tmp_folder)


@pytest.fixture
def app():
    yield tmp_folder


@pytest.fixture
def midas_model(app):
    id_file = "1VuJRqB79JJ8GLCJJjbWPmvxGuADMcnVx"
    path = os.path.join(app, "model.pt")
    utils.g_download(id_file, path)

    return path
