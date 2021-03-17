"""
Testing utils.py
"""

import os


def test_g_download(midas_model):
    # id_file = "1VuJRqB79JJ8GLCJJjbWPmvxGuADMcnVx"
    # path = os.path.join(app, "model.pt")

    # utils.g_download(id_file, path)
    assert os.path.isfile(midas_model)
