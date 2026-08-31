# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import os.path as pth

import pytest

from ..npv_curve import npv_curve

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_production_npv():
    """
    Basic tests for NPV curve.
    """

    # Check that we can create a plot with no previous plot
    fig = npv_curve(
        pth.join(DATA_FOLDER_PATH, "tbm900_lca.xml"), name="Production NPV", period_var=10
    )

    fig.show()
