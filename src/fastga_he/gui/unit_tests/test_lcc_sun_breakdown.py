# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import os
import os.path as pth
import pathlib

import time

import pytest

from ..lcc_cost import lcc_production_cost_sun_breakdown, lcc_operation_cost_sun_breakdown

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULT_FOLDER_PATH = pathlib.Path(__file__).parent / "results"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_lcc_sun_breakdown():
    # Check that we can create a plot
    fig = lcc_production_cost_sun_breakdown(pth.join(DATA_FOLDER_PATH, "twin_otter_lcc.xml"))

    fig.show()

    fig = lcc_operation_cost_sun_breakdown(pth.join(DATA_FOLDER_PATH, "twin_otter_lcc.xml"))

    fig.show()
