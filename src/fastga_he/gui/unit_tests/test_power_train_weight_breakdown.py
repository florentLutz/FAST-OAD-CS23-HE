# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
from shutil import rmtree

import pytest

from ..power_train_weight_breakdown import power_train_mass_breakdown

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_pt_weight_breakdown(cleanup):
    """
    Basic tests for testing the power train weight breakdown.
    """

    # No real way to verify the plot, so we will simply endure the next line doesn't cause a crash
    _ = power_train_mass_breakdown(
        pth.join(DATA_FOLDER_PATH, "sample_out.xml"),
        pth.join(DATA_FOLDER_PATH, "simple_assembly.yml"),
    )
