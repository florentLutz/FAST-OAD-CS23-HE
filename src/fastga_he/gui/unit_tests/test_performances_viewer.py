# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
from shutil import rmtree

import pytest

from ..performances_viewer import PerformancesViewer

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_performances_viewer(cleanup):
    """
    Basic tests for testing the performances viewer.
    """

    mission_data_filename = pth.join(DATA_FOLDER_PATH, "mission_data.csv")
    pt_data_filename = pth.join(DATA_FOLDER_PATH, "power_train_data.csv")

    performances_viewer = PerformancesViewer(
        power_train_data_file_path=pt_data_filename, mission_data_file_path=mission_data_filename
    )

    # Testing display
    performances_viewer.display()
