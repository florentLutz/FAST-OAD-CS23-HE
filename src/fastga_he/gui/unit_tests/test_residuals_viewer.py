# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os

import pytest

from ..residuals_viewer import residuals_viewer

DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data")

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_residuals_viewer():
    """
    Basic tests for testing the power train weight breakdown.
    """

    # No real way to verify the plot, we wil just check that it is created.
    fig = residuals_viewer(
        recorder_data_file_path=os.path.join(DATA_FOLDER_PATH, "cases.sql"),
        case="root.performances.nonlinear_solver",
        power_train_file_path=os.path.join(DATA_FOLDER_PATH, "assembly_for_recorder.yml"),
        what_to_plot="residuals",
    )

    fig.show()
