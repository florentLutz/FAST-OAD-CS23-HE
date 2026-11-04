# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import os.path as pth
import pathlib

from ..financial_performance_curve import financial_performance_curve

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULT_FOLDER_PATH = pathlib.Path(__file__).parent / "results"


def test_production_npv():
    """
    Test the production of the npv curve.
    """
    financial_performance_curve(
        data_folder_path=DATA_FOLDER_PATH,
        result_folder_path=RESULT_FOLDER_PATH,
        show_plot=False,
    )
