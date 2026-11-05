# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import pathlib

from ..financial_performance_curve import financial_performance_curve

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"

DHC_6 = DATA_FOLDER_PATH / "twin_otter_lcc.xml"
DHC_6_H2 = DATA_FOLDER_PATH / "h2_twin_otter_lcc.xml"


def test_production_npv():
    """
    Test the production of the npv curve.
    """
    financial_performance_curve(
        aircraft_file_path=[DHC_6, DHC_6_H2],
        name=["DHC6 Twin Otter", "Hydrogen Retrofit DHC6"],
        y_variable="data:cost:production:cumulative_annual_delivery_count",
        y2_axis_label="A/C delivered [units]",
    )
