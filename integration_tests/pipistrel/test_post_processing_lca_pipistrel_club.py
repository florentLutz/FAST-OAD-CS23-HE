#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import pathlib

from fastga_he.gui.lca_impact import (
    lca_impacts_bar_chart_simple,
    lca_raw_impact_comparison_advanced,
    lca_impacts_bar_chart_with_contributors,
)

RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"


def test_compare_impacts_designs_simple_bar_chart():
    fig = lca_impacts_bar_chart_simple(
        [
            RESULTS_FOLDER_PATH / "pipistrel_club_with_lca_out.xml",
            RESULTS_FOLDER_PATH / "pipistrel_club_heavy_with_lca_out.xml",
        ],
        names_aircraft=[
            "Reference Pipistrel Club",
            "Heavy Pipistrel Club",
        ],
    )

    fig.show()


def test_compare_impacts_designs_with_contributor():
    fig = lca_raw_impact_comparison_advanced(
        [
            RESULTS_FOLDER_PATH / "pipistrel_club_with_lca_out.xml",
            RESULTS_FOLDER_PATH / "pipistrel_club_heavy_with_lca_out.xml",
        ],
        names_aircraft=[
            "Reference Pipistrel Club",
            "Heavy Pipistrel Club",
        ],
        impact_category="human toxicity non-carcinogenic",
        aggregate_and_sort_contributor={
            "Airframe": "airframe",  # Just a renaming, should work as well,
            "In flight emissions": "ice_1",  # Just a renaming, should work as well,
            "Fuel production": "gasoline_for_mission",  # Just a renaming, should work as well,
            "Others": [
                "propeller_1",
                "fuel_system_1",
                "manufacturing",
                "distribution",
            ],
        },
    )
    fig.update_layout(width=1000)

    fig.show()


def test_lca_bar_chart_relative_contribution_original_design():
    fig = lca_impacts_bar_chart_with_contributors(
        RESULTS_FOLDER_PATH / "pipistrel_club_with_lca_out.xml",
        name_aircraft="Reference Pipistrel Club",
    )

    fig.show()
