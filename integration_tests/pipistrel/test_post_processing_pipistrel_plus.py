#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import pathlib

from fastga_he.gui.lca_impact import (
    lca_raw_impact_comparison_advanced,
    lca_impacts_sun_breakdown,
    lca_impacts_bar_chart_normalised_weighted,
)

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"


def test_sun_breakdown_pipistrel_plus_plus():
    fig = lca_impacts_sun_breakdown(
        [
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca.xml",
            RESULTS_FOLDER_PATH / "pipistrel_plus_plus_out_with_lca_no_losses.xml",
        ],
        full_burst=True,
        name_aircraft=[
            "Pipistrel Velis Electro",
            "Pipistrel Velis Electro++ (no production losses)",
        ],
    )

    fig.show()


def test_impact_evolution_three_designs():
    fig = lca_impacts_bar_chart_normalised_weighted(
        [
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca.xml",
            RESULTS_FOLDER_PATH / "pipistrel_plus_out_with_lca.xml",
            RESULTS_FOLDER_PATH / "pipistrel_plus_plus_out_with_lca_no_losses.xml",
        ],
        names_aircraft=[
            "Pipistrel Velis Electro",
            "Pipistrel Velis Electro+",
            "Pipistrel Velis Electro++ (no production losses)",
        ],
    )

    fig.show()


def test_compare_impacts_three_designs_with_contributor():
    fig = lca_raw_impact_comparison_advanced(
        [
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca.xml",
            RESULTS_FOLDER_PATH / "pipistrel_plus_out_with_lca.xml",
            RESULTS_FOLDER_PATH / "pipistrel_plus_plus_out_with_lca_no_losses.xml",
        ],
        names_aircraft=[
            "Pipistrel Velis Electro",
            "Pipistrel Velis Electro+",
            "Pipistrel Velis Electro++ (no production losses)",
        ],
        impact_category="material resources metals minerals",  # "climate change", "material resources metals minerals"
        aggregate_and_sort_contributor={
            "Airframe": "airframe",  # Just a renaming, should work as well
            "Battery pack 1": "battery_pack_1",
            "Battery pack 2": "battery_pack_2",
            "Use phase": "electricity_for_mission",  # Just a renaming, should work as well
            "Others": [
                "propeller_1",
                "motor_1",
                "inverter_1",
                "harness_1",
                "dc_bus_1",
                "manufacturing",
                "distribution",
                "dc_sspc_1",
                "dc_sspc_2",
                "dc_splitter_1",
            ],
        },
    )
    fig.update_layout(width=800)

    fig.show()


def test_compare_impacts_designs_with_production_losses():
    fig = lca_raw_impact_comparison_advanced(
        [
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca.xml",
            RESULTS_FOLDER_PATH / "pipistrel_plus_plus_out_with_lca_no_losses.xml",
            RESULTS_FOLDER_PATH / "pipistrel_plus_plus_out_with_lca.xml",
        ],
        names_aircraft=[
            "Pipistrel Velis Electro",
            "Pipistrel Velis Electro++ (no production losses)",
            "Pipistrel Velis Electro++ (production losses)",
        ],
        impact_category="material resources metals minerals",  # "climate change", "material resources metals minerals"
        aggregate_and_sort_contributor={
            "Airframe": "airframe",  # Just a renaming, should work as well
            "Battery pack 1": "battery_pack_1",
            "Battery pack 2": "battery_pack_2",
            "Use phase": "electricity_for_mission",  # Just a renaming, should work as well
            "Others": [
                "propeller_1",
                "motor_1",
                "inverter_1",
                "harness_1",
                "dc_bus_1",
                "manufacturing",
                "distribution",
                "dc_sspc_1",
                "dc_sspc_2",
                "dc_splitter_1",
            ],
        },
    )
    fig.update_layout(width=800)

    fig.show()


def test_impact_evolution_three_designs_with_production_losses():
    fig = lca_impacts_bar_chart_normalised_weighted(
        [
            RESULTS_FOLDER_PATH / "pipistrel_out_with_lca.xml",
            RESULTS_FOLDER_PATH / "pipistrel_plus_out_with_lca.xml",
            RESULTS_FOLDER_PATH / "pipistrel_plus_plus_out_with_lca.xml",
        ],
        names_aircraft=[
            "Pipistrel Velis Electro",
            "Pipistrel Velis Electro+",
            "Pipistrel Velis Electro++",
        ],
    )

    fig.show()
