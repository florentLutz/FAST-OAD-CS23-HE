#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import os
import pathlib

import pytest

from ..lca_impact import (
    lca_impacts_bar_chart_simple,
    lca_impacts_sun_breakdown,
    lca_impacts_bar_chart_normalised,
    lca_impacts_bar_chart_with_contributors,
)

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data_lca_pipistrel"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart():
    fig = lca_impacts_bar_chart_simple(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_eu_mix.xml",
        ],
        names_aircraft=[
            "Pipistrel SW121",
            "Pipistrel Velis Electro (FR mix)",
            "Pipistrel Velis Electro (EU mix)",
        ],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_sun_breakdown_pipistrel_presentation():
    # Check that we can create a plot
    fig = lca_impacts_sun_breakdown(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out.xml",
        ],
        full_burst=True,
        rel="single_score",
        name_aircraft=["Thermal propulsion", "Electric Propulsion"],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized():
    fig = lca_impacts_bar_chart_normalised(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out_recipe.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_recipe.xml",
        ],
        names_aircraft=[
            "Pipistrel SW121",
            "Pipistrel Velis Electro",
        ],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized_ef():
    fig = lca_impacts_bar_chart_normalised(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out.xml",
        ],
        names_aircraft=[
            "Pipistrel SW121",
            "Pipistrel Velis Electro",
        ],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized_arvidsson():
    fig = lca_impacts_bar_chart_normalised(
        [
            DATA_FOLDER_PATH / "pipistrel_club_lca_out_arvidsson.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_arvidsson.xml",
        ],
        names_aircraft=[
            "Pipistrel Alpha Trainer",
            "Pipistrel Alpha Electro",
        ],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_absolute_phase_hybrid():
    fig = lca_impacts_bar_chart_with_contributors(
        DATA_FOLDER_PATH / "pipistrel_electro_lca_out_recipe.xml",
        name_aircraft="pipistrel Velis Electro",
        impact_step="normalized",
        impact_filter_list=[
            "ecotoxicity_freshwater",
            "ecotoxicity_marine",
            "human_toxicity_carcinogenic",
            "energy_resources_non-renewablefossil",
            "climate_change",
        ],
        aggregate_and_sort_contributor={
            "Airframe": "airframe",
            "Battery pack": ["battery_pack_1", "battery_pack_2"],
            "Others": [
                "motor_1",
                "inverter_1",
                "harness_1",
                "dc_sspc_1",
                "dc_sspc_2",
                "dc_splitter_1",
                "dc_bus_1",
                "manufacturing",
                "distribution",
            ],
            "Use phase": "electricity_for_mission",
            "Propeller": "propeller_1",
        },
    )
    fig.update_layout(title_text=None, height=800, width=1000)
    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_normalized_comparison_with_heavy():
    fig = lca_impacts_bar_chart_simple(
        [
            DATA_FOLDER_PATH / "pipistrel_electro_lca_out_recipe.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_heavy_lca_out_recipe.xml",
            DATA_FOLDER_PATH / "pipistrel_electro_heavy_lca_out_recipe_btf.xml",
        ],
        names_aircraft=[
            "Pipistrel Velis Electro (composite version, buy-to-fly=1)",
            "Pipistrel Velis Electro (metallic version, buy-to-fly=1)",
            "Pipistrel Velis Electro (metallic version, buy-to-fly=10)",
        ],
        impact_step="normalized",
        graph_title="Comparison of Pipistrel version with different materials for the airframe"
    )

    fig.show()
