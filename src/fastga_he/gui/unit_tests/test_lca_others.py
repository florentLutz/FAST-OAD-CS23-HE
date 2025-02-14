# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import pathlib
import time

import pytest

import plotly.graph_objects as go

import fastoad.api as oad

from fastga_he.exceptions import ImpactUnavailableForPlotError
from ..lca_impact import (
    lca_score_sensitivity_simple,
    lca_score_sensitivity_advanced_impact_category,
    lca_score_sensitivity_advanced_components,
    lca_score_sensitivity_advanced_components_and_phase,
    lca_impacts_bar_chart_simple,
    lca_impacts_bar_chart_normalised_weighted,
    lca_impacts_bar_chart_with_contributors,
    lca_impacts_bar_chart_with_phases_absolute,
    lca_impacts_search_table,
)

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"
RESULT_FOLDER_PATH = pathlib.Path(__file__).parent / "results"

PATH_TO_CURRENT_FILE = pathlib.Path(__file__)

SENSITIVITY_STUDIES_FOLDER_PATH = (
    pathlib.Path(__file__).parents[2]
    / "models"
    / "environmental_impacts"
    / "unit_tests"
    / "results"
    / "parametric_study"
)

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_score_sensitivity_analysis():
    # Check that we can create a plot
    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
    )

    # We do that so that the legend doesn't overlap the y axis, which as a reminder, we have to
    # put on the right otherwise we can't change the font without changing the yaxis range
    fig.update_xaxes(domain=[0, 0.95])

    fig = go.FigureWidget(fig)

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_single_score_sensitivity_analysis_two_plots():
    # Check that we can create a plot
    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
    )

    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="ref_kodiak_op",
        name="Reference Kodiak",
        fig=fig,
    )

    # We do that so that the legend doesn't overlap the y-axis, which as a reminder, we have to
    # put on the right otherwise we can't change the font without changing the yaxis range
    fig.update_xaxes(domain=[0, 0.95])

    fig = go.FigureWidget(fig)

    fig.show()

    fig.update_layout(height=800.0, width=1600.0)
    fig.write_image(PATH_TO_CURRENT_FILE.parent / "results" / "ga_single_score_evolution.pdf")
    time.sleep(3)
    fig.write_image(PATH_TO_CURRENT_FILE.parent / "results" / "ga_single_score_evolution.pdf")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_other_impact_sensitivity_analysis():
    # Check that we can create a plot
    fig = lca_score_sensitivity_simple(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        impact_to_plot="material_resources_metals_minerals",
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_unavailable_impact_error():
    try:
        _ = lca_score_sensitivity_simple(
            results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
            impact_to_plot="orange_eutrophication",
            prefix="hybrid_kodiak",
        )
    except ImpactUnavailableForPlotError as e:
        assert " unavailable in the output file. Available impacts include: " in e.args[0]


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_sensitivity_analysis_advanced_impact_categories():
    fig = lca_score_sensitivity_advanced_impact_category(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
        cutoff_criteria=5,
    )

    fig.update_xaxes(domain=[0, 0.95])
    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_sensitivity_analysis_advanced_components():
    fig = lca_score_sensitivity_advanced_components(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
        cutoff_criteria=2,
    )

    fig.update_xaxes(domain=[0, 0.95])
    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_sensitivity_analysis_advanced_components_and_phase():
    fig = lca_score_sensitivity_advanced_components_and_phase(
        results_folder_path=SENSITIVITY_STUDIES_FOLDER_PATH,
        prefix="hybrid_kodiak",
        name="Hybrid Kodiak",
        cutoff_criteria=2,
    )

    fig.update_xaxes(domain=[0, 0.95])
    fig.show()


def test_lca_bar_chart():
    fig = lca_impacts_bar_chart_simple(
        [
            DATA_FOLDER_PATH / "kodiak_100_ef.xml",
            DATA_FOLDER_PATH / "hybrid_kodiak_100_ef.xml",
        ],
        names_aircraft=["Reference Kodiak 100", "Hybrid Kodiak 100"],
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_relative_paper():
    fig = lca_impacts_bar_chart_simple(
        [
            SENSITIVITY_STUDIES_FOLDER_PATH / "ref_kodiak_op_7077.xml",
            SENSITIVITY_STUDIES_FOLDER_PATH / "hybrid_kodiak_7077.xml",
        ],
        names_aircraft=["Reference Kodiak 100", "Hybrid Kodiak 100"],
    )

    fig.show()
    fig.update_layout(height=800, width=1600)
    fig.write_image(RESULT_FOLDER_PATH / "ga_impacts_evolution.pdf")
    time.sleep(3)
    fig.write_image(RESULT_FOLDER_PATH / "ga_impacts_evolution.pdf")


def test_lca_bar_chart_normalised_and_weighted():
    fig = lca_impacts_bar_chart_normalised_weighted(
        [DATA_FOLDER_PATH / "hybrid_kodiak_100_ef.xml"],
        names_aircraft=["Hybrid Kodiak 100"],
    )

    fig.show()


def test_lca_bar_chart_normalised_and_weighted_multiples():
    fig = lca_impacts_bar_chart_normalised_weighted(
        [
            DATA_FOLDER_PATH / "kodiak_100_ef.xml",
            DATA_FOLDER_PATH / "hybrid_kodiak_100_ef.xml",
        ],
        names_aircraft=["Reference Kodiak 100", "Hybrid Kodiak 100"],
    )

    fig.show()


def test_lca_bar_chart_relative_contribution():
    fig = lca_impacts_bar_chart_with_contributors(
        DATA_FOLDER_PATH / "hybrid_kodiak_100_ef.xml",
        name_aircraft="Hybrid Kodiak 100",
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_absolute_phase():
    fig = lca_impacts_bar_chart_with_phases_absolute(
        SENSITIVITY_STUDIES_FOLDER_PATH / "ref_kodiak_op_7077.xml",
        name_aircraft="Hybrid Kodiak 100",
    )
    fig.update_layout(title_text=None)

    fig.show()
    fig.update_layout(height=800, width=1600)
    # Somehow this prevents the ugly footer from appearing !
    fig.write_image(RESULT_FOLDER_PATH / "ga_component_contribution.pdf")
    time.sleep(3)
    fig.write_image(RESULT_FOLDER_PATH / "ga_component_contribution.pdf")


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_bar_chart_absolute_phase_hybrid():
    fig = lca_impacts_bar_chart_with_phases_absolute(
        SENSITIVITY_STUDIES_FOLDER_PATH / "hybrid_kodiak_7077.xml",
        name_aircraft="Hybrid Kodiak 100",
    )
    fig.update_layout(title_text=None)

    fig.show()
    fig.update_layout(height=800, width=1600)
    # Somehow this prevents the ugly footer from appearing !
    fig.write_image(RESULT_FOLDER_PATH / "ga_component_contribution.pdf")
    time.sleep(3)
    fig.write_image(RESULT_FOLDER_PATH / "ga_component_contribution.pdf")


def test_search_engine():
    impact_list = ["*", "acidification", "acidification", "*", "*"]
    phase_list = ["*", "*", "production", "*", "operation"]
    component_list = ["*", "*", "*", "turboshaft_1", "*"]

    impacts_value = lca_impacts_search_table(
        DATA_FOLDER_PATH / "kodiak_100_ef.xml",
        impact_list,
        phase_list,
        component_list,
        rel=False,
    )

    # Should be equal to the single score.
    assert impacts_value[0] == pytest.approx(1.214e-05, rel=1e-3)
    assert impacts_value[1] == pytest.approx(7.478e-07, rel=1e-3)
    assert impacts_value[2] == pytest.approx(1.298e-08, rel=1e-3)
    assert impacts_value[3] == pytest.approx(6.072e-06, rel=1e-3)
    assert impacts_value[4] == pytest.approx(1.179e-05, rel=1e-3)

    impact_list = ["*", "*", "*", "*"]
    phase_list = ["manufacturing", "distribution", "production", "operation"]
    component_list = ["*", "*", "*", "*"]

    impacts_value = lca_impacts_search_table(
        SENSITIVITY_STUDIES_FOLDER_PATH / "ref_kodiak_op_7077.xml",
        impact_list,
        phase_list,
        component_list,
        rel=False,
    )

    assert sum(impacts_value) == pytest.approx(1.286e-05, rel=1e-3)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_search_engine_production_ref_kodiak_paper():
    impact_list_ref_design = ["*"]
    phase_list_ref_design = ["production"]
    component_list_ref_design = ["*"]

    impacts_value_ref_design = lca_impacts_search_table(
        SENSITIVITY_STUDIES_FOLDER_PATH / "ref_kodiak_op_7077.xml",
        impact_list_ref_design,
        phase_list_ref_design,
        component_list_ref_design,
        rel=True,
    )

    print(impacts_value_ref_design[0])


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_search_engine_paper():
    # For the reference design, we consider the impacts of the fuel consumption and the production
    # of the fuel. Last one is sanity check.
    # Results are naturally given for a kg*km, but since we want the impact per MJ of primary
    # energy used, we must first put them back to one flight and then to one MJ of that flight

    ref_design_datafile = oad.DataFile(SENSITIVITY_STUDIES_FOLDER_PATH / "ref_kodiak_op_7077.xml")
    flights_per_fu_ref_design = ref_design_datafile[
        "data:environmental_impact:flight_per_fu"
    ].value[0]
    print("Flights per FU design", flights_per_fu_ref_design)
    print("FU per flights design", 1.0 / flights_per_fu_ref_design)

    # Not available here directly, have to rely on the amount of kerosene in the tanks
    fuel_burned_ref_design = (
        ref_design_datafile[
            "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_mission"
        ].value[0]
        * 2.0
    )
    energy_mission_ref_design = fuel_burned_ref_design * 11.9  # in kWh
    print("Energy required per FU", energy_mission_ref_design * flights_per_fu_ref_design)

    impact_list_ref_design = ["*", "*", "*"]
    phase_list_ref_design = ["operation", "*", "*"]
    component_list_ref_design = ["turboshaft_1", "kerosene_for_mission", "*"]

    impacts_value_ref_design = lca_impacts_search_table(
        SENSITIVITY_STUDIES_FOLDER_PATH / "ref_kodiak_op_7077.xml",
        impact_list_ref_design,
        phase_list_ref_design,
        component_list_ref_design,
        rel=False,
    )

    assert sum(impacts_value_ref_design[:-1]) > 0.94 * impacts_value_ref_design[-1]

    impact_one_flight = sum(impacts_value_ref_design[:2]) / flights_per_fu_ref_design
    impact_per_kwh_of_energy_used = impact_one_flight / energy_mission_ref_design
    print(impact_per_kwh_of_energy_used)
    print("\n")

    # For the hybrid design, we consider the impacts of the fuel consumption and the production
    # of the fuel, the impact of the production of electricity for the mission and the production
    # of the battery. Last one is sanity check again, those should represent a big part of the
    # total. Results are naturally given for a kg*km, but since we want the impact per MJ of primary
    # energy used, we must first put them back to one flight and then to one MJ of that flight

    hybrid_design_datafile = oad.DataFile(
        SENSITIVITY_STUDIES_FOLDER_PATH / "hybrid_kodiak_7077.xml"
    )
    flights_per_fu_hybrid_design = hybrid_design_datafile[
        "data:environmental_impact:flight_per_fu"
    ].value[0]
    fu_per_flights_hybrid_design = 1.0 / flights_per_fu_hybrid_design
    print("Flights per FU hybrid design", flights_per_fu_hybrid_design)
    print("FU per flights hybrid design", fu_per_flights_hybrid_design)

    # Not available here directly, have to rely on the amount of kerosene in the tanks
    fuel_burned_hybrid_design = (
        hybrid_design_datafile[
            "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_mission"
        ].value[0]
        * 2.0
    )
    electricity_used_hybrid_design = hybrid_design_datafile[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_mission"
    ].value[0]
    electricity_unit = hybrid_design_datafile[
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_mission"
    ].units
    if electricity_unit == "W*h":
        electricity_used_hybrid_design /= 1000.0
    energy_mission_hybrid_design = (
        fuel_burned_hybrid_design * 11.9 + electricity_used_hybrid_design
    )  # in kWh
    print("Energy required per FU", energy_mission_hybrid_design * flights_per_fu_hybrid_design)

    impact_list_hybrid_design = ["*", "*", "*", "*", "*"]
    phase_list_hybrid_design = ["operation", "*", "*", "production", "*"]
    component_list_hybrid_design = [
        "turboshaft_1",
        "kerosene_for_mission",
        "electricity_for_mission",
        "battery_pack_1",
        "*",
    ]

    impacts_value_hybrid_design = lca_impacts_search_table(
        SENSITIVITY_STUDIES_FOLDER_PATH / "hybrid_kodiak_7077.xml",
        impact_list_hybrid_design,
        phase_list_hybrid_design,
        component_list_hybrid_design,
        rel=False,
    )

    assert sum(impacts_value_hybrid_design[:-1]) > 0.93 * impacts_value_hybrid_design[-1]

    impact_one_flight_hybrid = sum(impacts_value_hybrid_design[:-1]) * fu_per_flights_hybrid_design
    impact_per_kwh_of_energy_used_hybrid = impact_one_flight_hybrid / energy_mission_hybrid_design
    print(impact_per_kwh_of_energy_used_hybrid)
    print("\n")
