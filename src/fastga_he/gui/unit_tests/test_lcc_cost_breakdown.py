# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import os
import pathlib
import pytest

from ..lcc_cost import (
    lcc_cost_bar_chart_simple,
    lcc_power_train_cost_breakdown,
    lcc_cost_bar_chart_advanced,
)

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"

DHC_6 = DATA_FOLDER_PATH / "twin_otter_lcc.xml"
DHC_6_H2 = DATA_FOLDER_PATH / "h2_twin_otter_lcc.xml"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lcc_cost_breakdown_simple():
    fig = lcc_cost_bar_chart_simple(
        aircraft_file_paths=[DHC_6, DHC_6_H2],
        names_aircraft=["DHC6 Twin Otter", "Hydrogen Retrofit DHC6"],
        color_list=["#F59E0B", "#6CC5F0"],
        cost_type="production",
    )

    fig.show()

    fig = lcc_cost_bar_chart_simple(
        aircraft_file_paths=[DHC_6, DHC_6_H2],
        names_aircraft=["DHC6 Twin Otter", "Hydrogen Retrofit DHC6"],
        color_list=["#F59E0B", "#6CC5F0"],
        cost_type="operation",
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lcc_powertrain_cost_breakdown():
    fig = lcc_power_train_cost_breakdown(
        aircraft_file_paths=[DHC_6, DHC_6_H2],
        names_aircraft=["DHC6 Twin Otter", "Hydrogen Retrofit DHC6"],
        group_by="type",
        cost_type="production",
        aggregate_and_sort_contributor={"Electronics": ["DC_DC_converter", "inverter"]},
    )

    fig.show()

    fig = lcc_power_train_cost_breakdown(
        aircraft_file_paths=[DHC_6, DHC_6_H2],
        names_aircraft=["DHC6 Twin Otter", "Hydrogen Retrofit DHC6"],
        group_by="type",
        cost_type="operation",
        exclude_list=["pemfc_stack_1", "turboshaft"],
        cutoff_criteria=5.0,
    )

    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lcc_cost_breakdown_advanced():
    fig = lcc_cost_bar_chart_advanced(
        aircraft_file_paths=[DHC_6, DHC_6_H2],
        names_aircraft=["DHC6 Twin Otter", "Hydrogen Retrofit DHC6"],
        cost_type="production",
        power_train_detail="type",
        aggregate_and_sort_contributor={"Electronics": ["DC_DC_converter", "inverter"]},
    )

    fig.show()

    fig = lcc_cost_bar_chart_advanced(
        aircraft_file_paths=[DHC_6, DHC_6_H2],
        names_aircraft=["DHC6 Twin Otter", "Hydrogen Retrofit DHC6"],
        cost_type="operation",
        power_train_detail="type",
        exclude_list=["pemfc_stack_1", "turboshaft"],
        cutoff_criteria=5.0,
    )

    fig.show()

    flight_per_year = 489.195
    dhc_6_passenger_per_flight = 9
    h2_dhc_6_passenger_per_flight = 4

    fig = lcc_cost_bar_chart_advanced(
        aircraft_file_paths=[DHC_6, DHC_6_H2],
        names_aircraft=["DHC6 Twin Otter", "Hydrogen Retrofit DHC6"],
        cost_type="operation",
        power_train_detail="type",
        normalisation_factors=[
            flight_per_year * dhc_6_passenger_per_flight,
            flight_per_year * h2_dhc_6_passenger_per_flight,
        ],
        value_unit="USD/PAX/flight/year",
    )

    fig.show()
