# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import pathlib

import pytest

import numpy as np

import plotly.graph_objects as go

import fastoad.api as oad

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from ..lca import LCA

DATA_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "data"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "results" / "parametric_study"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_environmental_impact_function_span_hybrid():
    input_file_name = "hybrid_kodiak.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "hybrid_propulsion.yml",
        component_level_breakdown=True,
        airframe_material="aluminium",
        delivery_method="flight",
        impact_assessment_method="EF v3.1",
        normalization=True,
        weighting=True,
        aircraft_lifespan_in_hours=True,
    )

    ivc = get_indep_var_comp(
        list_inputs(component),
        __file__,
        DATA_FOLDER_PATH / input_file_name,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    mean_airframe_hours = 3524.9
    std_airframe_hours = 266.5

    # To have a widespread, we cover values of airframe hours that goes from the average airframe
    # hours of the fleet (with confidence interval) to an estimate of the average max airframe
    # hours of the fleet (which we'll estimate as twice the average airframe hours of the fleet,
    # with the confidence interval)

    print("First run done")

    for airframe_hours in np.linspace(
        mean_airframe_hours - 3.0 * std_airframe_hours,
        2.0 * mean_airframe_hours + 6.0 * std_airframe_hours,
    ):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")

        problem.run_model()

        file_name = input_file_name.split(".")[0] + "_" + str(int(airframe_hours)) + ".xml"
        problem.output_file_path = RESULTS_FOLDER_PATH / file_name
        problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_draw_sensitivity_to_lifespan():
    fig = go.Figure()

    x, y = [], []

    for dirpath, _, filenames in os.walk(RESULTS_FOLDER_PATH):
        for filename in filenames:
            x.append(int(filename.split(".")[0]))
            datafile = oad.DataFile(os.path.join(dirpath, filename))
            single_score = datafile["data:environmental_impact:single_score"].value[0]
            y.append(single_score)

    scatter = go.Scatter(x=x, y=y)

    fig.add_trace(scatter)
    fig.update_layout(
        plot_bgcolor="white",
        title_x=0.5,
        title_text="Evolution of the single score with life expectancy of the aircraft",
    )
    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        title="Airframe hours [h]",
    )
    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
        range=[0, None],
        title="Single score [-]",
    )
    fig.show()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_environmental_impact_function_span_conventional():
    input_file_name = "ref_kodiak_op.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "turboshaft_propulsion.yml",
        component_level_breakdown=True,
        airframe_material="aluminium",
        delivery_method="flight",
        impact_assessment_method="EF v3.1",
        normalization=True,
        weighting=True,
        aircraft_lifespan_in_hours=True,
        use_operational_mission=True,
    )

    ivc = get_indep_var_comp(
        list_inputs(component),
        __file__,
        DATA_FOLDER_PATH / input_file_name,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    mean_airframe_hours = 3524.9
    std_airframe_hours = 266.5

    # To have a widespread, we cover values of airframe hours that goes from the average airframe
    # hours of the fleet (with confidence interval) to an estimate of the average max airframe
    # hours of the fleet (which we'll estimate as twice the average airframe hours of the fleet,
    # with the confidence interval)

    print("First run done")

    for airframe_hours in np.linspace(
        mean_airframe_hours - 3.0 * std_airframe_hours,
        2.0 * mean_airframe_hours + 6.0 * std_airframe_hours,
    ):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")

        problem.run_model()

        file_name = input_file_name.split(".")[0] + "_" + str(int(airframe_hours)) + ".xml"
        problem.output_file_path = RESULTS_FOLDER_PATH / file_name
        problem.write_outputs()
