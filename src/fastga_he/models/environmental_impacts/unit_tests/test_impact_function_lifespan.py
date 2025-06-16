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
RESULTS2_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "results" / "parametric_study_2"
RESULTS_PIPISTREL_FOLDER_PATH = (
    pathlib.Path(__file__).parents[0] / "results" / "parametric_study_pipistrel"
)

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
        write_lca_conf=False,
        lca_conf_file_path=DATA_FOLDER_PATH / "hybrid_propulsion_lca_paper.yml",
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

    # The fuel stored in the tanks do not consider the fuel necessary for takeoff, it doesn't affect
    # the sizing but does affect the LCA, this is a temporary fix. For the battery it's not a
    # problem as there was no energy consumed during takeoff
    datafile = oad.DataFile(DATA_FOLDER_PATH / input_file_name)

    # Fuel is equally distributed in both tanks so we can do this
    fuel_main_route = 2.0 * problem.get_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_main_route",
        units="kg",
    )
    fuel_mission = 2.0 * datafile["data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_mission"].value[0]
    fuel_takeoff = datafile["data:mission:sizing:takeoff:fuel"].value[0]
    new_fuel = fuel_main_route + fuel_takeoff
    tmp = fuel_mission / (fuel_mission + fuel_takeoff)
    ratio_for_emissions = tmp * new_fuel / fuel_main_route

    problem.set_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_main_route",
        units="kg",
        val=new_fuel / 2.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_2:fuel_consumed_main_route",
        units="kg",
        val=new_fuel / 2.0,
    )

    # For now the emissions are computed using emission index which means they are proportional
    # to the fuel so we can do the following to remove the emissions in reserve
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO_main_route"
        ].value[0]
        * ratio_for_emissions,
    )
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO2_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO2_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO2_main_route"
        ].value[0]
        * ratio_for_emissions,
    )
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:H2O_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:H2O_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:H2O_main_route"
        ].value[0]
        * ratio_for_emissions,
    )
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:HC_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:HC_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:HC_main_route"
        ].value[0]
        * ratio_for_emissions,
    )
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:NOx_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:NOx_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:NOx_main_route"
        ].value[0]
        * ratio_for_emissions,
    )
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:SOx_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:SOx_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:SOx_main_route"
        ].value[0]
        * ratio_for_emissions,
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

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        8.80965454150987e-06, rel=1e-3
    )
    assert problem.get_val(
        "data:LCA:operation:he_power_train:kerosene:mass_per_fu", units="kg"
    ) == pytest.approx(0.03083699, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:turboshaft:turboshaft_1:CO2_per_fu", units="kg"
    ) == pytest.approx(0.09034836871998066, rel=1e-3)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_environmental_impact_function_span_hybrid_longer():
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
        write_lca_conf=False,
        lca_conf_file_path=DATA_FOLDER_PATH / "hybrid_propulsion_lca_paper.yml",
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
        3.0 * mean_airframe_hours + 9.0 * std_airframe_hours,
        100,
    ):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")

        problem.run_model()

        file_name = input_file_name.split(".")[0] + "_" + str(int(airframe_hours)) + ".xml"
        problem.output_file_path = RESULTS2_FOLDER_PATH / file_name
        problem.write_outputs()


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
        write_lca_conf=False,
        lca_conf_file_path=DATA_FOLDER_PATH / "turboshaft_propulsion_lca_paper.yml",
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

    # The fuel stored in the tanks do not consider the fuel necessary for takeoff, it doesn't affect
    # the sizing but does affect the LCA, this is a temporary fix
    datafile = oad.DataFile(DATA_FOLDER_PATH / input_file_name)

    # Fuel is equally distributed in both tanks so we can do this
    fuel_main_route = 2.0 * problem.get_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_main_route",
        units="kg",
    )
    fuel_mission = 2.0 * datafile["data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_mission"].value[0]
    fuel_takeoff = datafile["data:mission:operational:takeoff:fuel"].value[0]
    new_fuel = fuel_main_route + fuel_takeoff
    tmp = fuel_mission / (fuel_mission + fuel_takeoff)
    ratio_for_emissions = tmp * new_fuel / fuel_main_route

    problem.set_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_main_route",
        units="kg",
        val=new_fuel / 2.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_2:fuel_consumed_main_route",
        units="kg",
        val=new_fuel / 2.0,
    )

    # For now the emissions are computed using emission index which means they are proportional
    # to the fuel so we can do the following to remove the emissions in reserve
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO_main_route"
        ].value[0]
        * ratio_for_emissions,
    )
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO2_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO2_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:CO2_main_route"
        ].value[0]
        * ratio_for_emissions,
    )
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:H2O_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:H2O_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:H2O_main_route"
        ].value[0]
        * ratio_for_emissions,
    )
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:HC_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:HC_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:HC_main_route"
        ].value[0]
        * ratio_for_emissions,
    )
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:NOx_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:NOx_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:NOx_main_route"
        ].value[0]
        * ratio_for_emissions,
    )
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:SOx_main_route",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:SOx_main_route"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:turboshaft:turboshaft_1:SOx_main_route"
        ].value[0]
        * ratio_for_emissions,
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

    assert problem.get_val("data:environmental_impact:single_score") == pytest.approx(
        9.34530896e-06, rel=1e-3
    )
    assert problem.get_val(
        "data:LCA:operation:he_power_train:kerosene:mass_per_fu", units="kg"
    ) == pytest.approx(0.036003152728011176, rel=1e-3)
    assert problem.get_val(
        "data:LCA:operation:he_power_train:turboshaft:turboshaft_1:CO2_per_fu", units="kg"
    ) == pytest.approx(0.10665994973455112, rel=1e-3)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_environmental_impact_function_span_pipistrel_velis_electro():
    input_file_name = "pipistrel_out.xml"

    pipistrel_data = pathlib.Path(__file__).parents[0] / "data_lca_pipistrel" / input_file_name

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
        impact_assessment_method="EF v3.1",
    )

    ivc = get_indep_var_comp(
        list_inputs(component),
        __file__,
        pipistrel_data,
    )

    # First run of the problem but unused, just to be able to change the values without having to
    # recreate a problem each time
    problem = run_system(
        component,
        ivc,
    )

    mean_airframe_hours = 2956.7
    std_airframe_hours = 93.7

    # To have a widespread, we cover values of airframe hours that goes from the average airframe
    # hours of the fleet (with confidence interval) to an estimate of the average max airframe
    # hours of the fleet (which we'll estimate as twice the average airframe hours of the fleet,
    # with the confidence interval)

    print("First run done")

    for airframe_hours in np.linspace(
        mean_airframe_hours - 3.0 * std_airframe_hours,
        2.0 * mean_airframe_hours + 6.0 * std_airframe_hours,
        100,
    ):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")

        problem.run_model()

        file_name = input_file_name.split(".")[0] + "_" + str(int(airframe_hours)) + ".xml"
        problem.output_file_path = RESULTS_PIPISTREL_FOLDER_PATH / file_name
        problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_environmental_impact_function_span_pipistrel_velis_electro_longer_battery():
    input_file_name = "pipistrel_out.xml"

    pipistrel_data = pathlib.Path(__file__).parents[0] / "data_lca_pipistrel" / input_file_name

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
        impact_assessment_method="EF v3.1",
    )

    ivc = get_indep_var_comp(
        list_inputs(component),
        __file__,
        pipistrel_data,
    )

    # First run of the problem but unused, just to be able to change the values without having to
    # recreate a problem each time
    problem = run_system(
        component,
        ivc,
    )

    mean_airframe_hours = 2956.7
    std_airframe_hours = 93.7

    # To have a widespread, we cover values of airframe hours that goes from the average airframe
    # hours of the fleet (with confidence interval) to an estimate of the average max airframe
    # hours of the fleet (which we'll estimate as twice the average airframe hours of the fleet,
    # with the confidence interval)

    print("First run done")

    for airframe_hours in np.linspace(
        mean_airframe_hours - 3.0 * std_airframe_hours,
        2.0 * mean_airframe_hours + 6.0 * std_airframe_hours,
        100,
    ):
        problem.set_val("data:TLAR:max_airframe_hours", val=airframe_hours, units="h")

        problem.run_model()

        file_name = (
            "long_batt" + input_file_name.split(".")[0] + "_" + str(int(airframe_hours)) + ".xml"
        )
        problem.output_file_path = RESULTS_PIPISTREL_FOLDER_PATH / file_name
        problem.write_outputs()
