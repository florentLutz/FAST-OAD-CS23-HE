#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import os
import pathlib

from typing import List

import pytest

import openmdao.api as om
import fastoad.api as oad
from fastoad.io import VariableIO

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from ..lca import LCA

XML_FILE = "data.xml"
DATA_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "data_lca_pipistrel"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "results_lca_pipistrel"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def local_get_indep_var_comp(var_names: List[str], xml_file_name: str) -> om.IndepVarComp:
    """Reads required input data from xml file and returns an IndepVarcomp() instance"""
    reader = VariableIO(DATA_FOLDER_PATH / xml_file_name)
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()

    return ivc


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel():
    input_file_name = "pipistrel_out.xml"

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

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    # The energy stored in the battery does not consider the fuel necessary for takeoff,
    # it doesn't affect the sizing but does affect the LCA, this is a temporary fix. Also, we must
    # not include the energy necessary for the reserve as it does not contribute to the
    # functional unit
    datafile = oad.DataFile(DATA_FOLDER_PATH / input_file_name)

    total_energy_mission = datafile["data:mission:sizing:energy"].value[0]
    energy_reserves = datafile["data:mission:sizing:main_route:reserve:energy"].value[0]
    energy_for_fu = total_energy_mission - energy_reserves

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_mission",
        units="W*h",
        val=energy_for_fu / 2.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:energy_consumed_mission",
        units="W*h",
        val=energy_for_fu / 2.0,
    )

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_electro_lca_out.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_european_mix():
    input_file_name = "pipistrel_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly_eu_mix.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        # electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
        impact_assessment_method="EF v3.1",
    )

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    # The energy stored in the battery does not consider the fuel necessary for takeoff,
    # it doesn't affect the sizing but does affect the LCA, this is a temporary fix. Also, we must
    # not include the energy necessary for the reserve as it does not contribute to the
    # functional unit
    datafile = oad.DataFile(DATA_FOLDER_PATH / input_file_name)

    total_energy_mission = datafile["data:mission:sizing:energy"].value[0]
    energy_reserves = datafile["data:mission:sizing:main_route:reserve:energy"].value[0]
    energy_for_fu = total_energy_mission - energy_reserves

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_mission",
        units="W*h",
        val=energy_for_fu / 2.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:energy_consumed_mission",
        units="W*h",
        val=energy_for_fu / 2.0,
    )

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_electro_lca_out_eu_mix.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_club():
    input_file_name = "pipistrel_club_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_club_assembly.yml",
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

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    # The fuel stored in the tanks does not consider the fuel necessary for takeoff.
    # It doesn't affect the sizing but does affect the LCA, this is a temporary fix. Also, we must
    # not include the energy necessary for the reserve as it does not contribute to the
    # functional unit
    datafile = oad.DataFile(DATA_FOLDER_PATH / input_file_name)

    total_fuel_mission = datafile["data:mission:sizing:fuel"].value[0]
    fuel_reserves = datafile["data:mission:sizing:main_route:reserve:fuel"].value[0]
    fuel_for_fu = total_fuel_mission - fuel_reserves

    # We only consider the emissions that were produced in phases of the flight which contribute to
    # the functional unit
    ratio_for_emissions = fuel_for_fu / total_fuel_mission

    problem.set_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_mission",
        units="kg",
        val=fuel_for_fu / 2.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_2:fuel_consumed_mission",
        units="kg",
        val=fuel_for_fu / 2.0,
    )

    # 1)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO"
        ].value[0]
        * ratio_for_emissions,
    )
    # 2)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO2",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO2"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO2"
        ].value[0]
        * ratio_for_emissions,
    )
    # 3)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:H2O",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:H2O"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:H2O"
        ].value[0]
        * ratio_for_emissions,
    )
    # 4)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:HC",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:HC"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:HC"
        ].value[0]
        * ratio_for_emissions,
    )
    # 5)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:NOx",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:NOx"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:NOx"
        ].value[0]
        * ratio_for_emissions,
    )
    # 6)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:SOx",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:SOx"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:SOx"
        ].value[0]
        * ratio_for_emissions,
    )
    # 7)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:lead",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:lead"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:lead"
        ].value[0]
        * ratio_for_emissions,
    )

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_club_lca_out.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_club_recipe():
    input_file_name = "pipistrel_club_out.xml"

    component = LCA(
        power_train_file_path=DATA_FOLDER_PATH / "pipistrel_club_assembly.yml",
        functional_unit="Flight hours",
        aircraft_lifespan_in_hours=True,
        component_level_breakdown=True,
        airframe_material="composite",
        delivery_method="train",
        electric_mix="french",
        normalization=True,
        weighting=True,
        ecoinvent_version="3.9.1",
    )

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    # The fuel stored in the tanks does not consider the fuel necessary for takeoff.
    # It doesn't affect the sizing but does affect the LCA, this is a temporary fix. Also, we must
    # not include the energy necessary for the reserve as it does not contribute to the
    # functional unit
    datafile = oad.DataFile(DATA_FOLDER_PATH / input_file_name)

    total_fuel_mission = datafile["data:mission:sizing:fuel"].value[0]
    fuel_reserves = datafile["data:mission:sizing:main_route:reserve:fuel"].value[0]
    fuel_for_fu = total_fuel_mission - fuel_reserves

    # We only consider the emissions that were produced in phases of the flight which contribute to
    # the functional unit
    ratio_for_emissions = fuel_for_fu / total_fuel_mission

    problem.set_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:fuel_consumed_mission",
        units="kg",
        val=fuel_for_fu / 2.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_2:fuel_consumed_mission",
        units="kg",
        val=fuel_for_fu / 2.0,
    )

    # 1)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO"
        ].value[0]
        * ratio_for_emissions,
    )
    # 2)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO2",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO2"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:CO2"
        ].value[0]
        * ratio_for_emissions,
    )
    # 3)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:H2O",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:H2O"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:H2O"
        ].value[0]
        * ratio_for_emissions,
    )
    # 4)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:HC",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:HC"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:HC"
        ].value[0]
        * ratio_for_emissions,
    )
    # 5)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:NOx",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:NOx"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:NOx"
        ].value[0]
        * ratio_for_emissions,
    )
    # 6)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:SOx",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:SOx"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:SOx"
        ].value[0]
        * ratio_for_emissions,
    )
    # 7)
    problem.set_val(
        "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:lead",
        units=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:lead"
        ].units,
        val=datafile[
            "data:environmental_impact:operation:sizing:he_power_train:high_rpm_ICE:ice_1:lead"
        ].value[0]
        * ratio_for_emissions,
    )

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_club_lca_out_recipe.xml"
    problem.write_outputs()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_lca_pipistrel_recipe():
    input_file_name = "pipistrel_out.xml"

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
    )

    ivc = local_get_indep_var_comp(list_inputs(component), input_file_name)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        component,
        ivc,
    )

    # The energy stored in the battery does not consider the fuel necessary for takeoff,
    # it doesn't affect the sizing but does affect the LCA, this is a temporary fix. Also, we must
    # not include the energy necessary for the reserve as it does not contribute to the
    # functional unit
    datafile = oad.DataFile(DATA_FOLDER_PATH / input_file_name)

    total_energy_mission = datafile["data:mission:sizing:energy"].value[0]
    energy_reserves = datafile["data:mission:sizing:main_route:reserve:energy"].value[0]
    energy_for_fu = total_energy_mission - energy_reserves

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:energy_consumed_mission",
        units="W*h",
        val=energy_for_fu / 2.0,
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:energy_consumed_mission",
        units="W*h",
        val=energy_for_fu / 2.0,
    )

    problem.run_model()
    problem.output_file_path = RESULTS_FOLDER_PATH / "pipistrel_electro_lca_out_recipe.xml"
    problem.write_outputs()
