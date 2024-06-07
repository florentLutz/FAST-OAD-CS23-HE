# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import os.path as pth

import pytest

import numpy as np

import networkx as nx
import matplotlib.pyplot as plt

from ..powertrain import FASTGAHEPowerTrainConfigurator
from ..exceptions import FASTGAHESingleSSPCAtEndOfLine, FASTGAHEImpossiblePair

YML_FILE = "sample_power_train_file.yml"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"
NB_POINTS_TEST = 10


def test_power_train_file_components_sizing():

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        components_name,
        components_name_id,
        components_type,
        components_om_type,
        components_position,
    ) = power_train_configurator.get_sizing_element_lists()

    # Check that they are not empty
    assert components_name
    assert components_name_id
    assert components_type
    assert components_om_type
    assert components_position


def test_power_train_file_components_performances():

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        components_name,
        components_name_id,
        components_type,
        components_om_type,
        components_options,
        components_connection_outputs,
        components_connection_inputs,
        components_promotes,
        sspc_list,
        sspc_default_state,
    ) = power_train_configurator.get_performances_element_lists()

    # Check that they are not empty
    assert components_name
    assert components_name_id
    assert components_type
    assert components_om_type
    assert components_options
    assert components_connection_outputs
    assert components_connection_inputs
    assert components_promotes
    assert sspc_list
    assert sspc_default_state


def test_power_train_file_components_slipstream():

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        components_name,
        components_name_id,
        components_type,
        components_om_type,
        components_slipstream_promotes,
        components_slipstream_flaps,
        components_slipstream_wing_lift,
    ) = power_train_configurator.get_slipstream_element_lists()

    # Check that they are not empty
    assert components_name
    assert components_name_id
    assert components_type
    assert components_om_type
    assert components_slipstream_promotes
    assert components_slipstream_flaps
    assert components_slipstream_wing_lift


def test_power_train_file_components_performances_sspc_last():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_sspc_last.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        cp_name,
        cp_id,
        cp_type,
        cp_om_type,
        cp_option,
        _,
        _,
        cp_promotes,
        _,
        _,
    ) = power_train_configurator.get_performances_element_lists()

    (components_name, components_name_id, _, _, _,) = power_train_configurator.enforce_sspc_last(
        cp_name,
        cp_id,
        cp_om_type,
        cp_option,
        cp_promotes,
    )

    # Check that they are not empty
    assert components_name == ["dc_bus_1", "dc_line_1", "dc_bus_2", "dc_sspc_1", "dc_sspc_2"]
    assert components_name_id == [
        "dc_bus_id",
        "harness_id",
        "dc_bus_id",
        "dc_sspc_id",
        "dc_sspc_id",
    ]


def test_power_train_file_connections():

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    for om_output, om_input in zip(
        power_train_configurator._components_connection_outputs,
        power_train_configurator._components_connection_inputs,
    ):

        print("[" + om_output + ", " + om_input + "]")


def test_power_train_file_direct_bus_battery_connection():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_direct_battery_bus_connection.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    # Battery voltage is no longer an output, rather, it becomes an input
    assert (
        "battery_pack_1.voltage_out" not in power_train_configurator._components_connection_outputs
    )
    assert "battery_pack_1.voltage_out" in power_train_configurator._components_connection_inputs

    assert (
        "battery_pack_1.dc_current_out"
        not in power_train_configurator._components_connection_inputs
    )
    assert (
        "battery_pack_1.dc_current_out" in power_train_configurator._components_connection_outputs
    )


def test_power_train_file_connections_splitter():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    for om_output, om_input in zip(
        power_train_configurator._components_connection_outputs,
        power_train_configurator._components_connection_inputs,
    ):

        print("[" + om_output + ", " + om_input + "]")


def test_power_train_watcher_path():

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    print(power_train_configurator.get_watcher_file_path())


def test_power_train_logic_check():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_sspc_fail.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    with pytest.raises(FASTGAHESingleSSPCAtEndOfLine):
        power_train_configurator.check_sspc_states({})

    sample_power_train_file_path = pth.join(pth.dirname(__file__), "data", YML_FILE)
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )
    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    # This state should be correct, let's check it
    actual_state = power_train_configurator.check_sspc_states(
        power_train_configurator._sspc_default_state
    )

    assert actual_state == power_train_configurator._sspc_default_state

    # Check that if we open one end, the other will be opened
    new_state = {"dc_sspc_1": True, "dc_sspc_2": False, "dc_sspc_3": True}
    state_check = {"dc_sspc_1": True, "dc_sspc_2": False, "dc_sspc_3": False}

    assert power_train_configurator.check_sspc_states(new_state) == state_check


def test_distance_from_propulsive_load():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    print("\n")
    distance_from_prop_load, _ = power_train_configurator.get_distance_from_propulsive_load()

    assert distance_from_prop_load["battery_pack_1"] == 9
    assert distance_from_prop_load["ice_1"] == 10

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    print("\n")
    power_train_configurator._get_components()
    power_train_configurator._get_connections()

    print("\n")
    distance_from_prop_load, _ = power_train_configurator.get_distance_from_propulsive_load()

    assert distance_from_prop_load["dc_bus_1"] == 3
    assert distance_from_prop_load["dc_bus_2"] == 3
    assert distance_from_prop_load["dc_bus_3"] == 3

    assert distance_from_prop_load["dc_bus_4"] == 7


def test_independent_voltage_subgraph():
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    # Small trick to be able to run this test from everywhere
    old_working_directory = os.getcwd()
    if "unit_tests" in old_working_directory:
        new_working_directory = os.path.dirname(os.path.dirname(old_working_directory))
    elif "powertrain_builder" in old_working_directory:
        new_working_directory = os.path.dirname(old_working_directory)
    else:
        new_working_directory = old_working_directory

    os.chdir(new_working_directory)

    print("\n")
    sub_graphs = power_train_configurator.get_graphs_connected_voltage()

    # Skip drawing if in GitHub actions
    if not IN_GITHUB_ACTIONS:
        for i, sub_graph in enumerate(sub_graphs):
            fig = plt.figure(figsize=(12, 9), dpi=80)

            nx.draw_kamada_kawai(sub_graph, ax=fig.add_subplot(), with_labels=True)
            fig.savefig("powertrain_builder/unit_tests/outputs/graph_" + str(i + 1) + ".png")

    os.chdir(old_working_directory)


def test_voltage_setter_list():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    voltage_setter_list = power_train_configurator._list_voltage_coherence_to_check()

    # In the .yml that serves for this test, there are 3 subgraphs, one with a DC/DC converter
    # and a rectifier (both sets voltage), one with a generator (which sets voltage) and one with
    # nothing that sets the voltage
    assert [] in voltage_setter_list
    assert ["dc_dc_converter_1_out", "rectifier_1_out"] in voltage_setter_list
    assert ["generator_1_out"] in voltage_setter_list


def test_wing_punctual_mass_identification():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter_position.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        wing_punctual_mass_list,
        wing_punctual_mass_type_list,
        _,
    ) = power_train_configurator.get_wing_punctual_mass_element_list()

    assert "dc_dc_converter_1" in wing_punctual_mass_list
    assert "DC_DC_converter" in wing_punctual_mass_type_list
    assert "rectifier_1" in wing_punctual_mass_list
    assert "rectifier" in wing_punctual_mass_type_list
    assert "generator_1" in wing_punctual_mass_list
    assert "generator" in wing_punctual_mass_type_list
    assert "ice_1" in wing_punctual_mass_list
    assert "ICE" in wing_punctual_mass_type_list

    assert not ("battery_pack_1" in wing_punctual_mass_list)

    assert not ("propeller_1" in wing_punctual_mass_list)
    assert not ("motor_1" in wing_punctual_mass_list)
    assert not ("inverter_1" in wing_punctual_mass_list)
    assert not ("dc_sspc_1" in wing_punctual_mass_list)
    assert not ("dc_bus_1" in wing_punctual_mass_list)
    assert not ("dc_sspc_2" in wing_punctual_mass_list)
    assert not ("dc_line_1" in wing_punctual_mass_list)
    assert not ("dc_sspc_3" in wing_punctual_mass_list)
    assert not ("dc_splitter_1" in wing_punctual_mass_list)


def test_wing_distributed_mass_identification():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter_position.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        wing_distributed_mass_list,
        wing_distributed_mass_type_list,
        _,
    ) = power_train_configurator.get_wing_distributed_mass_element_list()

    assert len(wing_distributed_mass_list) == 1
    assert "battery_pack_1" in wing_distributed_mass_list
    assert "battery_pack" in wing_distributed_mass_type_list


def test_no_distributed_mass():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_propeller_symmetrical.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        wing_distributed_mass_list,
        _,
        pairs_list,
    ) = power_train_configurator.get_wing_distributed_mass_element_list()

    assert not wing_distributed_mass_list
    assert not pairs_list


def test_wing_punctual_mass_symmetrical():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_propeller_symmetrical.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        wing_punctual_mass_list,
        _,
        symmetrical_pair_list,
    ) = power_train_configurator.get_wing_punctual_mass_element_list()

    assert "propeller_1" in wing_punctual_mass_list
    assert "propeller_2" in wing_punctual_mass_list

    assert ["propeller_1", "propeller_2"] in symmetrical_pair_list
    assert ["propeller_2", "propeller_1"] not in symmetrical_pair_list


def test_wing_distributed_mass_symmetrical():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_battery_symmetrical.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    (
        wing_distributed_mass_list,
        _,
        symmetrical_pair_list,
    ) = power_train_configurator.get_wing_distributed_mass_element_list()

    assert "propeller_1" not in wing_distributed_mass_list
    assert "propeller_2" not in wing_distributed_mass_list

    assert "battery_pack_1" in wing_distributed_mass_list
    assert "battery_pack_2" in wing_distributed_mass_list

    assert ["propeller_1", "propeller_2"] not in symmetrical_pair_list
    assert ["battery_pack_1", "battery_pack_2"] in symmetrical_pair_list
    assert ["battery_pack_2", "battery_pack_1"] not in symmetrical_pair_list


def test_bad_pair():

    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_bad_pair.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    with pytest.raises(FASTGAHEImpossiblePair) as e_info:

        # This should prompt the error
        (
            _,
            _,
            _,
        ) = power_train_configurator.get_wing_punctual_mass_element_list()

    assert (
        e_info.value.args[0]
        == "Cannot pair propeller_1 with propeller_3 because propeller_3 does not exist. Valid pair choice are among the following list: propeller_1, propeller_2. \nBest regards."
    )


def test_mass_variation_identification():

    # Mass should vary
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    assert power_train_configurator.will_aircraft_mass_vary()

    # Mass shouldn't vary
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    assert not power_train_configurator.will_aircraft_mass_vary()


def test_identification_unconsumable_source():

    # Mass should vary but there also is an unconsumable energy source
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    assert power_train_configurator.will_aircraft_mass_vary()
    assert power_train_configurator.has_fuel_non_consumable_energy_source()

    # Mass shouldn't vary and there is an unconsumable energy source
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    assert not power_train_configurator.will_aircraft_mass_vary()
    assert power_train_configurator.has_fuel_non_consumable_energy_source()

    # Mass should vary and there is no unconsumable energy sources
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_full_turbo.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    assert power_train_configurator.will_aircraft_mass_vary()
    assert not power_train_configurator.has_fuel_non_consumable_energy_source()


def test_get_power_on_each_node():

    # Very simple power train
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    propulsive_power_dict = {"propeller_1": np.array([50e3])}

    power_at_each_node = power_train_configurator.get_power_to_set(
        inputs=None, propulsive_power_dict=propulsive_power_dict
    )[0]

    # Considering the components for the test, the efficiency should be equal to
    # eta = 0.8 * 0.95 * 0.98 * 0.99 * 1.0 * 0.99 * 0.98 * 0.99 * 1.0 * 0.98 = 0.69406061887008
    assert power_at_each_node["propeller_1_out"] / power_at_each_node[
        "battery_pack_1_out"
    ] == pytest.approx(0.694, abs=1e-3)

    # With a splitter
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    propulsive_power_dict = {"propeller_1": np.array([50e3])}

    # Usually the inputs will be a OpenMDAO vector but for the sake of testing we can trick him
    # to use a plain ol' dictionary
    # Here the splitter is in power share mode so we have to give a power value
    inputs = {
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:power_share": np.array([50e3])
    }
    power_at_each_node = power_train_configurator.get_power_to_set(
        inputs=inputs, propulsive_power_dict=propulsive_power_dict
    )[0]

    # Since there is a splitter we will check that the two input are indeed created
    assert "dc_splitter_1_in_1" in list(power_at_each_node.keys())
    assert "dc_splitter_1_in_2" in list(power_at_each_node.keys())

    # We will check that the power we set as power share goes in the right place
    assert power_at_each_node["dc_splitter_1_in_1"] == pytest.approx(50e3, rel=1e-6)
    # Also, since the propulsive power is equal to the power, considering efficiencies,
    # the secondary input should not be empty
    assert power_at_each_node["dc_splitter_1_in_2"] > 0

    # Then we try with a bus with uniform input
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    propulsive_power_dict = {
        "propeller_1": np.array([25e3]),
        "propeller_2": np.array([50e3]),
        "propeller_3": np.array([25e3]),
    }

    power_at_each_node = power_train_configurator.get_power_to_set(
        inputs=None, propulsive_power_dict=propulsive_power_dict
    )[0]

    # Same components so the branch efficiency should be the same, and since we put twice as much
    # power in the middle branch the following should hold
    assert power_at_each_node["dc_sspc_1_3_in"] == power_at_each_node["dc_sspc_3_3_in"]
    assert power_at_each_node["dc_sspc_1_3_in"] * 2.0 == power_at_each_node["dc_sspc_2_3_in"]

    assert (
        power_at_each_node["dc_bus_4_out"]
        == power_at_each_node["dc_sspc_1_3_in"]
        + power_at_each_node["dc_sspc_2_3_in"]
        + power_at_each_node["dc_sspc_3_3_in"]
    )

    # Then we try with a bus with a non-uniform input
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop_shorter_nose.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    propulsive_power_dict = {
        "propeller_1": np.array([25e3]),
        "propeller_2": np.array([50e3]),
        "propeller_3": np.array([25e3]),
    }

    power_at_each_node = power_train_configurator.get_power_to_set(
        inputs=None, propulsive_power_dict=propulsive_power_dict
    )[0]

    # Less components in the outer branch so better efficiency :)
    assert power_at_each_node["dc_sspc_3_3_in"] == power_at_each_node["dc_sspc_1_3_in"]
    assert (
        power_at_each_node["propeller_3_out"] / power_at_each_node["dc_sspc_3_3_in"]
        > power_at_each_node["propeller_2_out"] / power_at_each_node["dc_sspc_2_3_in"]
    )

    assert (
        power_at_each_node["dc_bus_4_out"]
        == power_at_each_node["dc_sspc_1_3_in"]
        + power_at_each_node["dc_sspc_2_3_in"]
        + power_at_each_node["dc_sspc_3_3_in"]
    )

    # Then we try with a bus and a splitter
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter_and_bus.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    # No mode defined in the
    inputs = {
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_0:power_split": np.array([50.0])
    }

    power_at_each_node = power_train_configurator.get_power_to_set(
        inputs=inputs, propulsive_power_dict=propulsive_power_dict
    )[0]

    # Assert that they exist and have the same value (since we have a 50% split)
    assert power_at_each_node["battery_pack_2_out"] == power_at_each_node["battery_pack_1_out"]

    # Assert that they are properly order

    assert power_at_each_node["dc_bus_0_in"] < power_at_each_node["dc_splitter_0_out"]

    # Test with multiple independent power train
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_tri_prop_two_chainz.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )
    power_at_each_node = power_train_configurator.get_power_to_set(
        inputs=None, propulsive_power_dict=propulsive_power_dict
    )[0]

    assert "propeller_1_out" in list(power_at_each_node.keys())
    assert "propeller_3_out" in list(power_at_each_node.keys())
    assert "propeller_2_out" in list(power_at_each_node.keys())

    # Efficiency in the first chain should be smaller than the second chain since there are more
    # components
    assert (
        power_at_each_node["propeller_3_out"] / power_at_each_node["battery_pack_1_out"]
        < power_at_each_node["propeller_2_out"] / power_at_each_node["battery_pack_2_out"]
    )


def test_get_power_on_each_node_unequal_branches():

    # Very simple power train
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_quad_prop.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    propulsive_power_dict = {
        "propeller_0": np.array([50e3]),
        "propeller_1": np.array([50e3]),
        "propeller_2": np.array([50e3]),
        "propeller_3": np.array([50e3]),
    }

    power_at_each_node = power_train_configurator.get_power_to_set(
        inputs=None, propulsive_power_dict=propulsive_power_dict
    )[0]

    # Considering the components for the test the power at the input of the 4th bus should be
    # 2 * 50 / 0.8 / 0.98 / 0.95 / 0.98 / 1.0 / 0.98 + 2 * 50 / 0.8 / 0.95 / 0.98 / 1.0 / 0.98
    assert power_at_each_node["dc_bus_4_in"] == pytest.approx(276.8e3, rel=1e-3)


def test_power_to_set():

    # Very simple power train
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    propulsive_power_dict = {"propeller_1": np.array([50e3])}

    powers_to_set = power_train_configurator.get_power_to_set(
        inputs=None, propulsive_power_dict=propulsive_power_dict
    )[1]

    assert "propeller_1.shaft_power_in" in list(powers_to_set.keys())
    assert "motor_1.active_power" in list(powers_to_set.keys())
    assert "dc_sspc_1.power_flow" in list(powers_to_set.keys())
    assert "dc_dc_converter_1.converter_relation.power_rel" in list(powers_to_set.keys())

    assert powers_to_set["dc_sspc_1.power_flow"] == pytest.approx(67810.218, abs=1e-3)


def test_current_to_set():
    # Very simple power train
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    propulsive_power_dict = {"propeller_1": np.array([50e3])}
    inputs = {
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:voltage_out_target_mission": np.array(
            [400.0]
        ),
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:module:number_cells": np.array(
            [96.0]
        ),
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:module:number_cells": np.array(
            [96.0]
        ),
    }

    current_to_set = power_train_configurator.get_current_to_set(
        inputs=inputs, propulsive_power_dict=propulsive_power_dict, number_of_points=1
    )

    assert "motor_1.ac_current_rms_in" in current_to_set.keys()
    assert "motor_1.ac_current_rms_in_one_phase" in current_to_set.keys()

    assert (
        current_to_set["motor_1.ac_current_rms_in"]
        == 3.0 * current_to_set["motor_1.ac_current_rms_in_one_phase"]
    )

    assert "inverter_1.dc_current_in" in current_to_set.keys()

    assert "dc_line_1.dc_current_one_cable" in current_to_set.keys()

    assert "dc_dc_converter_1.dc_current_in" in current_to_set.keys()
    assert "dc_dc_converter_1.dc_current_out" in current_to_set.keys()

    assert "dc_sspc_1.dc_current_in" in current_to_set.keys()
    assert "dc_sspc_2.dc_current_in" in current_to_set.keys()
    assert "dc_sspc_3.dc_current_in" in current_to_set.keys()

    # Then we try with a bus and a splitter
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter_and_bus.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    # No mode defined in the
    inputs = {
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_0:power_split": np.array([50.0]),
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:voltage_out_target_mission": np.array(
            [400.0]
        ),
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:module:number_cells": np.array(
            [96.0]
        ),
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:module:number_cells": np.array(
            [96.0]
        ),
    }

    propulsive_power_dict = {
        "propeller_1": np.array([25e3]),
        "propeller_2": np.array([50e3]),
        "propeller_3": np.array([25e3]),
    }

    current_to_set = power_train_configurator.get_current_to_set(
        inputs=inputs, propulsive_power_dict=propulsive_power_dict, number_of_points=1
    )

    for current in list(current_to_set.keys()):
        assert "dc_splitter_1" not in current


def test_control_parameter_identification():

    # Mass should vary
    sample_power_train_file_path = pth.join(
        pth.dirname(__file__), "data", "sample_power_train_file_splitter.yml"
    )
    power_train_configurator = FASTGAHEPowerTrainConfigurator(
        power_train_file_path=sample_power_train_file_path
    )

    control_parameter_list = power_train_configurator.get_control_parameter_list()

    assert control_parameter_list
    assert (
        "data:propulsion:he_power_train:propeller:propeller_1:rpm_mission" in control_parameter_list
    )
    assert (
        "data:propulsion:he_power_train:inverter:inverter_1:junction_temperature_mission"
        in control_parameter_list
    )
    assert (
        "data:propulsion:he_power_train:rectifier:rectifier_1:voltage_out_target_mission"
        in control_parameter_list
    )
