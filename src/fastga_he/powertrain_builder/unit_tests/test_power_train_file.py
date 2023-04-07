# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os
import os.path as pth

import pytest

import networkx as nx
import matplotlib.pyplot as plt

from ..powertrain import FASTGAHEPowerTrainConfigurator
from ..exceptions import FASTGAHESingleSSPCAtEndOfLine

YML_FILE = "sample_power_train_file.yml"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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

    print("\n")
    sub_graphs = power_train_configurator.get_graphs_connected_voltage()

    # Skip drawing if in GitHub actions
    if not IN_GITHUB_ACTIONS:
        for i, sub_graph in enumerate(sub_graphs):
            fig = plt.figure(figsize=(12, 9), dpi=80)

            nx.draw_circular(sub_graph, ax=fig.add_subplot(), with_labels=True)
            fig.savefig("powertrain_builder/unit_tests/outputs/graph_" + str(i + 1) + ".png")
