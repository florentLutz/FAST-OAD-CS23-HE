# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import pytest

from ..powertrain import FASTGAHEPowerTrainConfigurator
from ..exceptions import FASTGAHESingleSSPCAtEndOfLine

YML_FILE = "sample_power_train_file.yml"


def test_power_train_file_components():

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
