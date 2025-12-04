"""
Module for the construction of all the groups necessary for the proper interaction of the
power train module with the aircraft sizing modules from FAST-OAD-GA based on the power train file.
"""
# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import copy
import json
import logging
import sys
import os.path as pth
import pathlib
import time

from abc import ABC
from importlib.resources import open_text
from typing import Tuple, List, Dict

import re

import numpy as np
from jsonschema import validate
from ruamel.yaml import YAML

import networkx as nx

from .exceptions import (
    FASTGAHEUnknownComponentID,
    FASTGAHEUnknownOption,
    FASTGAHEInvalidOptionDefinition,
    FASTGAHESingleSSPCAtEndOfLine,
    FASTGAHEImpossiblePair,
    FASTGAHEIncoherentVoltage,
    FASTGAHEComponentConnectionError,
    FASTGAHECriticalComponentMissingError,
    FASTGAHEInputCountError,
    FASTGAHEOutputCountError,
    FASTGAHEComponentsNotIdentified,
)

from . import resources

_LOGGER = logging.getLogger(__name__)  # Logger for this module

JSON_SCHEMA_NAME = "power_train.json"

KEY_TITLE = "title"
KEY_PT_COMPONENTS = "power_train_components"
KEY_PT_CONNECTIONS = "component_connections"
KEY_PT_WATCHER = "watcher_file_path"

PT_DATA_PREFIX = "data:propulsion:he_power_train:"

PROMOTION_FROM_MISSION = {
    "thrust": "N",
    "altitude": "m",
    "time_step": "s",
    "true_airspeed": "m/s",
    "exterior_temperature": "degK",
}
# TODO: Find a more generic way to do that, as an attributes in registered_components.py maybe ?
TYPE_TO_FUEL = {
    "turboshaft": "jet_fuel",
    "ICE": "avgas",
    "high_rpm_ICE": "avgas",
    "PEMFC_stack": "hydrogen",
}
ELECTRICITY_STORAGE_TYPES = ["battery_pack"]
DEFAULT_VOLTAGE_VALUE = 737.800


class FASTGAHEPowerTrainConfigurator:
    """
    Class for the configuration of the components necessary for the performances and sizing of the
    power train.

    :param power_train_file_path: if provided, power train will be read directly from it
    """

    _cache = {}

    def __init__(self, power_train_file_path=None):
        self._power_train_file = None

        self._serializer = _YAMLSerializer()

        # Contains the id of the components
        self._components_id = None

        # Contains the position of the components
        self._components_position = None

        # Contains the name of the component as it will be found in the input/output file to
        # contain the data. Will also be used as subsystem name
        self._components_name = None

        # Contains the name of the options used to provide the names in self._components_name
        self._components_name_id = None

        # Contains the suffix of the component added to Performances and Sizing, will be used to
        # instantiate the subsystems
        self._components_om_type = None

        # Contains the type of the component as it will be found in the input/output file to
        # contain the data
        self._components_type = None

        # Contains a special tag on the class of element as some may need specific assemblers to
        # work such as propulsor
        self._components_type_class = None

        # Contains the options of the component which will be given during object instantiation
        self._components_options = None

        # Contains the list of aircraft inputs that are necessary to promote in the performances
        # modules for the code to work
        self._components_promotes = None

        # Contains the list of aircraft inputs that are necessary to promote in the slipstream
        # modules for the code to work
        self._components_slipstream_promotes = None

        # Contains the list of variables that needs to be promoted from the performances
        # computations to slipstream computation
        self._components_performances_to_slipstream = None

        # Contains a list with, for each component, a boolean telling whether the component
        # needs the flaps position for the computation of the slipstream effects
        self._components_slipstream_flaps = None

        # Contains a list with, for each component, a boolean telling whether the component
        # lift increase is added to the wing. Will be used for the increase in induced drag
        self._components_slipstream_wing_lift = None

        # Contains a basic list of the connections in the power train, with no processing whatsoever
        self._connection_list = None

        # Contains the list of all outputs (in the OpenMDAO sense of the term) needed to make the
        # connections between components
        self._components_connection_outputs = None

        # Contains the list of all inputs (in the OpenMDAO sense of the term) needed to make the
        # connections between components
        self._components_connection_inputs = None

        # Contains a list, for each component, of all the variables that will be monitored in the
        # performances watcher of the power train, meaning this should be a list of list
        self._components_perf_watchers = None

        # Contains a list, for each component, of all the variables in the slipstream computation
        # that will be monitored in the performances watcher of the power train, meaning this
        # should be a list of list
        self._components_slipstream_perf_watchers = None

        # Contains a list of all pair of components which are symmetrical on the y axis with
        # respect to the fuselage center line. This is for now intended for the computation of the
        # loads on the wing to avoid accounting twice for the components as the wing mass will be
        # computed as twice the weight of a half-wing
        self._components_symmetrical_pairs = None

        # Contains the list of all boolean telling whether the components will make the
        # aircraft weight vary during flight
        self._components_makes_mass_vary = None

        # Contains the list of all boolean telling whether the components are energy
        # sources that do not make the aircraft vary (ergo they will have a non-nil unconsumable
        # energy)
        self._source_does_not_make_mass_vary = None

        # Contains the list of an initial guess of the component's efficiency. Is used to compute
        # the initial of the currents and power of each component
        self._components_efficiency = None

        # Contains the list of control parameters name for each component. Is used to detect
        # them in cas we want to give them a different name during the mission
        self._components_control_parameters = None

        # Because of their very peculiar role, we will scan the architecture for any SSPC defined
        # by the user and whether they are at the output of a bus, because a specific
        # option needs to be turned on in this was
        self._sspc_list = {}

        # Contains the default state of the SSPC, will be used if other states are not specified
        # as an option of the performances group
        self._sspc_default_state = {}

        # After construction contains a graph (graph theory) with all components and their
        # connection. It will for instance allow to check if a cable has SSPC's at both its end
        # or check if a propulsor is not connected to a power source, in which case, we should not
        # be able to require thrust from him (will come later)
        self._connection_graph = None

        # Contains the results of the function that sets the power in the graphs, is declared as
        # an attribute to avoid having to recompute everything
        self._power_at_each_node = None

        # Contains the results of the function that sets the voltage in the graphs, is declared as
        # an attribute to avoid having to recompute everything
        self._voltage_at_each_node = None

        if power_train_file_path:
            self.load(power_train_file_path)

    def load(self, power_train_file):
        """
        Reads the power train definition

        :param power_train_file: Path to the file to open.
        """
        start_time = time.perf_counter()

        self._power_train_file = pth.abspath(power_train_file)

        if not FASTGAHEPowerTrainConfigurator._cache.get(self._power_train_file):
            FASTGAHEPowerTrainConfigurator._cache[self._power_train_file] = {}

        self._serializer = _YAMLSerializer()
        self._serializer.read(self._power_train_file)

        # Syntax validation
        with open_text(resources, JSON_SCHEMA_NAME) as json_file:
            json_schema = json.loads(json_file.read())
        validate(self._serializer.data, json_schema)

        for key in self._serializer.data:
            if key not in json_schema["properties"].keys():
                _LOGGER.warning('Power train file: "%s" is not a FAST-OAD-GA-HE key.', key)

        end_time = time.perf_counter()

        if not FASTGAHEPowerTrainConfigurator._cache[self._power_train_file].get("load_time"):
            FASTGAHEPowerTrainConfigurator._cache[self._power_train_file]["load_time"] = (
                end_time - start_time
            )

    def get_watcher_file_path(self):
        """
        Returns the path to where the performance watch file will be. If name is not absolute
        we complete it.
        """

        watcher_file_path = self._serializer.data.get(KEY_PT_WATCHER)
        if watcher_file_path:
            if not pth.isabs(watcher_file_path):
                return pth.join(pth.dirname(self._power_train_file), watcher_file_path)
            else:
                return watcher_file_path
        else:
            return None

    def _get_components(self):
        # We will work under the assumption that is one list is empty, all are hence only one if
        # statement. This allows us to know whether re-triggering the identification of
        # components is necessary

        start_time = time.perf_counter()

        if self._components_id is None:
            self._generate_components_list()

        end_time = time.perf_counter()

        if not FASTGAHEPowerTrainConfigurator._cache[self._power_train_file].get(
            "get_component_time"
        ):
            FASTGAHEPowerTrainConfigurator._cache[self._power_train_file]["get_component_time"] = (
                end_time - start_time
            )

    def _generate_components_list(self):
        components_list = self._serializer.data.get(KEY_PT_COMPONENTS)

        components_id = []
        components_position = []
        components_name_id_list = []
        components_type_list = []
        components_om_type_list = []
        components_options_list = []
        components_promote_list = []
        components_slip_promote_list = []
        components_perf_to_slip_list = []
        components_type_class_list = []
        components_perf_watchers_list = []
        components_slipstream_perf_watchers_list = []
        components_slipstream_needs_flaps = []
        components_slipstream_wing_lift = []
        components_symmetrical_pairs = []
        components_makes_mass_vary = []
        source_does_not_make_mass_vary = []
        components_efficiency = []
        components_control_parameter = []

        # Doing it like that allows us to have the names of the components before we start the
        # loop, which I'm gonna use to check if the pairs are valid
        components_name_list = list(components_list.keys())

        for component_name in components_list:
            component = copy.deepcopy(components_list[component_name])
            component_id = component["id"]
            components_id.append(component_id)
            if component_id not in resources.KNOWN_ID:
                raise FASTGAHEUnknownComponentID(
                    component_id + " is not a known ID of a power train component"
                )
            if "position" in component:
                component_position = component["position"]
                components_position.append(component_position)
            else:
                components_position.append("")

            if "symmetrical" in component:
                component_symmetrical = component["symmetrical"]

                if component_symmetrical not in components_name_list:
                    raise FASTGAHEImpossiblePair(
                        "Cannot pair "
                        + component_name
                        + " with "
                        + component_symmetrical
                        + " because "
                        + component_symmetrical
                        + " does not exist. Valid pair choice are among the following list: "
                        + ", ".join(components_name_list)
                        + ". \nBest regards."
                    )

                # We sort the pair to ensure that if the pair is already there because the
                # symmetrical tag is defined twice (propeller1 is symmetrical to propeller2 and
                # propeller2 is symmetrical to propeller1) it will have the same name, and we
                # don't have to register it twice.
                sorted_pair = sorted([component_name, component_symmetrical])

                if sorted_pair not in components_symmetrical_pairs:
                    components_symmetrical_pairs.append(sorted_pair)
                    # We don't put an else because as opposed to options, we don't expected all
                    # components to have symmetrical tag

            if component_id == "fastga_he.pt_component.dc_sspc":
                # Create a dictionary with SSPC name and a tag to see if they are at bus output
                # or not, it will be set at False by default but be changed later on

                self._sspc_list[component_name] = False

                if "options" in component.keys():
                    if "closed_by_default" in component["options"]:
                        self._sspc_default_state[component_name] = component["options"][
                            "closed_by_default"
                        ]
                    else:
                        self._sspc_default_state[component_name] = True
                else:
                    self._sspc_default_state[component_name] = True

            components_name_id_list.append(resources.DICTIONARY_CN_ID[component_id])
            components_type_list.append(resources.DICTIONARY_CT[component_id])
            components_om_type_list.append(resources.DICTIONARY_CN[component_id])
            components_promote_list.append(resources.DICTIONARY_PT[component_id])
            components_slip_promote_list.append(resources.DICTIONARY_SPT[component_id])
            components_perf_to_slip_list.append(resources.DICTIONARY_PTS[component_id])
            components_type_class_list.append(resources.DICTIONARY_CTC[component_id])
            components_perf_watchers_list.append(resources.DICTIONARY_MP[component_id])
            components_slipstream_perf_watchers_list.append(resources.DICTIONARY_SMP[component_id])
            components_slipstream_needs_flaps.append(resources.DICTIONARY_SFR[component_id])
            components_slipstream_wing_lift.append(resources.DICTIONARY_SWL[component_id])
            components_makes_mass_vary.append(resources.DICTIONARY_VARIES_MASS[component_id])
            source_does_not_make_mass_vary.append(resources.DICTIONARY_VARIESN_T_MASS[component_id])
            components_efficiency.append(resources.DICTIONARY_ETA[component_id])
            components_control_parameter.append(resources.DICTIONARY_CTRL_PARAM[component_id])

            if "options" in component.keys():
                # SSPC is treated above, this way of doing things however makes no other option
                # for SSPC can be set, may need to be changed
                if component_id != "fastga_he.pt_component.dc_sspc":
                    components_options_list.append(component["options"])

                    # While we are at it, we also check that we have the right options and with the
                    # right names

                    if set(component["options"].keys()) != set(
                        resources.DICTIONARY_ATT[component_id]
                    ):
                        raise FASTGAHEUnknownOption(
                            "Component "
                            + component_id
                            + " does not have all options declare or they "
                            "have an erroneous name. The following options should be declared: "
                            + ", ".join(resources.DICTIONARY_ATT[component_id])
                        )
                else:
                    components_options_list.append(None)

            else:
                components_options_list.append(None)

        self._components_id = components_id
        self._components_position = components_position
        self._components_name = components_name_list
        self._components_name_id = components_name_id_list
        self._components_type = components_type_list
        self._components_om_type = components_om_type_list
        self._components_options = components_options_list
        self._components_promotes = components_promote_list
        self._components_slipstream_promotes = components_slip_promote_list
        self._components_performances_to_slipstream = components_perf_to_slip_list
        self._components_type_class = components_type_class_list
        self._components_perf_watchers = components_perf_watchers_list
        self._components_slipstream_perf_watchers = components_slipstream_perf_watchers_list
        self._components_slipstream_flaps = components_slipstream_needs_flaps
        self._components_slipstream_wing_lift = components_slipstream_wing_lift
        self._components_symmetrical_pairs = components_symmetrical_pairs
        self._components_makes_mass_vary = components_makes_mass_vary
        self._source_does_not_make_mass_vary = source_does_not_make_mass_vary
        self._components_efficiency = components_efficiency
        self._components_control_parameters = components_control_parameter

    def _get_connections(self):
        """
        This function inspects all the connections detected in the power train file and prepare
        the list necessary to do the connections in the performance file.

        The _get_components method must be run beforehand.
        """

        start_time = time.perf_counter()
        # This should do nothing if it has already been run.
        self._get_components()

        connections_list = self._serializer.data.get(KEY_PT_CONNECTIONS)

        if not self._check_existing_connection_cache_instance():
            self._check_connection(connections_list)
            self._add_connection_check_cache_instance()
            _LOGGER.info("Powertrain components' connections checked.")

        self._connection_list = connections_list

        # Create a dictionary to translate component name back to component_id to identify
        # outputs and inputs in each case
        translator = dict(zip(self._components_name, self._components_id))

        openmdao_output_list = []
        openmdao_input_list = []

        for connection in connections_list:
            # Check in case the source or target is not a string but an array, meaning we are
            # dealing with a component which might have multiple inputs/outputs (bus, gearbox,
            # splitter, ...)
            if type(connection["source"]) is str:
                source_name = connection["source"]
                source_id = translator[source_name]
                source_number = ""
                source_inputs = resources.DICTIONARY_IN[source_id]
            else:
                source_name = connection["source"][0]
                source_id = translator[source_name]
                source_number = str(connection["source"][1])
                source_inputs = resources.DICTIONARY_IN[source_id]

            if type(connection["target"]) is str:
                target_name = connection["target"]
                target_id = translator[target_name]
                target_number = ""
                target_outputs = resources.DICTIONARY_OUT[target_id]
            else:
                target_name = connection["target"][0]
                target_id = translator[target_name]
                target_number = str(connection["target"][1])
                target_outputs = resources.DICTIONARY_OUT[target_id]

            # First we check if we are dealing with an SSPC, because of their nature explained
            # more in depth in the perf_voltage_out module, they will get a special treatment.
            # They will always be connected to a bus and even more, their 'input' side will
            # always be connected to a bus.

            # If SSPC is source and connected to a bus there should be no worries, else we need a
            # special treatment since the "input" side of the SSPC should be connected to the
            # bus. Same reasoning apply for splitter since they are a type of bus.
            if source_id == "fastga_he.pt_component.dc_sspc" and not (
                target_id == "fastga_he.pt_component.dc_bus"
                or target_id == "fastga_he.pt_component.dc_splitter"
            ):
                # We reverse the SSPC inputs and outputs
                source_inputs = resources.DICTIONARY_OUT[source_id]

            # Same reasoning here, we just have to reverse the SSPC inputs and outputs
            elif target_id == "fastga_he.pt_component.dc_sspc" and (
                source_id == "fastga_he.pt_component.dc_bus"
                or source_id == "fastga_he.pt_component.dc_splitter"
            ):
                # We reverse the SSPC outputs and input
                target_outputs = resources.DICTIONARY_IN[target_id]

            # Because we need to know if the SSPC is at a bus output for the model to work,
            # this check is necessary
            if source_id == "fastga_he.pt_component.dc_sspc" and (
                target_id == "fastga_he.pt_component.dc_bus"
                or target_id == "fastga_he.pt_component.dc_splitter"
            ):
                self._sspc_list[source_name] = True

            # The possibility to connect a battery or a PEMFC stack directly to a bus has been
            # added. However, to make it backward compatible (whatever it means today because I
            # have no users) and to impart less burden during the writing of the pt file,
            # we won't ask the user to set the option accordingly, rather, we will do it here.

            if (
                target_id == "fastga_he.pt_component.battery_pack"
                or target_id == "fastga_he.pt_component.pemfc_stack"
            ) and (
                source_id == "fastga_he.pt_component.dc_bus"
                or source_id == "fastga_he.pt_component.dc_splitter"
                or source_id == "fastga_he.pt_component.dc_sspc"
            ):
                # First we'll check if the option has already been set or no, just to avoid
                # losing time

                target_index = self._components_name.index(target_name)
                target_option = self._components_options[target_index]

                if not target_option:
                    self._components_options[target_index] = {"direct_bus_connection": True}

                current_outputs = resources.DICTIONARY_OUT[target_id]

                target_outputs = []
                for current_output in current_outputs:
                    target_outputs.append(tuple(reversed(current_output)))

            # Compressor connection for the PEMFC stack. This if condition won't be activated until
            # the implementation of the compressor component.

            # if (
            #     target_id == "fastga_he.pt_component.pemfc_stack"
            #     and source_id == "fastga_he.pt_component.compressor"
            # ):
            #     # First we'll check if the option has already been set or no, just to avoid
            #     # losing time
            #
            #     target_index = self._components_name.index(target_name)
            #     target_option = self._components_options[target_index]
            #
            #     if not target_option:
            #         self._components_options[target_index] = {"compressor_connection": True}
            #
            #     current_outputs = resources.DICTIONARY_OUT[target_id]
            #
            #     target_outputs = []
            #     for current_output in current_outputs:
            #         target_outputs.append(tuple(reversed(current_output)))

            for system_input, system_output in zip(source_inputs, target_outputs):
                if system_input[0]:
                    if system_input[0][-1] == "_":
                        system_input_str = system_input[0] + source_number
                    else:
                        system_input_str = system_input[0]

                    if system_output[1][-1] == "_":
                        system_output_str = system_output[1] + target_number
                    else:
                        system_output_str = system_output[1]

                    openmdao_input_list.append(source_name + "." + system_input_str)
                    openmdao_output_list.append(target_name + "." + system_output_str)

                else:
                    if system_input[1][-1] == "_":
                        system_input_str = system_input[1] + source_number
                    else:
                        system_input_str = system_input[1]

                    if system_output[0][-1] == "_":
                        system_output_str = system_output[0] + target_number
                    else:
                        system_output_str = system_output[0]

                    openmdao_input_list.append(target_name + "." + system_output_str)
                    openmdao_output_list.append(source_name + "." + system_input_str)

        self._components_connection_outputs = openmdao_output_list
        self._components_connection_inputs = openmdao_input_list

        end_time = time.perf_counter()

        if not FASTGAHEPowerTrainConfigurator._cache[self._power_train_file].get(
            "get_connection_time"
        ):
            FASTGAHEPowerTrainConfigurator._cache[self._power_train_file]["get_connection_time"] = (
                end_time - start_time
            )

    def _check_connection(self, connections_list):
        """
        This function ensures that all the connections defined in the powertrain respect the
        components connection limit. It also checks that no components are left unconnected and
        ensure at least one propulsor and one energy storing device is defined.

        The _get_components method must be run beforehand.
        """

        # This should do nothing if it has already been run.
        self._get_components()

        propulsor_component = []
        aux_load_component = []
        energy_storage_component = []
        one_to_one_component = []
        connector_component = []
        connector_option = []
        connector_type = []

        for components_name, components_options, components_type_class, components_type in zip(
            self._components_name,
            self._components_options,
            self._components_type_class,
            self._components_type,
        ):
            if components_type_class == "propulsor":
                propulsor_component.append(components_name)

            elif components_type_class == "tank" or components_type_class == "source":
                energy_storage_component.append(components_name)

            elif "propulsive_load" in components_type_class:
                one_to_one_component.append(components_name)

            elif components_type_class == "load":
                aux_load_component.append(components_name)

            elif components_type_class == "connector":
                connector_component.append(components_name)
                connector_option.append(components_options)
                connector_type.append(components_type)

        if not propulsor_component:
            raise FASTGAHECriticalComponentMissingError("Propulsor missing!")

        if not energy_storage_component:
            raise FASTGAHECriticalComponentMissingError("Storage tank or battery missing!")

        (
            one_to_one_component,
            one_to_many_component,
            many_to_one_component,
            many_to_many_component,
            many_to_many_input_count,
            many_to_many_output_count,
            many_to_one_input_count,
            one_to_many_output_count,
            option_defined_many_to_one,
            option_defined_one_to_many,
        ) = self._categorize_connector_type_component(
            one_to_one_component, connector_component, connector_option, connector_type
        )

        # Check component existence
        if many_to_one_component:
            for components_name, input_count_defined, option_defined in zip(
                many_to_one_component, many_to_one_input_count, option_defined_many_to_one
            ):
                # counter reset
                input_count = 0
                option_defined_string = " from the option definition" if option_defined else ""
                for connection in connections_list:
                    if components_name in connection.get("source"):
                        input_count += 1

                if int(input_count_defined) != input_count:
                    raise FASTGAHEInputCountError(
                        f"Component {components_name} defines {input_count_defined} inputs"
                        + option_defined_string
                        + f", but "
                        f"{input_count} input(s) is/are listed in the connection section"
                    )

        # Check component existence
        if one_to_many_component:
            for components_name, output_count_defined, option_defined in zip(
                one_to_many_component, one_to_many_output_count, option_defined_one_to_many
            ):
                # counter reset
                output_count = 0
                option_defined_string = " from the option definition" if option_defined else ""

                for connection in connections_list:
                    if components_name in connection.get("target"):
                        output_count += 1

                if int(output_count_defined) != output_count:
                    raise FASTGAHEOutputCountError(
                        f"Component {components_name} defines {output_count_defined} outputs"
                        + option_defined_string
                        + f", but "
                        f"{output_count} output(s) is/are listed in the connection section"
                    )

        # Check component existence
        if many_to_many_component:
            for components_name, input_count_defined, output_count_defined in zip(
                many_to_many_component, many_to_many_input_count, many_to_many_output_count
            ):
                input_count = 0
                output_count = 0

                for connection in connections_list:
                    if components_name in connection.get("source"):
                        input_count += 1

                    elif components_name in connection.get("target"):
                        output_count += 1

                if int(input_count_defined) != input_count:
                    raise FASTGAHEInputCountError(
                        f"Component {components_name} defines {input_count_defined} inputs from "
                        f"the option definition, but {output_count} input(s) is/are listed in the "
                        f"connection section"
                    )

                if int(output_count_defined) != output_count:
                    raise FASTGAHEOutputCountError(
                        f"Component {components_name} defines {output_count_defined} outputs from "
                        f"the option definition, but {output_count} output(s) is/are listed in the "
                        f"connection section"
                    )

        # Check typo or undefined component
        for connection in connections_list:
            source_name = (
                connection["source"][0]
                if type(connection["source"]) is list
                else connection.get("source")
            )
            target_name = (
                connection["target"][0]
                if type(connection["target"]) is list
                else connection.get("target")
            )

            if source_name not in self._components_name or target_name not in self._components_name:
                if source_name not in self._components_name:
                    raise FASTGAHEComponentsNotIdentified(
                        f"{source_name} is not defined as a component!"
                    )
                else:
                    raise FASTGAHEComponentsNotIdentified(
                        f"{target_name} is not defined as a component!"
                    )

        # Check if there is any component missing in connection
        for components_name in self._components_name:
            if components_name not in propulsor_component + aux_load_component and not any(
                components_name in connection.get("target") for connection in connections_list
            ):
                raise FASTGAHEComponentConnectionError(f"{components_name} is missing as output!")

            if components_name not in energy_storage_component and not any(
                components_name in connection.get("source") for connection in connections_list
            ):
                raise FASTGAHEComponentConnectionError(f"{components_name} is missing as input!")

    def _categorize_connector_type_component(
        self, one_to_one_component, connector_names, connector_options, connector_type
    ):
        """
        This function categorizes the components in the connector component type class according
        to their number of input and output connections, the generator and turbo_generator are
        exceptions that are energy source component but categorized as connector for the
        powertrain component registry. This only applies in the _check_connection function.
        """

        one_to_many_component = []
        many_to_one_component = []
        many_to_many_component = []
        many_to_one_input_count = []
        one_to_many_output_count = []
        many_to_many_input_count = []
        many_to_many_output_count = []
        option_defined_one_to_many = []
        option_defined_many_to_one = []

        for name, options, type in zip(connector_names, connector_options, connector_type):
            defined_multi_connection_exists = options is not None and any(
                key.startswith("number_of_") for key in options.keys()
            )

            if defined_multi_connection_exists:
                # This is for the connectors having given input and output numbers from pt file
                # TODO: A concise way to define the input/output number option list

                input_options = ["number_of_inputs", "number_of_tanks"]
                output_options = [
                    "number_of_outputs",
                    "number_of_engines",
                    "number_of_power_sources",
                ]

                for option, num in zip(options.keys(), options.values()):
                    if option in input_options:
                        num_input = int(num)
                        if num_input < num or num_input <= 0:
                            raise FASTGAHEInvalidOptionDefinition(
                                f"{num} is invalid as input option value, only positive integers "
                                f"are allowed"
                            )

                    elif option in output_options:
                        num_output = int(num)
                        if num_output < num or num_output <= 0:
                            raise FASTGAHEInvalidOptionDefinition(
                                f"{num} is invalid as output option value, only positive integers "
                                f"are allowed"
                            )

                # First check if there is any side having multiple connection
                if num_input > 1 or num_output > 1:
                    # This is to identify many-to-many component
                    if num_input > 1 and num_output > 1:
                        many_to_many_component.append(name)
                        many_to_many_input_count.append(num_input)
                        many_to_many_output_count.append(num_output)

                    # This is to identify many-to-one component
                    elif num_input > 1:
                        many_to_one_component.append(name)
                        many_to_one_input_count.append(num_input)
                        option_defined_many_to_one.append(True)

                    # This is to identify one-to-many component
                    else:
                        one_to_many_component.append(name)
                        one_to_many_output_count.append(num_output)
                        option_defined_one_to_many.append(True)
                else:
                    one_to_one_component.append(name)

            else:
                if type == "DC_splitter" or type == "planetary_gear":
                    many_to_one_component.append(name)
                    many_to_one_input_count.append(2)
                    option_defined_many_to_one.append(False)

                elif type == "gearbox":
                    one_to_many_component.append(name)
                    one_to_many_output_count.append(2)
                    option_defined_one_to_many.append(False)

                else:
                    one_to_one_component.append(name)

        return (
            one_to_one_component,
            one_to_many_component,
            many_to_one_component,
            many_to_many_component,
            many_to_many_input_count,
            many_to_many_output_count,
            many_to_one_input_count,
            one_to_many_output_count,
            option_defined_many_to_one,
            option_defined_one_to_many,
        )

    def _construct_connection_graph(self):
        graph = nx.Graph()

        for component in self._components_name:
            graph.add_node(component)

        for connection in self._connection_list:
            # When the component is connected to a bus, the output number is also specified but it
            # isn't meaningful when drawing a graph, so we will just filter it

            if type(connection["source"]) is list:
                source = connection["source"][0]
            else:
                source = connection["source"]

            if type(connection["target"]) is list:
                target = connection["target"][0]
            else:
                target = connection["target"]

            graph.add_edge(source, target)

        self._connection_graph = graph

    def get_component_distance(self, references):
        """
        Calculate the shortest distance from each component to the nearest reference component(s) of
        a certain type or type class.

        :param references: Component type class or component type of reference component(s).
                          Can be a single string or a list of type string entries.
                          Each string should match a valid component type or component type class.

        :return: A dictionary mapping each component name to its minimum distance to any
                 of the reference components. Distance is measured as the number of edges in
                 the shortest path.
        """
        reference_component_types = []

        # Packing single component type / component type class string into list
        if isinstance(references, str):
            references = [references]

        # Collect the reference component type by comparing the registered component type classes
        # and component types to the given reference list
        for id in resources.KNOWN_ID:
            for reference in references:
                if isinstance(reference, str):
                    if (
                        reference == resources.DICTIONARY_CT[id]
                        or reference in resources.DICTIONARY_CTC[id]
                    ):
                        reference_component_types.append(resources.DICTIONARY_CT[id])

                    else:
                        raise AttributeError(
                            f"{reference} is not a valid entry for component type or "
                            f"component type class"
                        )

                else:
                    raise TypeError(
                        f"{reference} is not a valid data type for a component type or component "
                        f"type class"
                    )

        if not reference_component_types:
            raise ValueError("Invalid component type(s) or component type class(es)")

        # Collect reference component of the powertrain from the reference component type(s)
        reference_component_names = []
        propulsor_names = self.get_thrust_element_list()
        generator_names = self.get_generator_list()
        for component_type_class, component_type, component_name in zip(
            self._components_type_class, self._components_type, self._components_name
        ):
            if component_type in reference_component_types:
                reference_component_names.append(component_name)

            # This section checks whether a component with multiple component-type classes is
            # explicitly included in the reference list, or whether its two type classes are subsets
            # of that list. If only one of the two component types appears in the list, an
            # additional check is performed to confirm that the component indeed belongs to that
            # specific type class.
            if isinstance(component_type_class, list):
                # Check if the component type is in the reference list
                if component_type in references:
                    continue
                # Check if all component type class in the list is a subset of the reference list
                elif set(component_type_class).issubset(set(references)):
                    continue
                # Check if the component is a propulsive load
                elif "propulsive_load" in component_type_class and "propulsive_load" in references:
                    distance_from_component_dict = self._get_distance_from_component_name(
                        component_name
                    )
                    min_distance_from_propulsor = np.inf
                    min_distance_from_generator = np.inf

                    for component, distance in distance_from_component_dict.items():
                        if component in propulsor_names:
                            if distance < min_distance_from_propulsor:
                                min_distance_from_propulsor = distance

                        if component in generator_names:
                            if distance < min_distance_from_generator:
                                min_distance_from_generator = distance

                    if min_distance_from_generator < min_distance_from_propulsor:
                        reference_component_names.remove(component_name)

        return self._get_distance_from_component_name(reference_component_names)

    def _get_distance_from_component_name(self, component_names):
        """
        Calculate the shortest distance from each component to the nearest reference component(s).

        :param component_names: Component name of reference component(s).
                          Can be a single string or a list of type string entries.

        :return: A dictionary mapping each component name to its minimum distance to any
                 of the reference components. Distance is measured as the number of edges in
                 the shortest path.
        """
        if isinstance(component_names, str):
            component_names = [component_names]

        self._construct_connection_graph()
        graph = self._connection_graph

        distance_from_reference_component = {}
        connections_length_between_nodes = dict(nx.all_pairs_shortest_path_length(graph))

        for component_name in self._components_name:
            connected_components = list(connections_length_between_nodes[component_name].keys())
            connected_reference_component = list(set(component_names) & set(connected_components))

            min_distance = np.inf
            for component in connected_reference_component:
                distance_to_reference = connections_length_between_nodes[component_name][component]
                if distance_to_reference < min_distance:
                    min_distance = distance_to_reference

            distance_from_reference_component[component_name] = min_distance

        return distance_from_reference_component

    def reorder_components(self, *lists):
        """
        Reorders components by their distance from the nearest propeller and assigns proper
        sequential indices. Takes multiple property lists where the first list contains component
        names/keys, and reorders all lists according to the distance-based mapping. This improves
        robustness by ensuring that variables are updated in a correct order for each run.

        :param *lists: Variable number of property lists to be reordered. The first list should
        contain component names/keys that correspond to keys in the distance_from_propulsor
        dictionary. All subsequent lists will be reordered according to the same mapping.

        :return: tuple: All input lists reordered according to distance from propulsor, maintaining
        the same order and count as input lists.
        """
        # Sort items by value first, then by original key order to maintain consistency
        distance_from_prop = self.get_component_distance("propulsor")
        sorted_items = sorted(distance_from_prop.items(), key=lambda x: (x[1], x[0]))

        # Create new dictionary with proper sequential indices
        reindexed_dict = {}
        for index, (key, original_value) in enumerate(sorted_items):
            reindexed_dict[key] = index

        # Reorder other property lists using the same mapping
        reordered_lists = []
        for lst in lists:
            reordered = [None] * len(lst)
            for old_pos, key in enumerate(lists[0]):
                new_pos = reindexed_dict[key]
                reordered[new_pos] = lst[old_pos]
            reordered_lists.append(reordered)

        return tuple(reordered_lists)

    def check_sspc_states(self, declared_state):
        self._construct_connection_graph()
        graph = self._connection_graph

        components_to_check = {}

        name_to_id = dict(zip(self._components_name, self._components_id))

        # For now we will only check cable that have SSPC on both ends
        for component_id, component_name in zip(self._components_id, self._components_name):
            if component_id == "fastga_he.pt_component.dc_line":
                neighbors = graph.adj[component_name]
                # If component is a dc line, check that it has neighbors, then check if one at
                # least one of those is an sspc
                if neighbors:
                    sspc_neighbors = []
                    for neighbor in neighbors:
                        if name_to_id[neighbor] == "fastga_he.pt_component.dc_sspc":
                            sspc_neighbors.append(neighbor)
                    if len(sspc_neighbors) == 1:
                        raise FASTGAHESingleSSPCAtEndOfLine(
                            "Line " + component_name + " is connected to a single SSPC, this will "
                            "work as long as the SSPC is closed, but won't allow to open it"
                        )
                    # If there are no SSPC neighbor, no need to check the harness
                    elif len(sspc_neighbors) == 2:
                        components_to_check[component_name] = sspc_neighbors

        # For all case to check, see the default state that was given to them and change it if
        # need be
        actual_state = copy.deepcopy(declared_state)
        if components_to_check:
            for component_to_check in components_to_check:
                front_end, back_end = components_to_check[component_to_check]
                # If both are in a different state raise a warning and change both of them to
                # False (circuit open)
                if declared_state[front_end] ^ declared_state[back_end]:
                    _LOGGER.warning(
                        "SSPCs " + front_end + " and " + back_end + " should be in "
                        "the same state, they are thus forced at open"
                    )
                    actual_state[front_end] = False
                    actual_state[back_end] = False

        return actual_state

    def get_sizing_element_lists(self) -> tuple:
        """
        Returns the list of parameters necessary to create the sizing group based on what is
        inside the power train file.
        """

        self._get_components()

        return (
            self._components_name,
            self._components_name_id,
            self._components_type,
            self._components_om_type,
            self._components_options,
            self._components_position,
        )

    def get_performances_element_lists(self) -> tuple:
        """
        Returns the list of parameters necessary to create the performances group based on what is
        inside the power train file.
        """

        self._get_connections()

        # We now need to check if the SSPC "logic" is respected, the main rule being for now that
        # if a cable has one SSPC at each end of the cable, they should both be in the same
        # state. We will consider that the open state has the priority since it is what would
        # happen in reality.

        return (
            self._components_name,
            self._components_name_id,
            self._components_type,
            self._components_om_type,
            self._components_options,
            self._components_connection_outputs,
            self._components_connection_inputs,
            self._components_promotes,
            self._sspc_list,
            self._sspc_default_state,
        )

    def get_slipstream_element_lists(self) -> tuple:
        """
        Returns the list of parameters necessary to create the slipstream group based on what is
        inside the power train file.
        """

        self._get_components()

        return (
            self._components_name,
            self._components_name_id,
            self._components_type,
            self._components_om_type,
            self._components_slipstream_promotes,
            self._components_slipstream_flaps,
            self._components_slipstream_wing_lift,
        )

    def get_performances_to_slipstream_element_lists(self) -> tuple:
        """
        Returns the list of variable to promote from the performances component to the slipstream
        component.
        """

        self._get_connections()

        variables_to_check = []

        # Get a list of the variables to connect from performances to slipstream.
        for candidate_component, candidate_connections in zip(
            self._components_name, self._components_performances_to_slipstream
        ):
            for candidate_connection in candidate_connections:
                variables_to_check.append(candidate_component + "." + candidate_connection)

        inputs_in_slipstream = []
        outputs_in_performances = []

        for variable_to_check in variables_to_check:
            inputs_in_slipstream.append(variable_to_check)
            outputs_in_performances.append(
                self._components_connection_outputs[
                    self._components_connection_inputs.index(variable_to_check)
                ]
            )

        return inputs_in_slipstream, outputs_in_performances

    def get_control_parameter_list(self) -> List[str]:
        """
        Returns the list of control parameters of the components inside the powertrain.
        """

        self._get_components()

        # Because we might want different thrust distribution for mission and landing. As the
        # default is always an array of one for that variable it shouldn't cause any problem.
        ctrl_param_list = ["data:propulsion:he_power_train:thrust_distribution"]

        for comp_name, comp_type, comp_ctrl_params in zip(
            self._components_name, self._components_type, self._components_control_parameters
        ):
            for comp_ctrl_param in comp_ctrl_params:
                ctrl_param_name = (
                    PT_DATA_PREFIX + comp_type + ":" + comp_name + ":" + comp_ctrl_param
                )
                ctrl_param_list.append(ctrl_param_name)

        return ctrl_param_list

    @staticmethod
    def enforce_sspc_last(
        components_name: list,
        components_name_id: list,
        components_om_type: list,
        components_options: list,
        components_promotes: list,
    ) -> Tuple[list, list, list, list, list]:
        """
        It turns out that the SSPC can cause a bit of a mess when connected to cable, because,
        as one side is computed and the other not, this might create huge current which will more
        often than not prevent the code from converging. A solution found was to make it so that
        the SSPC are always computed last in the performances. So far it works.

        :param components_name: list that contains the name of the component as it will be found
        in the input/output file
        :param components_name_id: list that contains the name of the options used to provide the
        names in self._components_name
        :param components_om_type: list that contains the suffix of the component added to
        Performances and Sizing
        :param components_options: list that contains the options of the components
        :param components_promotes: list that contains the list of aircraft inputs that are
        necessary to promote in the performances modules for the code to work
        """

        sspc_list_components_name = []
        sspc_list_components_name_id = []
        sspc_list_components_om_type = []
        sspc_list_components_options = []
        sspc_list_components_promotes = []

        other_components_name = []
        other_components_name_id = []
        other_components_om_type = []
        other_components_options = []
        other_components_promotes = []

        for (
            component_name,
            component_name_id,
            component_om_type,
            component_options,
            component_promotes,
        ) in zip(
            components_name,
            components_name_id,
            components_om_type,
            components_options,
            components_promotes,
        ):
            if resources.DICTIONARY_CN_ID["fastga_he.pt_component.dc_sspc"] in component_name_id:
                sspc_list_components_name.append(component_name)
                sspc_list_components_name_id.append(component_name_id)
                sspc_list_components_om_type.append(component_om_type)
                sspc_list_components_options.append(component_options)
                sspc_list_components_promotes.append(component_promotes)

            else:
                other_components_name.append(component_name)
                other_components_name_id.append(component_name_id)
                other_components_om_type.append(component_om_type)
                other_components_options.append(component_options)
                other_components_promotes.append(component_promotes)

        components_name = other_components_name + sspc_list_components_name
        components_name_id = other_components_name_id + sspc_list_components_name_id
        components_om_type = other_components_om_type + sspc_list_components_om_type
        components_options = other_components_options + sspc_list_components_options
        components_promotes = other_components_promotes + sspc_list_components_promotes

        return (
            components_name,
            components_name_id,
            components_om_type,
            components_options,
            components_promotes,
        )

    def get_mass_element_lists(self) -> list:
        """
        Returns the list of OpenMDAO variables necessary to create the component which computes
        the mass of the power train.
        """

        self._get_components()

        variable_names = []

        for component_type, component_name in zip(self._components_type, self._components_name):
            variable_names.append(PT_DATA_PREFIX + component_type + ":" + component_name + ":mass")

        return variable_names

    def get_cg_element_lists(self) -> list:
        """
        Returns the list of OpenMDAO variables necessary to create the component which computes
        the center of gravity of the power train.
        """

        self._get_components()

        variable_names_cg = []

        for component_type, component_name in zip(self._components_type, self._components_name):
            variable_names_cg.append(
                PT_DATA_PREFIX + component_type + ":" + component_name + ":CG:x"
            )

        return variable_names_cg

    def get_drag_element_lists(self) -> Tuple[list, list]:
        """
        Returns the list of OpenMDAO variables necessary to create the component which computes
        the drag of the power train. Will return both the name of high speed and low speed drag
        as they are meant to be used once in the same component
        """

        self._get_components()

        variable_names_drag_ls = []
        variable_names_drag_cruise = []

        for component_type, component_name in zip(self._components_type, self._components_name):
            variable_names_drag_ls.append(
                PT_DATA_PREFIX + component_type + ":" + component_name + ":low_speed:CD0"
            )
            variable_names_drag_cruise.append(
                PT_DATA_PREFIX + component_type + ":" + component_name + ":cruise:CD0"
            )

        return variable_names_drag_ls, variable_names_drag_cruise

    def get_thrust_element_list(self) -> list:
        """
        Returns the list of OpenMDAO variables necessary to create the component which computes
        the repartition of thrust among propellers.
        """

        self._get_components()
        components_names = []

        for component_type_class, component_name in zip(
            self._components_type_class, self._components_name
        ):
            if "propulsor" in component_type_class:
                components_names.append(component_name)

        return components_names

    def get_propulsive_element_list(self) -> tuple:
        """
        Returns the list of OpenMDAO variables necessary to create the component which computes
        the ratio between the required power and the max available power on propulsive loads.
        """

        self._get_components()
        components_names = []
        components_types = []

        for component_type_class, component_name, component_type in zip(
            self._components_type_class, self._components_name, self._components_type
        ):
            if "propulsive_load" in component_type_class:
                components_names.append(component_name)
                components_types.append(component_type)

        return components_names, components_types

    def get_energy_consumption_list(self) -> list:
        """
        Returns the list of OpenMDAO variables necessary to create the component which sum the
        contribution of each source to the global energy consumption.
        """

        self._get_components()
        components_names = []

        for component_type_class, component_name in zip(
            self._components_type_class, self._components_name
        ):
            if "source" in component_type_class:
                components_names.append(component_name)

        return components_names

    def get_fuel_tank_list(self) -> Tuple[list, list]:
        """
        Returns the list of components inside the power train which may cause the CG to shift
        during flight because of a varying mass (but a constant position)
        """

        self._get_components()
        components_names = []
        components_types = []

        for component_type_class, component_name, component_type in zip(
            self._components_type_class, self._components_name, self._components_type
        ):
            if "tank" in component_type_class:
                components_names.append(component_name)
                components_types.append(component_type)

        return components_names, components_types

    def get_generator_list(self) -> list:
        """
        Returns the list of generator component(s) in the powertrain architecture.
        """

        self._get_components()
        components_names = []

        for component_type, component_name in zip(self._components_type, self._components_name):
            if "generator" in component_type:
                components_names.append(component_name)

        return components_names

    def get_fuel_tank_list_and_fuel(self) -> Tuple[list, list, list]:
        """
        Returns the list of components inside the power train which contain fuel and what type of
        fuel they contain. To do so we'll analyse the source they are connected to.
        """

        fuel_tanks_names, fuel_tanks_types = self.get_fuel_tank_list()
        source_names = self.get_energy_consumption_list()
        name_to_type = dict(zip(self._components_name, self._components_type))

        fuel_types = []

        connected_graphs = self.get_connection_graph()

        for fuel_tank_name in fuel_tanks_names:
            for connected_graph in connected_graphs:
                if fuel_tank_name not in list(connected_graph.nodes):
                    continue
                else:
                    # We check which sources is the closest neighbor
                    distance_closest_source = np.inf
                    distances_in_graph = dict(nx.all_pairs_shortest_path_length(connected_graph))
                    distances_to_tank = distances_in_graph[fuel_tank_name]

                    for source_name in source_names:
                        if source_name in distances_to_tank:
                            if distances_to_tank[source_name] < distance_closest_source:
                                closest_source = source_name
                                distance_closest_source = distances_to_tank[source_name]

            # I trust that there will always be at least one source connected to tank.
            # I shouldn't
            # But I do
            fuel_types.append(TYPE_TO_FUEL[name_to_type[closest_source]])

        return fuel_tanks_names, fuel_tanks_types, fuel_types

    def get_electricity_storage_list(self) -> Tuple[list, list]:
        """
        Returns the list of electricity storage components inside the power train.
        """

        self._get_components()
        components_names = []
        components_types = []

        for component_id, component_name, component_type in zip(
            self._components_id, self._components_name, self._components_type
        ):
            if component_id in ELECTRICITY_STORAGE_TYPES:
                components_names.append(component_name)
                components_types.append(component_type)

        return components_names, components_types

    def get_residuals_watcher_elements_list(self) -> tuple:
        """
        Returns the list of OpenMDAO variables that are interesting to monitor in the residuals
        watcher.
        """
        self._get_components()

        components_residuals_watchers_name_organised_list = []
        components_name_organised_list = []

        for component_name, component_id in zip(self._components_name, self._components_id):
            component_res_list = resources.DICTIONARY_RSD[component_id]
            for components_res_watcher in component_res_list:
                components_name_organised_list.append(component_name)
                components_residuals_watchers_name_organised_list.append(components_res_watcher)

        return components_name_organised_list, components_residuals_watchers_name_organised_list

    def get_performance_watcher_elements_list(self) -> tuple:
        """
        Returns the list of OpenMDAO variables that are to be registered by the performances
        watcher.
        """

        self._get_connections()

        components_perf_watchers_name_organised_list = []
        components_perf_watchers_unit_organised_list = []
        components_name_organised_list = []

        name_to_id = dict(zip(self._components_name, self._components_id))
        id_to_option = dict(zip(self._components_id, self._components_options))

        for component_name, components_perf_watchers in zip(
            self._components_name, self._components_perf_watchers
        ):
            component_perf_watchers_copy = copy.deepcopy(components_perf_watchers)

            # Need a more generic way to do this, here we will do it once because the battery is
            # a unique case
            component_id = name_to_id[component_name]
            if component_id == "fastga_he.pt_component.battery_pack":
                component_option = id_to_option[component_id]
                # If there is a direct connection, the option won't be empty
                if component_option and {"voltage_out": "V"} in component_perf_watchers_copy:
                    # We remove what has become an input and add what has become an output
                    component_perf_watchers_copy.remove({"voltage_out": "V"})
                    component_perf_watchers_copy.append({"dc_current_out": "A"})

            for components_perf_watcher in component_perf_watchers_copy:
                key, value = list(components_perf_watcher.items())[0]
                components_name_organised_list.append(component_name)
                components_perf_watchers_name_organised_list.append(key)
                components_perf_watchers_unit_organised_list.append(value)

        return (
            components_name_organised_list,
            components_perf_watchers_name_organised_list,
            components_perf_watchers_unit_organised_list,
        )

    def get_slipstream_performance_watcher_elements_list(self) -> tuple:
        """
        Returns the list of OpenMDAO variables used in the computation of the slipstream effects
        that are to be registered by the performances watcher.
        """

        self._get_components()

        components_slip_perf_watchers_name_organised_list = []
        components_slip_perf_watchers_unit_organised_list = []
        components_slip_name_organised_list = []

        for component_name, components_slip_perf_watchers in zip(
            self._components_name, self._components_slipstream_perf_watchers
        ):
            for components_perf_watcher in components_slip_perf_watchers:
                key, value = list(components_perf_watcher.items())[0]
                components_slip_name_organised_list.append(component_name)
                components_slip_perf_watchers_name_organised_list.append(key)
                components_slip_perf_watchers_unit_organised_list.append(value)

        return (
            components_slip_name_organised_list,
            components_slip_perf_watchers_name_organised_list,
            components_slip_perf_watchers_unit_organised_list,
        )

    def get_wing_punctual_mass_element_list(self) -> Tuple[list, list, list]:
        """
        This function returns a list of the components that are to be considered as punctual
        masses acting on the wing due to their positions as defined in the powertrain file
        """

        self._get_components()

        punctual_mass_names = []
        punctual_mass_types = []
        component_pairs = copy.deepcopy(self._components_symmetrical_pairs)

        for component_id, component_name, component_position, component_type in zip(
            self._components_id,
            self._components_name,
            self._components_position,
            self._components_type,
        ):
            if component_position in resources.DICTIONARY_PCT_W[component_id]:
                punctual_mass_names.append(component_name)
                punctual_mass_types.append(component_type)

        # TODO: improve the way this is done, as I'm not satisfied with it
        for component_pair in self._components_symmetrical_pairs:
            if component_pair[0] in punctual_mass_names:
                continue

            elif component_pair[1] in punctual_mass_names:
                continue

            else:
                component_pairs.remove(component_pair)

        return punctual_mass_names, punctual_mass_types, component_pairs

    def get_wing_punctual_fuel_element_list(self) -> Tuple[list, list, list]:
        """
        This function returns a list of the components that are to be considered as punctual
        fuel tanks acting on the wing due to their positions as defined in the powertrain file
        """

        self._get_components()

        punctual_tank_names = []
        punctual_tank_types = []
        component_pairs = copy.deepcopy(self._components_symmetrical_pairs)

        for component_id, component_name, component_position, component_type in zip(
            self._components_id,
            self._components_name,
            self._components_position,
            self._components_type,
        ):
            if component_position in resources.DICTIONARY_PCT_W_F[component_id]:
                punctual_tank_names.append(component_name)
                punctual_tank_types.append(component_type)

        # TODO: improve the way this is done, as I'm not satisfied with it
        for component_pair in self._components_symmetrical_pairs:
            if component_pair[0] in punctual_tank_names:
                continue

            elif component_pair[1] in punctual_tank_names:
                continue

            else:
                component_pairs.remove(component_pair)

        return punctual_tank_names, punctual_tank_types, component_pairs

    def get_wing_distributed_mass_element_list(self) -> Tuple[list, list, list]:
        """
        This function returns a list of the components that are to be considered as distributed
        masses acting on the wing due to their positions as defined in the powertrain file
        """

        self._get_components()

        distributed_mass_names = []
        distributed_mass_types = []
        component_pairs = copy.deepcopy(self._components_symmetrical_pairs)

        for component_id, component_name, component_position, component_type in zip(
            self._components_id,
            self._components_name,
            self._components_position,
            self._components_type,
        ):
            if component_position in resources.DICTIONARY_DST_W[component_id]:
                distributed_mass_names.append(component_name)
                distributed_mass_types.append(component_type)

        # TODO: improve the way this is done, as I'm not satisfied with it
        for component_pair in self._components_symmetrical_pairs:
            if component_pair[0] in distributed_mass_names:
                continue

            elif component_pair[1] in distributed_mass_names:
                continue

            else:
                component_pairs.remove(component_pair)

        return distributed_mass_names, distributed_mass_types, component_pairs

    def get_wing_distributed_fuel_element_list(self) -> Tuple[list, list, list]:
        """
        This function returns a list of the components that are to be considered as distributed
        fuel tanks acting on the wing due to their positions as defined in the powertrain file
        """

        self._get_components()

        distributed_tanks_names = []
        distributed_tanks_types = []
        component_pairs = copy.deepcopy(self._components_symmetrical_pairs)

        for component_id, component_name, component_position, component_type in zip(
            self._components_id,
            self._components_name,
            self._components_position,
            self._components_type,
        ):
            if component_position in resources.DICTIONARY_DST_W_F[component_id]:
                distributed_tanks_names.append(component_name)
                distributed_tanks_types.append(component_type)

        # TODO: improve the way this is done, as I'm not satisfied with it
        for component_pair in self._components_symmetrical_pairs:
            if component_pair[0] in distributed_tanks_names:
                continue

            elif component_pair[1] in distributed_tanks_names:
                continue

            else:
                component_pairs.remove(component_pair)

        return distributed_tanks_names, distributed_tanks_types, component_pairs

    def will_aircraft_mass_vary(self):
        """
        This function returns a boolean telling whether or not there are components in the
        powertrain that will make the aircraft mass vary during the flight (like burning fuel or
        certain types of batteries). For now, will only be used in the initial guess.
        """

        self._get_components()

        return any(self._components_makes_mass_vary)

    def has_fuel_non_consumable_energy_source(self):
        """
        This function returns a boolean telling whether or not there are energy sources in the
        powertrain that will not make the aircraft vary (like batteries). For now is only used to
        provide smart initial guess.
        """

        self._get_components()

        return any(self._source_does_not_make_mass_vary)

    def get_connection_graph(self) -> list:
        """
        This function returns a graph of connection inside the powertrain without doubling the
        components like what get_graphs_connected_voltage() does
        """

        self._get_connections()

        graph = nx.Graph()

        for component_name in self._components_name:
            graph.add_node(component_name)

        for connection in self._connection_list:
            # For bus and splitter, we don't really care about what number of input it is
            # connected to, so we do the following

            if type(connection["source"]) is list:
                source = connection["source"][0]
            else:
                source = connection["source"]

            if type(connection["target"]) is list:
                target = connection["target"][0]
            else:
                target = connection["target"]

            graph.add_edge(source, target)

        sub_graphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

        return sub_graphs

    def get_graphs_connected_voltage(self) -> list:
        """
        This function returns a list of graphs of connected PT components that have more or less
        the same imposed voltage. What is meant by that is that since some component impose the
        voltage on the circuit while other have independent I/O in terms of voltage e.g the DC/DC
        converter this will make it so that there might some subgraph of the architecture with
        different connected voltage.
        """

        self._get_connections()

        graph = nx.Graph()

        for component_name, component_id in zip(self._components_name, self._components_id):
            graph.add_node(
                component_name + "_out",
            )
            graph.add_node(
                component_name + "_in",
            )

            if not resources.DICTIONARY_IO_INDEP_V[component_id]:
                graph.add_edge(
                    component_name + "_out",
                    component_name + "_in",
                )

        for connection in self._connection_list:
            # For bus and splitter, we don't really care about what number of input it is
            # connected to, so we do the following

            if type(connection["source"]) is list:
                source = connection["source"][0]
            else:
                source = connection["source"]

            if type(connection["target"]) is list:
                target = connection["target"][0]
            else:
                target = connection["target"]

            graph.add_edge(
                source + "_in",
                target + "_out",
            )

        sub_graphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

        return sub_graphs

    def _list_voltage_coherence_to_check(self) -> Tuple[list, list]:
        """
        Makes a list, for all sub graphs, of the components that sets the voltage inside of them,
        the check on the coherency of the value will be ade later.
        """

        # This line prompts the identification of the power train from the file
        sub_graphs = self.get_graphs_connected_voltage()

        # We create a dictionary to associate name to id
        name_to_id_dict = dict(zip(self._components_name, self._components_id))

        sub_graphs_voltage_setter = []

        for sub_graph in sub_graphs:
            # First we make a list of all components in the sub graph that sets the voltage.
            nodes_list = list(sub_graph.nodes)

            # Then we turns those nodes (components name) in the corresponding component id and
            # check if this id sets the voltage and count how many of them do.
            node_that_sets_voltage = []

            for node in nodes_list:
                clean_node_name = node.replace("_in", "").replace("_out", "")

                node_id = name_to_id_dict[clean_node_name]

                # Since only the output of the components set the voltage, we will oly include
                # them in the list
                if resources.DICTIONARY_SETS_V[node_id] and "_out" in node:
                    node_that_sets_voltage.append(node)

            sub_graphs_voltage_setter.append(node_that_sets_voltage)

        return sub_graphs, sub_graphs_voltage_setter

    def check_voltage_coherence(self, inputs, number_of_points: int):
        """
        Check that all the sub graphs of independent voltage are compatible, meaning that if
        there is more than one component that sets the voltage, they have the same target voltage.

        :param inputs: inputs vector, in the OpenMDAO format, which contains the value of the
        voltages to check
        :param number_of_points: number of points in the data to check
        """

        sub_graphs_voltage_setters = self._list_voltage_coherence_to_check()[1]

        name_to_type = dict(zip(self._components_name, self._components_type))

        for sub_graph_voltage_setters in sub_graphs_voltage_setters:
            ref_voltage = None
            # If zero or one voltage setters nothing to check
            if len(sub_graph_voltage_setters) < 2:
                pass
            else:
                # Now for all those setter, we put them in the same format (if it was given as a
                # float we transform it in array)
                will_work = True
                variables_to_check = []
                for voltage_setter in sub_graph_voltage_setters:
                    clean_setter_name = voltage_setter.replace("_in", "").replace("_out", "")
                    setter_type = name_to_type[clean_setter_name]
                    input_name = (
                        PT_DATA_PREFIX
                        + setter_type
                        + ":"
                        + clean_setter_name
                        + ":voltage_out_target_mission"
                    )
                    variables_to_check.append(input_name)
                    data_value = inputs[input_name]
                    # We initiate the test with the first value we find
                    if ref_voltage is None:
                        # If not a float, it is an array !
                        if len(data_value) == 1:
                            ref_voltage = np.full(number_of_points, data_value)
                        else:
                            ref_voltage = data_value
                    # We check if coherent with other value
                    else:
                        if len(data_value) == 1:
                            data_value = np.full(number_of_points, data_value)

                        if not np.array_equal(ref_voltage, data_value):
                            will_work = False

                if not will_work:
                    raise FASTGAHEIncoherentVoltage(
                        "The target voltage chosen for the following input: "
                        + ", ".join(variables_to_check)
                        + " is incoherent. Ensure that they have the same value and/or units"
                    )

    def get_voltage_to_set(self, inputs, number_of_points: int) -> List[dict]:
        """
        Returns a list of the dict of voltage variable names and the value they should be set at
        for each of the subgraph. Dict will be empty if there is no voltage to set. The voltage
        to set are defined in the registered_components.py file. Note that this function was
        coded based on the assumption that the voltage coherence checker was ran before hand. It
        also assumes that all the voltage setter are RMS value when the voltage to set is an AC
        voltage.

        :param inputs: inputs vector, in the OpenMDAO format, which contains the value of the
        voltages to check
        :param number_of_points: number of points in the data to check
        """

        # This line prompts the identification of the power train from the file
        sub_graphs, sub_graphs_voltage_setters = self._list_voltage_coherence_to_check()

        name_to_type = dict(zip(self._components_name, self._components_type))
        name_to_id = dict(zip(self._components_name, self._components_id))
        name_to_ct = dict(zip(self._components_name, self._components_type))
        name_to_option = dict(zip(self._components_name, self._components_options))

        final_list = []
        voltage_at_each_node = {}

        for sub_graph, sub_graph_voltage_setters in zip(sub_graphs, sub_graphs_voltage_setters):
            # First and foremost, we get the value that will serve as the for the setting of the
            # voltage in this subgraph. If there are not setters in this subgraph we just pass along

            voltage_dict_subgraph = {}

            if sub_graph_voltage_setters:
                voltage_setter = sub_graph_voltage_setters[0]
                clean_setter_name = voltage_setter.replace("_in", "").replace("_out", "")
                setter_type = name_to_type[clean_setter_name]
                input_name = (
                    PT_DATA_PREFIX
                    + setter_type
                    + ":"
                    + clean_setter_name
                    + ":voltage_out_target_mission"
                )
                reference_voltage = inputs[input_name]
            else:
                # We need to use a fake value here, but, a priori, since there are no voltage
                # setter there won't be any voltage to set so we can put anything there. Expect,
                # again, for the battery which is a particular case. It doesn't appear as a
                # setter in the condition above but actually is when it is directly connected to
                # a bus.

                reference_voltage = None

                # The way we'll do it is a bit horrible but it is quick and works
                for node in sub_graph.nodes:
                    component_name = node.replace("_in", "").replace("_out", "")
                    component_id = name_to_id[component_name]

                    if (
                        component_id == "fastga_he.pt_component.battery_pack"
                        and "_out" in node
                        and name_to_option[component_name]
                    ):
                        number_of_cell_in_series = self.get_number_of_cell_in_series(
                            component_name=component_name,
                            component_type=name_to_ct[component_name],
                            inputs=inputs,
                        )
                        reference_voltage = (
                            np.linspace(4.2, 2.65, number_of_points) * number_of_cell_in_series
                        )

                if reference_voltage is None:
                    reference_voltage = np.array([DEFAULT_VOLTAGE_VALUE])

            # We now transform it in the proper array, if it already has the right shape,
            # this line does nothing
            if len(reference_voltage):
                reference_voltage = np.full(number_of_points, reference_voltage)

            # Now that we have the voltage to set, we can go through the node in the
            # architecture, check if they have a voltage to set, how far from the setter they are
            # and then compute what the voltage to set is.

            nodes_list = list(sub_graph.nodes)
            for node in nodes_list:
                component_name = node.replace("_in", "").replace("_out", "")
                component_id = name_to_id[component_name]

                # Now that we have the id of the node, we can check if there are some voltage to set
                voltages_to_set = resources.DICTIONARY_V_TO_SET[component_id]

                for voltage_to_set in voltages_to_set:
                    variable_name = component_name + "." + voltage_to_set
                    voltage_dict_subgraph[variable_name] = reference_voltage

                # If the node in question is the output of a battery in "normal" mode,
                # we can guesstimate the voltage but since it is so peculiar (not constant during
                # mission) we won't make it appear in the registered_components.py. Yet another
                # point of the code where the battery is a exception ^^'

                if (
                    component_id == "fastga_he.pt_component.battery_pack"
                    and "_out" in node
                    and not name_to_option[component_name]
                ):
                    number_of_cell_in_series = self.get_number_of_cell_in_series(
                        component_name=component_name,
                        component_type=name_to_ct[component_name],
                        inputs=inputs,
                    )
                    voltage_to_set = (
                        np.linspace(4.2, 2.65, number_of_points) * number_of_cell_in_series
                    )

                    voltage_dict_subgraph[component_name + ".voltage_out"] = voltage_to_set

                voltage_at_each_node[node] = reference_voltage

            final_list.append(voltage_dict_subgraph)

        self._voltage_at_each_node = voltage_at_each_node

        return final_list

    @staticmethod
    def get_number_of_cell_in_series(component_name: str, component_type: str, inputs) -> float:
        """
        This function returns the number of cell in series inside a battery module. Was put there
        because there is quite a process to extract the value and we will need it twice at least.

        :param component_name: name of the battery pack
        :param component_type: type of battery pack, for now there is only one but who knows
        :param inputs: inputs vector, in the OpenMDAO format
        """

        # We only know the promoted name of the variable so we can't access it
        # directly, but in the error message that will pop up when we try to use it,
        # we have all the info we need

        try:
            number_of_cell_in_series = inputs[
                PT_DATA_PREFIX + component_type + ":" + component_name + ":module:number_cells"
            ].item()

        except RuntimeError as e:
            error_message = e.args[0]
            abs_names = re.findall(r"\[.*?\]", error_message)[0][1:-1].replace(" ", "").split(",")
            abs_name = abs_names[0]
            split_abs_name = abs_name.split(".")
            # Sometimes the number of cells is not one deep but two deep so we try both
            try:
                proper_abs_name = ".".join(split_abs_name[1:])
                number_of_cell_in_series = inputs[proper_abs_name].item()
            except KeyError:
                proper_abs_name = ".".join(split_abs_name[2:])
                number_of_cell_in_series = inputs[proper_abs_name].item()

        return number_of_cell_in_series

    def get_directed_graph_sub_propulsion_chain(self):
        """
        This function returns a list of directed graphs of connected PT sub propulsion chain. As a
        prevision for next step all component will be split between their inputs and outputs to
        allow to include efficiency.
        """

        self._get_connections()

        graph = nx.DiGraph()

        for component_name, component_id in zip(self._components_name, self._components_id):
            graph.add_node(
                component_name + "_out",
            )
            graph.add_node(
                component_name + "_in",
            )

            graph.add_edge(
                component_name + "_out",
                component_name + "_in",
            )

        for connection in self._connection_list:
            # For bus and splitter, we don't really care about what number of input it is
            # connected to so we do the following

            if type(connection["source"]) is list:
                source = connection["source"][0]
            else:
                source = connection["source"]

            if type(connection["target"]) is list:
                target = connection["target"][0]
            else:
                target = connection["target"]

            graph.add_edge(
                source + "_in",
                target + "_out",
            )

        return graph

    def are_propulsor_connected_to_source(self):
        """
        This function returns a dictionary which contains, for each propulsor, a boolean which
        tells whether the propulsor can be actuated (meaning its connected to a source).
        """

        propulsor_list = self.get_thrust_element_list()
        source_list = self.get_energy_consumption_list()

        is_propulsor_connected_dict = {}

        source_output_name_list = []
        for source in source_list:
            source_output_name_list.append(source + "_out")

        powertrain_graph = self.get_directed_graph_sub_propulsion_chain()
        undirected_graph = powertrain_graph.to_undirected()

        # We will iterate through the default state of the sspc and remover the connections
        # between sspc in and sspc out if they are open (meaning no current goes through it so it
        # doesn't carry power).
        for sspc_name, sspc_default_state in self._sspc_default_state.items():
            # If not closed by default
            if not sspc_default_state:
                undirected_graph.remove_edge(sspc_name + "_in", sspc_name + "_out")

        for propulsor in propulsor_list:
            # For each propulsor we check that its input is connected to a source output. The
            # should only be on input for each propulsor and one output for each source which allows
            # to do like this
            propulsor_input_name = propulsor + "_in"

            connected_nodes = nx.node_connected_component(undirected_graph, propulsor_input_name)
            is_propulsor_connected = bool(
                connected_nodes.intersection(set(source_output_name_list))
            )

            is_propulsor_connected_dict[propulsor] = is_propulsor_connected

        return is_propulsor_connected_dict

    @staticmethod
    def fuel_system_power_inputs(inputs, components_name: str, power_output: np.ndarray) -> dict:
        """
        Computes the power at each input of the fuel system, depending on the mode the power at the
        output and the splitting decided

        :param inputs: OpenMDAO vector containing the value of inputs
        :param components_name: the name of the fuel system in question
        :param power_output: the power at the output of the fuel system
        """

        output_dict = {}

        # First, we will get the amount of connected fuel tanks, which should correspond to the
        # length of the distributor
        input_name = (
            "data:propulsion:he_power_train:fuel_system:" + components_name + ":fuel_distribution"
        )
        distributor = inputs[input_name] / sum(inputs[input_name])

        for idx, value in enumerate(distributor):
            # Because we start at one :)
            output_dict["fuel_consumed_in_t_" + str(idx + 1)] = value * power_output

        return output_dict

    def splitter_power_inputs(
        self, inputs, components_name: str, power_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the power at each input of the splitter, depending on the mode the power at the
        output

        :param inputs: OpenMDAO vector containing the value of inputs
        :param components_name: the name of the splitter in question
        :param power_output: the power at the output of the splitter
        """

        number_of_points = len(power_output)

        # First we need to search what mode the splitter is in
        name_to_option = dict(zip(self._components_name, self._components_options))

        # Check that an option is declared, else it means it is in default mode which is
        # percent_split
        mode = "percent_split"
        if name_to_option[components_name]:
            if "splitter_mode" in list(name_to_option[components_name].keys()):
                mode = name_to_option[components_name]["splitter_mode"]

        if mode == "percent_split":
            input_name = (
                "data:propulsion:he_power_train:DC_splitter:" + components_name + ":power_split"
            )
            power_split = inputs[input_name]
            power_split = format_to_array(power_split, number_of_points)

            # Should be in %
            primary_input = power_split * power_output / 100.0
            secondary_output = (100.0 - power_split) * power_output / 100.0

        else:
            input_name = (
                "data:propulsion:he_power_train:DC_splitter:" + components_name + ":power_share"
            )
            power_share = inputs[input_name]
            power_share = format_to_array(power_share, number_of_points)

            # Should be in W
            primary_input = np.minimum(power_share, power_output)
            secondary_output = power_output - primary_input

        return primary_input, secondary_output

    def gearbox_power_inputs(
        self, inputs, components_name: str, power_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the power at each input of the gearbox, depending on the mode the power at the
        output

        :param inputs: OpenMDAO vector containing the value of inputs
        :param components_name: the name of the gearbox in question
        :param power_output: the power at the output of the gearbox
        """

        number_of_points = len(power_output)

        # First we need to search what mode the gearbox is in
        name_to_option = dict(zip(self._components_name, self._components_options))

        # Check that an option is declared, else it means it is in default mode which is
        # percent_split
        mode = "percent_split"
        if name_to_option[components_name]:
            if "gear_mode" in list(name_to_option[components_name].keys()):
                mode = name_to_option[components_name]["gear_mode"]

        if mode == "percent_split":
            input_name = (
                "data:propulsion:he_power_train:planetary_gear:" + components_name + ":power_split"
            )
            power_split = inputs[input_name]
            power_split = format_to_array(power_split, number_of_points)

            # Should be in %
            primary_input = power_split * power_output / 100.0
            secondary_output = (100.0 - power_split) * power_output / 100.0

        else:
            input_name = (
                "data:propulsion:he_power_train:planetary_gear:" + components_name + ":power_share"
            )
            power_share = inputs[input_name]
            power_share = format_to_array(power_share, number_of_points)

            # Should be in W
            primary_input = np.minimum(power_share, power_output)
            secondary_output = power_output - primary_input

        return primary_input, secondary_output

    def get_power_to_set(self, inputs, propulsive_power_dict: dict) -> Tuple[dict, dict]:
        """
        Returns a list of the power at each nodes of each subgraph. Also returns a list of the
        dict of current variable names and the value they should be set at for each of the
        subgraph. Dict will be empty if there is no power to set. The power to set are defined in
        the registered_components.py file.

        :param inputs: inputs vector, in the OpenMDAO format, which contains the value of the
        voltages to check
        :param propulsive_power_dict: dictionary with the propulsive power of each propulsor
        """

        # We rewrite the propulsive power dict to match the name of the nodes
        propulsive_loads_name = list(propulsive_power_dict.keys())
        propulsive_loads_proper_name = []
        proper_propulsive_power_dict = {}

        for propulsive_load_name in propulsive_loads_name:
            propulsive_load_proper_name = propulsive_load_name + "_out"
            propulsive_loads_proper_name.append(propulsive_load_name)
            proper_propulsive_power_dict[propulsive_load_proper_name] = propulsive_power_dict[
                propulsive_load_name
            ]

        graph = self.get_directed_graph_sub_propulsion_chain()

        # Need to be put here else the _get_component hasn't triggered yet
        name_to_id = dict(zip(self._components_name, self._components_id))
        name_to_eta = dict(zip(self._components_name, self._components_efficiency))

        # Get a list of nodes who hasn't been treated, will serve as way to check that good
        # progress is made.
        untreated_nodes = list(graph.nodes)

        power_at_each_node = {}
        # Initialize the dict with the power at each node with an array full of zeros except for
        # propulsors.
        template_power = list(proper_propulsive_power_dict.values())[0]
        treated_nodes = []
        for untreated_node in untreated_nodes:
            if untreated_node in list(proper_propulsive_power_dict.keys()):
                power_at_each_node[untreated_node] = proper_propulsive_power_dict[untreated_node]
                treated_nodes.append(untreated_node)
            else:
                power_at_each_node[untreated_node] = np.zeros_like(template_power)

        node_to_remove_at_the_end = []

        # Remove treated nodes
        untreated_nodes = [x for x in untreated_nodes if x not in treated_nodes]

        # Now we can start treating the nodes. We'll just keep going over the graph until all
        # nodes have been treated and we'll check that progress is being made by monitoring the
        # number of nodes treated each loop.
        previous_treated_node_number = 1

        while len(untreated_nodes) != 0 and previous_treated_node_number != 0:
            previous_treated_node_number = 0

            for node in untreated_nodes:
                # First check that we can treat the node. If we can't we move to the next node.
                # A node can be treated if all its predecessor were treated
                can_be_treated = True
                for predecessor in graph.pred[node]:
                    if predecessor not in treated_nodes:
                        can_be_treated = False

                if not can_be_treated:
                    continue

                component_name = copy.deepcopy(node)
                component_end = component_name[-1]
                if component_end.isdigit():
                    str_to_replace = "_" + component_end
                    component_name = component_name.replace(str_to_replace, "")
                component_name = component_name.replace("_out", "").replace("_in", "")

                # Here we are sure the node can be treated. But we'll add it to the list of
                # treated nodes only after we compute its value. Several case can however happen.

                # 1) If it only has one predecessor and that predecessor only has one successor it
                # means its either the connection between the input and output of a component so
                # we include the efficiency or its the output of a component connected to the
                # input of another one.
                if len(list(graph.pred[node])) == 1:
                    predecessor = list(graph.pred[node])[0]
                    if predecessor.endswith("_1"):
                        predecessor = predecessor.replace("_1", "")
                    predecessor_name = predecessor.replace("_out", "").replace("_in", "")

                    if len(list(graph.succ[predecessor])) == 1:
                        # To get the component name, we remove the "_in" the "_out" and the "_1".
                        # We should only have to remove the "_1" as we are ensure that there is
                        # only one predecessor it must be a "_1"

                        # if name is the same, we compute the power below base on the efficiency,
                        # else its the same power, to refactor, we'll just say the efficiency is 1
                        if component_name == predecessor_name:
                            eta = name_to_eta[component_name]
                        else:
                            eta = 1.0

                        power_at_each_node[node] = power_at_each_node[predecessor] / eta

                        treated_nodes.append(node)
                        previous_treated_node_number += 1
                        continue

                # 2) If it only has one predecessor and that predecessor only more than successor it
                # means its a component that splits_power (either planetary gear, splitter, ...).
                if len(list(graph.pred[node])) == 1:
                    predecessor = list(graph.pred[node])[0]
                    if predecessor.endswith("_1"):
                        predecessor = predecessor.replace("_1", "")
                    predecessor_name = predecessor.replace("_out", "").replace("_in", "")

                    if len(list(graph.succ[predecessor])) > 1:
                        # Check the predecessor type
                        predecessor_type = name_to_id[predecessor_name]

                        if predecessor_type == "fastga_he.pt_component.dc_splitter":
                            (
                                primary_input_power,
                                secondary_power_output,
                            ) = self.splitter_power_inputs(
                                inputs=inputs,
                                components_name=predecessor_name,
                                power_output=power_at_each_node[predecessor],
                            )

                            # Then identify which current output it correspond to. Here if the
                            # component in question is not an sspc, the following should work.
                            # Else we will try both way

                            output_name = component_name + ".dc_current_out"
                            if name_to_id[component_name] == "fastga_he.pt_component.dc_sspc":
                                if output_name not in self._components_connection_outputs:
                                    output_name = component_name + ".dc_current_in"

                            if name_to_id[component_name] == "fastga_he.pt_component.dc_line":
                                output_name = component_name + ".dc_current"

                            # We look at the number of the corresponding splitter input
                            index = self._components_connection_outputs.index(output_name)
                            splitter_input_name = self._components_connection_inputs[index]

                            if splitter_input_name.endswith("1"):
                                power_at_each_node[node] = primary_input_power
                                power_at_each_node[predecessor + "_1"] = primary_input_power
                            else:
                                power_at_each_node[node] = secondary_power_output
                                power_at_each_node[predecessor + "_2"] = secondary_power_output

                            if predecessor not in node_to_remove_at_the_end:
                                node_to_remove_at_the_end.append(predecessor)

                        elif predecessor_type == "fastga_he.pt_component.planetary_gear":
                            (
                                primary_input_power,
                                secondary_power_output,
                            ) = self.gearbox_power_inputs(
                                inputs=inputs,
                                components_name=predecessor_name,
                                power_output=power_at_each_node[predecessor],
                            )

                            # We look at the number of the corresponding gearbox input. Will be a
                            # bit farfetched as we need to find using the current neighbor. For
                            # gearboxes, the two connexion are rpm and shaft power and they are
                            # both output of the input side of the gearbox, so we'll have to use
                            # the input list
                            # We look at the number of the corresponding splitter input
                            input_name = component_name + ".shaft_power_out"
                            index = self._components_connection_inputs.index(input_name)
                            gearbox_input_name = self._components_connection_outputs[index]

                            if gearbox_input_name.endswith("1"):
                                power_at_each_node[node] = primary_input_power
                                power_at_each_node[predecessor + "_1"] = primary_input_power
                            else:
                                power_at_each_node[node] = secondary_power_output
                                power_at_each_node[predecessor + "_2"] = primary_input_power

                            if predecessor not in node_to_remove_at_the_end:
                                node_to_remove_at_the_end.append(predecessor)

                        elif predecessor_type == "fastga_he.pt_component.fuel_system":
                            input_power_dict = self.fuel_system_power_inputs(
                                inputs=inputs,
                                components_name=predecessor_name,
                                power_output=power_at_each_node[predecessor],
                            )

                            output_name = component_name + ".fuel_consumed_t"
                            index = self._components_connection_inputs.index(output_name)
                            fuel_system_input_name = self._components_connection_outputs[index]
                            input_number = fuel_system_input_name[-1]
                            power_at_each_node[node] = input_power_dict[
                                "fuel_consumed_in_t_" + input_number
                            ]
                            power_at_each_node[predecessor + "_" + input_number] = input_power_dict[
                                "fuel_consumed_in_t_" + input_number
                            ]

                            if predecessor not in node_to_remove_at_the_end:
                                node_to_remove_at_the_end.append(predecessor)

                        treated_nodes.append(node)
                        previous_treated_node_number += 1
                        continue

                # 2) If it has more than one predecessor it is the output of a bus/fuel
                # system/gear. In this case we simply sum all the predecessor. We should be able
                # to do so since we ensured to get there that all predecessor were already treated
                if len(list(graph.pred[node])) > 1:
                    power = np.zeros_like(template_power)
                    for current_predecessor in graph.pred[node]:
                        power += power_at_each_node[current_predecessor]

                    power_at_each_node[node] = power
                    treated_nodes.append(node)
                    previous_treated_node_number += 1
                    continue

            # Now we remove all nodes we treated in this while loop
            untreated_nodes = [x for x in untreated_nodes if x not in treated_nodes]

        for node_to_remove in node_to_remove_at_the_end:
            power_at_each_node.pop(node_to_remove)

        self._power_at_each_node = power_at_each_node

        final_list = {}

        for node in list(graph.nodes):
            component_name = node.replace("_in", "").replace("_out", "")
            component_id = name_to_id[component_name]

            power_to_set = resources.DICTIONARY_P_TO_SET[component_id]

            for power in power_to_set:
                # These are tuple which contains the "in" or "out" tag plus the name of the
                # variable
                if power[1] == "in" and node.endswith("_in"):
                    variable_name = component_name + "." + power[0]

                    # If we are to set the power of a component with multiple inputs,
                    # the node name will not match and will need to be modified. We will
                    # check that we are in this case if the variable name endswith a number
                    if variable_name[-1].isdigit():
                        final_list[variable_name] = power_at_each_node[
                            node + "_" + variable_name[-1]
                        ]
                    else:
                        final_list[variable_name] = power_at_each_node[node]

                elif power[1] == "out" and node.endswith("_out"):
                    variable_name = component_name + "." + power[0]
                    final_list[variable_name] = power_at_each_node[node]

        return power_at_each_node, final_list

    def get_network_elements_list(self) -> tuple:
        """
        Returns the name of the components and their connections for the visualisation of the
        power train as a network graph.
        """

        self._get_connections()

        icons_name = []
        icons_size = []
        for component_id in self._components_id:
            icons_name.append(resources.DICTIONARY_ICON[component_id])
            icons_size.append(resources.DICTIONARY_ICON_SIZE[component_id])

        # If the connection is between a bus and an sspc, we shorten the length
        curated_connection_list = []

        for connections in self._connection_list:
            curated_connection_list.append((connections["source"], connections["target"]))

        return (
            self._components_name,
            curated_connection_list,
            self._components_type_class,
            self._components_om_type,
            icons_name,
            icons_size,
        )

    def produce_simplified_pt_file_copy(self):
        """
        This function was created after the observation that the more components there are in the
        powertrain, the longer it takes to run (duh). It is even more striking when running the
        optimization to find a new wing area (it can takes minutes). However, for that particular
        observation the whole propulsion chain is not needed. We indeed only need the propulsors
        to compute the slipstream effect and the propulsive load to check that the power rate is
        below 1. Consequently, and only for that particular application, we will produce a
        simplified powertrain file which contains only the required elements.
        """

        simplified_serializer = copy.deepcopy(self._serializer)

        self._get_components()

        retained_components = []

        # First, we pop all the components that we don't need
        for component_name, component_type_class, component_id in zip(
            self._components_name, self._components_type_class, self._components_id
        ):
            if (
                "propulsor" not in component_type_class
                and "propulsive_load" not in component_type_class
            ):
                simplified_serializer.data[KEY_PT_COMPONENTS].pop(component_name)
            else:
                retained_components.append(component_name)

        # Then we pop all the connections that don't involve the components we have
        self._get_connections()

        cured_connection_list = copy.deepcopy(self._connection_list)

        for connection in self._connection_list:
            if type(connection["source"]) is str:
                if connection["source"] not in retained_components:
                    cured_connection_list.remove(connection)
                else:
                    if type(connection["target"]) is str:
                        if connection["target"] not in retained_components:
                            cured_connection_list.remove(connection)
                    else:
                        if connection["target"][0] not in retained_components:
                            cured_connection_list.remove(connection)

            else:
                if connection["source"][0] not in retained_components:
                    cured_connection_list.remove(connection)
                else:
                    if type(connection["target"]) is str:
                        if connection["target"] not in retained_components:
                            cured_connection_list.remove(connection)
                    else:
                        if connection["target"][0] not in retained_components:
                            cured_connection_list.remove(connection)

        simplified_serializer.data[KEY_PT_CONNECTIONS] = cured_connection_list

        pt_file_copy_path = self._power_train_file.replace(".yml", "_temp_copy.yml")

        simplified_serializer.write(pt_file_copy_path)

        return pt_file_copy_path

    def get_current_to_set(
        self, inputs, propulsive_power_dict: dict, number_of_points: int
    ) -> dict:
        """
        Returns a list of the dict of current variable names and the value they should be set at
        for each of the subgraph. Dict will be empty if there is no current to set. The current to
        set are defined in the registered_components.py file.

        :param inputs: inputs vector, in the OpenMDAO format, which contains the value of the
        voltages to check
        :param propulsive_power_dict: dictionary with the propulsive power of each propulsor
        :param number_of_points: number of points in the data to check
        """

        # First we get voltage and power at each point but we first check that they are already
        # registered to avoid redoing unnecessary operations
        if not self._voltage_at_each_node:
            _ = self.get_voltage_to_set(inputs, number_of_points)

        if not self._power_at_each_node:
            _, _ = self.get_power_to_set(inputs, propulsive_power_dict)

        name_to_id = dict(zip(self._components_name, self._components_id))
        name_to_option = dict(zip(self._components_name, self._components_options))

        all_voltage_dict = copy.deepcopy(self._voltage_at_each_node)
        all_power_dict = copy.deepcopy(self._power_at_each_node)

        # First step is to remove all the sources inputs from the voltage setter since they won't
        # appear in the power setter
        for source in self.get_energy_consumption_list():
            source_input_name = source + "_in"
            if source_input_name in all_voltage_dict:
                all_voltage_dict.pop(source_input_name)

        # Something worth mentioning here. Due to the way the power_at_each_node dict was
        # constructed, the splitter inputs are doubled, whereas their current aren't, meaning the
        # power_at_each_node dict will always be longer or at worst the same size. This is the
        # one we will use to iterate on.

        all_current_dict = {}

        for node in all_power_dict:
            # first a quick check on whether the component is a splitter or not. Since we are
            # iterating on the nodes in the power dictionary, if a components ends with either
            # "_in_1" or "_in_2" it is a splitter
            component_name = (
                node.replace("_in_1", "")
                .replace("_in_2", "")
                .replace("_in", "")
                .replace("_out", "")
            )
            component_id = name_to_id[component_name]

            current_to_set = resources.DICTIONARY_I_TO_SET[component_id]

            for current in current_to_set:
                # These are tuple which contains the "in" or "out" tag plus the name of the variable

                # Some value have been filled with a default value for voltage because they are
                # not set, by the code. Turns out some current might be computed base on them so,
                # instead of using this default value we will use the value on the other side of
                # the component. E.g for the input of a dc/dc converter we will take its outputs
                # rather than an arbitrary value.
                voltage_node = all_voltage_dict[node]

                if all(voltage_node == DEFAULT_VOLTAGE_VALUE):
                    if "_in" in node:
                        other_side_component = node.replace("_in", "_out")
                    else:
                        other_side_component = node.replace("_out", "_in")
                    voltage_node = all_voltage_dict[other_side_component]

                # Some current correspond correspond to the current in one phase, in which case
                # we need to divide by three the obtained current
                if "one_phase" in current[0]:
                    factor = 3.0
                else:
                    factor = 1.0

                if current[1] == "in" and node.endswith("_in"):
                    variable_name = component_name + "." + current[0]

                    current = all_power_dict[node] / voltage_node / factor
                    all_current_dict[variable_name] = current

                elif current[1] == "out" and node.endswith("_out"):
                    variable_name = component_name + "." + current[0]

                    current = all_power_dict[node] / voltage_node / factor
                    all_current_dict[variable_name] = current

            # If the component is a battery, there will be no "official" current to set,
            # but as seen empirically if the battery is directly connected to a bus and the
            # current is not set properly, it might prevent the code from converging.
            if (
                name_to_id[component_name] == "fastga_he.pt_component.battery_pack"
                and name_to_option[component_name]
            ):
                # We just don't deal with battery inputs
                if node.endswith("_out"):
                    voltage_node = all_voltage_dict[node]
                    current = all_power_dict[node] / voltage_node
                    variable_name = component_name + ".dc_current_out"

                    all_current_dict[variable_name] = current

        return all_current_dict

    def get_battery_list(self) -> Tuple[list, list]:
        """
        Returns the list of components inside the power train that are batteries. This function is
        used to see where the electricity is stored in the powertrain for the LCA. For now, it tests
        the id of the component, but it should be more generic in the future for components like
        super-capacitors and others.
        """

        self._get_components()
        components_names = []
        components_types = []

        for component_id, component_name, component_type in zip(
            self._components_id, self._components_name, self._components_type
        ):
            if "battery_pack" in component_id:
                components_names.append(component_name)
                components_types.append(component_type)

        return components_names, components_types

    def get_lca_production_element_list(self) -> Dict:
        # I hate doing that here, but it prevents a circular import
        import fastga_he.models.propulsion.components as he_comp

        variables_names_mass = self.get_mass_element_lists()

        # one possible way to get the path to the template LCA modules is to trace them back to
        # one class we know is ner them such as the Sizing, Performances, ... The downside is
        # that it will only work for package located with the component in the default delivery,
        # so it won't work like fast-oad plugins. But for now it works

        clean_dict = {}

        for component_name, component_om_type, variable_name_mass in zip(
            self._components_name, self._components_om_type, variables_names_mass
        ):
            sizing_group = he_comp.__dict__["Sizing" + component_om_type]
            path_to_sizing_file = pathlib.Path(sys.modules[sizing_group.__module__].__file__)

            # The sizing class is defined inside components/sizing...py and the lca template is in
            # components/lca_resources/lca_conf.yml so:

            path_to_lca_prod_conf_template = (
                path_to_sizing_file.parents[0] / "lca_resources/lca_conf_prod.yml"
            )

            if pth.exists(path_to_lca_prod_conf_template):
                # Now we open the file, convert each line to an element of a list and replace the
                # anchors by whatever is required
                clean_lines = []
                with open(path_to_lca_prod_conf_template, "r") as template_file:
                    for line in template_file.readlines():
                        # Important to add in the definition of the custom attribute, the name of
                        # the phase as the code writes it in the lca conf file.
                        # Also need to make sure we ask for the mass per functional unit and not
                        # the total mass
                        clean_lines.append(
                            line.replace(
                                "value: ANCHOR_COMPONENT_NAME",
                                "value: " + component_name + "_production",
                            )
                            .replace("ANCHOR_COMPONENT_NAME", component_name)
                            .replace(
                                "ANCHOR_COMPONENT_MASS",
                                variable_name_mass.replace("mass", "mass_per_fu").replace(
                                    ":", "__"
                                ),
                            )
                            .replace(
                                "ANCHOR_COMPONENT_LENGTH",
                                variable_name_mass.replace("mass", "length_per_fu").replace(
                                    ":", "__"
                                ),
                            )
                            .replace(
                                "ANCHOR_COMPONENT_MATERIAL",
                                variable_name_mass.replace("mass", "material").replace(":", "__"),
                            )
                        )

                clean_dict[component_name] = clean_lines

        return clean_dict

    def get_lca_use_phase_element_list(self) -> Tuple[Dict, List]:
        # I still hate doing that here, but it prevents a circular import
        import fastga_he.models.propulsion.components as he_comp

        # We will start with the assumption that if a component of the powertrain has an impact
        # in the use phase, it will have a computation of the emissions, even if they can be nil.

        clean_dict = {}

        species_list = []

        for component_name, component_om_type, component_type in zip(
            self._components_name, self._components_om_type, self._components_type
        ):
            sizing_group = he_comp.__dict__["Sizing" + component_om_type]
            path_to_sizing_file = pathlib.Path(sys.modules[sizing_group.__module__].__file__)

            # The sizing class is defined inside components/sizing...py and the lca template is in
            # components/lca_resources/lca_conf.yml so:

            path_to_lca_use_conf_template = (
                path_to_sizing_file.parents[0] / "lca_resources/lca_conf_use.yml"
            )

            if pth.exists(path_to_lca_use_conf_template):
                # If the component has an impact on the use phase, it must release species in the
                # air, which means it must have a species list. We intersect those list to have the
                # names of the species release by the power train  and update the NAME_TO_UNIT
                # dict in the lca_core script.
                pre_lca_group = he_comp.__dict__["PreLCA" + component_om_type]()
                species_list = species_list + pre_lca_group.species_list

                clean_lines = []
                with open(path_to_lca_use_conf_template, "r") as template_file:
                    for line in template_file.readlines():
                        # Important to add in the definition of the custom attribute, the name of
                        # the phase as the code writes it in the lca conf file.

                        # If an anchor for an emission is added, we put the right variable name
                        if "ANCHOR_EMISSION" in line:
                            line_to_add = line.replace(
                                "ANCHOR_EMISSION_",
                                "data__LCA__operation__he_power_train__"
                                + component_type
                                + "__"
                                + component_name
                                + "__",
                            )
                            line_to_add = line_to_add.replace("\n", "")
                            line_to_add = line_to_add + "_per_fu\n"
                        else:
                            line_to_add = line.replace(
                                "value: ANCHOR_COMPONENT_NAME",
                                "value: " + component_name + "_operation",
                            ).replace("ANCHOR_COMPONENT_NAME", component_name)

                        clean_lines.append(line_to_add)

                clean_dict[component_name] = clean_lines

        return clean_dict, list(set(species_list))

    def get_lca_manufacturing_phase_element_list(self) -> Tuple[Dict, List]:
        """
        Get a dict with all the lines to add to the LCA configuration file for the manufacturing
        phase. Theoretically, the manufacturing contains the assembly of the airframe plus tests
        plus the construction of the assembly plant. In our case, the assembly plant will be
        discarded and because we lack data, the assembly of the airframe has been aggregated in the
        production. So all that remains are the line tests, which will be very similar to the use
        phase. Except we won't attribute emission to each component (which means not adding the
        custom attributes), however we'll need to differentiate each CO2, NOx emissions ... so we'll
        tag them with component name.
        """
        # I still hate doing that here, but it prevents a circular import
        import fastga_he.models.propulsion.components as he_comp

        # We will start with the assumption that if a component of the powertrain has an impact
        # in the use phase, it will have a computation of the emissions, even if they can be nil.

        clean_dict = {}

        species_list = []

        for component_name, component_om_type, component_type in zip(
            self._components_name, self._components_om_type, self._components_type
        ):
            sizing_group = he_comp.__dict__["Sizing" + component_om_type]
            path_to_sizing_file = pathlib.Path(sys.modules[sizing_group.__module__].__file__)

            # The sizing class is defined inside components/sizing...py and the lca template is in
            # components/lca_resources/lca_conf.yml so:

            path_to_lca_use_conf_template = (
                path_to_sizing_file.parents[0] / "lca_resources/lca_conf_use.yml"
            )

            if pth.exists(path_to_lca_use_conf_template):
                # If the component has an impact on the use phase, it must release species in the
                # air, which means it must have a species list. We intersect those list to have the
                # names of the species release by the power train  and update the NAME_TO_UNIT
                # dict in the lca_core script.
                pre_lca_group = he_comp.__dict__["PreLCA" + component_om_type]()
                species_list = species_list + pre_lca_group.species_list

                clean_lines = []
                with open(path_to_lca_use_conf_template, "r") as template_file:
                    lines = template_file.readlines()
                    for idx, line in enumerate(lines):
                        # Important to add in the definition of the custom attribute, the name of
                        # the phase as the code writes it in the lca conf file.

                        # Since we are using the same template as the use phase we remove the lines
                        # with the custom attributes. Or rather, we simply don't add them which
                        # means continuing to the next loop of the for
                        if self.belongs_to_custom_attribute_definition(line, idx, lines):
                            continue

                        # If an anchor for an emission is added, we put the right variable name
                        if "ANCHOR_EMISSION" in line:
                            line_to_add = line.replace(
                                "ANCHOR_EMISSION_",
                                "data__LCA__manufacturing__he_power_train__"
                                + component_type
                                + "__"
                                + component_name
                                + "__",
                            )
                            line_to_add = line_to_add.replace("\n", "")
                            line_to_add = line_to_add + "_per_fu\n"
                        else:
                            line_to_add = line.replace("ANCHOR_COMPONENT_NAME", component_name)

                        clean_lines.append(line_to_add)

                clean_dict[component_name] = clean_lines

        return clean_dict, list(set(species_list))

    def get_lca_distribution_phase_element_list(self) -> Tuple[Dict, List]:
        """
        Get a dict with all the lines to add to the LCA configuration file for the distribution
        phase. Will be computed all the time but won't be used if the delivery method is the train.

        Lots of commonality with get_lca_manufacturing_phase_element_list
        # TODO: Refactor ? Refactor !
        """
        # I still hate doing that here, but it prevents a circular import
        import fastga_he.models.propulsion.components as he_comp

        clean_dict = {}

        species_list = []

        for component_name, component_om_type, component_type in zip(
            self._components_name, self._components_om_type, self._components_type
        ):
            sizing_group = he_comp.__dict__["Sizing" + component_om_type]
            path_to_sizing_file = pathlib.Path(sys.modules[sizing_group.__module__].__file__)

            # The sizing class is defined inside components/sizing...py and the lca template is in
            # components/lca_resources/lca_conf.yml so:

            path_to_lca_use_conf_template = (
                path_to_sizing_file.parents[0] / "lca_resources/lca_conf_use.yml"
            )

            if pth.exists(path_to_lca_use_conf_template):
                # If the component has an impact on the use phase, it must release species in the
                # air, which means it must have a species list. We intersect those list to have the
                # names of the species release by the power train  and update the NAME_TO_UNIT
                # dict in the lca_core script.
                pre_lca_group = he_comp.__dict__["PreLCA" + component_om_type]()
                species_list = species_list + pre_lca_group.species_list

                clean_lines = []
                with open(path_to_lca_use_conf_template, "r") as template_file:
                    lines = template_file.readlines()
                    for idx, line in enumerate(lines):
                        # Important to add in the definition of the custom attribute, the name of
                        # the phase as the code writes it in the lca conf file.

                        # Since we are using the same template as the use phase we remove the lines
                        # with the custom attributes. Or rather, we simply don't add them which
                        # means continuing to the next loop of the for
                        if self.belongs_to_custom_attribute_definition(line, idx, lines):
                            continue

                        # If an anchor for an emission is added, we put the right variable name
                        if "ANCHOR_EMISSION" in line:
                            line_to_add = line.replace(
                                "ANCHOR_EMISSION_",
                                "data__LCA__distribution__he_power_train__"
                                + component_type
                                + "__"
                                + component_name
                                + "__",
                            )
                            line_to_add = line_to_add.replace("\n", "")
                            line_to_add = line_to_add + "_per_fu\n"
                        else:
                            line_to_add = line.replace("ANCHOR_COMPONENT_NAME", component_name)

                        clean_lines.append(line_to_add)

                clean_dict[component_name] = clean_lines

        return clean_dict, list(set(species_list))

    def _check_existing_connection_cache_instance(self):
        """
        Checks the cache to see if an instance of the cache already exists and is usable. Usable
        means there was no modification to the powertrain configuration file.
        """

        # If cache is empty, no instance is usable
        if not FASTGAHEPowerTrainConfigurator._cache:
            return False

        # If the powertrain configuration file is a temporary copy or dedicated for a test,
        # the connection test will be omitted
        if self._power_train_file.endswith("_temp_copy.yml"):
            return True

        # If cache is not empty but there is no instance of that particular configuration file, no
        # instance is usable.
        if not (
            FASTGAHEPowerTrainConfigurator._cache[self._power_train_file].get("skip_test")
            or FASTGAHEPowerTrainConfigurator._cache[self._power_train_file].get("last_mod_time")
        ):
            return False

        if FASTGAHEPowerTrainConfigurator._cache[self._power_train_file]["skip_test"]:
            return True

        # Finally, if an instance exists, but it has been modified since, no instance is usable.
        if (
            FASTGAHEPowerTrainConfigurator._cache[self._power_train_file]["last_mod_time"]
            < pathlib.Path(self._power_train_file).lstat().st_mtime
        ):
            return False

        return True

    def _add_connection_check_cache_instance(self):
        """
        In the case where no instance were usable and the compilation needed to be redone, we add
        said compilation to the cache.
        """

        FASTGAHEPowerTrainConfigurator._cache[self._power_train_file]["last_mod_time"] = (
            pathlib.Path(self._power_train_file).lstat().st_mtime
        )
        FASTGAHEPowerTrainConfigurator._cache[self._power_train_file]["skip_test"] = False

    @staticmethod
    def belongs_to_custom_attribute_definition(line, line_idx, lines_to_inspect) -> bool:
        """
        Utility function to detect if the line is part of the definition of a custom attribute.
        """
        if (
            "custom_attributes" in line
            and 'attribute: "component"' in lines_to_inspect[line_idx + 1]
        ):
            return True

        # Trying to foolproof but if the format is respected it should not cause issues
        if line_idx >= 1:
            if (
                'attribute: "component"' in line
                and "custom_attributes" in lines_to_inspect[line_idx - 1]
            ):
                return True

        # Trying to foolproof but if the format is respected it should not cause issues
        if line_idx >= 2:
            if (
                'attribute: "component"' in lines_to_inspect[line_idx - 1]
                and "custom_attributes" in lines_to_inspect[line_idx - 2]
            ):
                return True

        return False


class _YAMLSerializer(ABC):
    """YAML-format serializer."""

    def __init__(self):
        self._data = None

    @property
    def data(self):
        return self._data

    def read(self, file_path: str):
        yaml = YAML(typ="safe")
        with open(file_path) as yaml_file:
            self._data = yaml.load(yaml_file)

    def write(self, file_path: str):
        yaml = YAML()
        yaml.default_flow_style = False
        with open(file_path, "w") as file:
            yaml.dump(self._data, file)


def format_to_array(input_array: np.ndarray, number_of_points: int) -> np.ndarray:
    """
    Takes an inputs which is either a one-element array or a multi-element array and formats it.
    """

    if len(input_array):
        output_array = np.full(number_of_points, input_array[0])
    else:
        output_array = input_array

    return output_array
