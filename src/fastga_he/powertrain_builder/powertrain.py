"""
Module for the construction of all the groups necessary for the proper interaction of the
power train module with the aircraft sizing modules from FAST-OAD-GA based on the power train file.
"""
# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import copy
import json
import logging
import os.path as pth

from abc import ABC
from importlib.resources import open_text
from typing import Tuple, List

import numpy as np
from jsonschema import validate
from ruamel.yaml import YAML

import networkx as nx

from .exceptions import (
    FASTGAHEUnknownComponentID,
    FASTGAHEUnknownOption,
    FASTGAHEComponentsNotIdentified,
    FASTGAHESingleSSPCAtEndOfLine,
    FASTGAHEIncoherentVoltage,
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


class FASTGAHEPowerTrainConfigurator:
    """
    Class for the configuration of the components necessary for the performances and sizing of the
    power train.

    :param power_train_file_path: if provided, power train will be read directly from it
    """

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

        # Because of their very peculiar role, we will scan the architecture for any SSPC defined
        # by the user and whether or not they are at the output of a bus, because a specific
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

        if power_train_file_path:
            self.load(power_train_file_path)

    def load(self, power_train_file):
        """
        Reads the power train definition

        :param power_train_file: Path to the file to open.
        """

        self._power_train_file = pth.abspath(power_train_file)

        self._serializer = _YAMLSerializer()
        self._serializer.read(self._power_train_file)

        # Syntax validation
        with open_text(resources, JSON_SCHEMA_NAME) as json_file:
            json_schema = json.loads(json_file.read())
        validate(self._serializer.data, json_schema)

        for key in self._serializer.data:
            if key not in json_schema["properties"].keys():
                _LOGGER.warning('Power train file: "%s" is not a FAST-OAD-GA-HE key.', key)

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

        components_list = self._serializer.data.get(KEY_PT_COMPONENTS)

        components_id = []
        components_position = []
        components_name_list = []
        components_name_id_list = []
        components_type_list = []
        components_om_type_list = []
        components_options_list = []
        components_promote_list = []
        components_type_class_list = []
        components_perf_watchers_list = []

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

            components_name_list.append(component_name)
            components_name_id_list.append(resources.DICTIONARY_CN_ID[component_id])
            components_type_list.append(resources.DICTIONARY_CT[component_id])
            components_om_type_list.append(resources.DICTIONARY_CN[component_id])
            components_promote_list.append(resources.DICTIONARY_PT[component_id])
            components_type_class_list.append(resources.DICTIONARY_CTC[component_id])
            components_perf_watchers_list.append(resources.DICTIONARY_MP[component_id])

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
        self._components_type_class = components_type_class_list
        self._components_perf_watchers = components_perf_watchers_list

    def _get_connections(self):
        """
        This function inspects all the connections detected in the power train file and prepare
        the list necessary to do the connections in the performance file.

        The _get_components method must be ran before hand.
        """

        # First check that the _get_components method has been ran
        if not self._components_name:
            raise FASTGAHEComponentsNotIdentified(
                "The _get_components must be run before running the _get_connections method"
            )

        connections_list = self._serializer.data.get(KEY_PT_CONNECTIONS)
        self._connection_list = connections_list

        # Create a dictionary to translate component name back to component_id to identify
        # outputs and inputs in each case
        translator = dict(zip(self._components_name, self._components_id))

        openmdao_output_list = []
        openmdao_input_list = []

        for connection in connections_list:

            # Check in case the source or target is not a string but an array, meaning we are
            # dealing with a component which might have multiple inputs/outputs (buses, gearbox,
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

    def get_distance_from_propulsive_load(self):

        propulsor_name = []
        propulsive_load_names = []

        # First and for reason that will appear clear later, we get a list of propulsor
        for component_type_class, component_name in zip(
            self._components_type_class, self._components_name
        ):
            if "propulsor" in component_type_class:
                propulsor_name.append(component_name)

        self._construct_connection_graph()
        graph = self._connection_graph

        # We now get a list of propulsive loads. Because of the use envisioned for this function (
        # mostly post-processing), the ICE won't be considered a propulsive load if no propulsor
        # is attached to it
        for component_type_class, component_name in zip(
            self._components_type_class, self._components_name
        ):
            if "propulsive_load" == [component_type_class]:
                propulsive_load_names.append(component_name)

            # This case will correspond to ICE/turbomachinery
            elif "propulsive_load" in component_type_class:

                # Check whether or not at least one neighbor is a propulsor
                neighbors = list(graph.neighbors(component_name))
                if set(neighbors).intersection(propulsor_name):
                    propulsive_load_names.append(component_name)

        distance_from_propulsive_load = {}
        connections_length_between_nodes = dict(nx.all_pairs_shortest_path_length(graph))

        for component_name in self._components_name:
            min_distance = np.inf
            for prop_load in propulsive_load_names:
                distance_to_load = connections_length_between_nodes[component_name][prop_load]
                if distance_to_load < min_distance:
                    min_distance = distance_to_load

            distance_from_propulsive_load[component_name] = min_distance

        return distance_from_propulsive_load, propulsive_load_names

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
            self._components_position,
        )

    def get_performances_element_lists(self) -> tuple:
        """
        Returns the list of parameters necessary to create the performances group based on what is
        inside the power train file.
        """

        self._get_components()
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

        self._get_components()
        components_perf_watchers_name_organised_list = []
        components_perf_watchers_unit_organised_list = []
        components_name_organised_list = []

        for component_name, components_perf_watchers in zip(
            self._components_name, self._components_perf_watchers
        ):
            for components_perf_watcher in components_perf_watchers:
                key, value = list(components_perf_watcher.items())[0]
                components_name_organised_list.append(component_name)
                components_perf_watchers_name_organised_list.append(key)
                components_perf_watchers_unit_organised_list.append(value)

        return (
            components_name_organised_list,
            components_perf_watchers_name_organised_list,
            components_perf_watchers_unit_organised_list,
        )

    def get_graphs_connected_voltage(self) -> list:
        """
        This function returns a list of graphs of connected PT components that have more or less
        the same imposed voltage. What is meant by that is that since some component impose the
        voltage on the circuit while other have independent I/O in terms of voltage e.g the DC/DC
        converter this will make it so that there might some subgraph of the architecture with
        different connected voltage.
        """

        self._get_components()
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

        sub_graphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

        return sub_graphs

    def _list_voltage_coherence_to_check(self) -> list:
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

        return sub_graphs_voltage_setter

    def check_voltage_coherence(self, inputs, number_of_points: int):
        """
        Check that all the sub graphs of independent voltage are compatible, meaning that if
        there is more than one component that sets the voltage, they have the same target voltage.

        :param inputs: inputs vector, in the OpenMDAO format, which contains the value of the
        voltages to check
        :param number_of_points: number of points in the data to check
        """

        sub_graphs_voltage_setters = self._list_voltage_coherence_to_check()

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
        sub_graphs = self.get_graphs_connected_voltage()
        # TODO: the line above is repeated in the function below, could be improved !
        sub_graphs_voltage_setters = self._list_voltage_coherence_to_check()

        name_to_type = dict(zip(self._components_name, self._components_type))
        name_to_id = dict(zip(self._components_name, self._components_id))

        final_list = []

        for sub_graph, sub_graph_voltage_setters in zip(sub_graphs, sub_graphs_voltage_setters):
            # First and foremost, we get the value that will serve as the for the setting of the
            # voltage in this subgraph. If there are not setters in this subgraph we just pass along

            if not sub_graph_voltage_setters:
                continue

            spl = dict(nx.all_pairs_shortest_path_length(sub_graph))
            voltage_dict_subgraph = {}

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

            # We now transform it in the proper array
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

                distance_from_setter = spl[node][voltage_setter] - 1

                for voltage_to_set in voltages_to_set:

                    variable_name = component_name + "." + voltage_to_set
                    # The further we go from the setter the lower the voltage because of
                    # component efficiencies
                    value_of_voltage = reference_voltage * 0.995 ** (distance_from_setter / 2.0)
                    voltage_dict_subgraph[variable_name] = value_of_voltage

            final_list.append(voltage_dict_subgraph)

        return final_list

    def get_network_elements_list(self) -> tuple:
        """
        Returns the name of the components and their connections for the visualisation of the
        power train as a network graph.
        """

        self._get_components()
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
            icons_name,
            icons_size,
        )


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
