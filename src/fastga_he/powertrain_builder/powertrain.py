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

import re

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
    FASTGAHEImpossiblePair,
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

DEFAULT_VOLTAGE_VALUE = 737.800


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

        # Contains the list of aircraft inputs that are necessary to promote in the slipstream
        # modules for the code to work
        self._components_slipstream_promotes = None

        # Contains the list of variables that needs to be promoted from the performances
        # computations to slipstream computation
        self._components_performances_to_slipstream = None

        # Contains a list with, for each component, a boolean telling whether or not the component
        # needs the flaps position for the computation of the slipstream effects
        self._components_slipstream_flaps = None

        # Contains a list with, for each component, a boolean telling whether or not the component
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

        # Contains the list of all boolean telling whether or not the components will make the
        # aircraft weight vary during flight
        self._components_makes_mass_vary = None

        # Contains the list of all boolean telling whether or not the components are energy
        # sources that do not make the aircraft vary (ergo they will have a non-nil unconsumable
        # energy)
        self._source_does_not_make_mass_vary = None

        # Contains the list of an initial guess of the components efficiency. Is used to compute
        # the initial of the currents and power of each component
        self._components_efficiency = None

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

        # We will work under the assumption that is one list is empty, all are hence only one if
        # statement. This allows us to know whether or not retriggering the identification of
        # components is necessary

        if self._components_id is None:
            self._generate_components_list()

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
                # propeller2 is symmetrical to propeller1) it will have the same name and we
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

            # The possibility to connect a battery directly to a bus has been added. However,
            # to make it backward compatible (whatever it means today because I have no used) and
            # to impart less burden during the writing of the pt file, we won't ask the user to
            # set the option accordingly, rather, we will do it here.

            if target_id == "fastga_he.pt_component.battery_pack" and (
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
                # If there are gearboxes among neighbor, we also check for the neighbor of the
                # gearbox. Not very generic way to do things :/
                else:
                    name_to_id = dict(zip(self._components_name, self._components_id))
                    for neighbor in set(neighbors):
                        if (
                            name_to_id[neighbor] == "fastga_he.pt_component.speed_reducer"
                            or name_to_id[neighbor] == "fastga_he.pt_component.planetary_gear"
                        ):
                            neighbors_gb = list(graph.neighbors(neighbor))
                            if set(neighbors_gb).intersection(propulsor_name):
                                propulsive_load_names.append(component_name)

        distance_from_propulsive_load = {}
        connections_length_between_nodes = dict(nx.all_pairs_shortest_path_length(graph))

        for component_name in self._components_name:

            # When there are two separate sub propulsion chain in the same propulsion file,
            # these line will cause an issue because, as it will browse all propulsive load he
            # will attempt to reach loads he is not connected to and therefore not in
            # connections_length_between_nodes. So first we must make sure to only browse
            # connected loads.

            connected_components = list(connections_length_between_nodes[component_name].keys())
            connected_propulsive_loads = list(
                set(propulsive_load_names) & set(connected_components)
            )

            min_distance = np.inf
            for prop_load in connected_propulsive_loads:
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

        self._get_components()
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
        component_types = []

        for component_type_class, component_name, component_type in zip(
            self._components_type_class, self._components_name, self._components_type
        ):
            if "tank" in component_type_class:
                components_names.append(component_name)
                component_types.append(component_type)

        return components_names, component_types

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
        # We need to trigger the generation of the connections because that what triggers the
        # identification of batteries directly connected to a bus
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
        name_to_ct = dict(zip(self._components_name, self._components_type))
        name_to_option = dict(zip(self._components_name, self._components_options))

        final_list = []
        voltage_at_each_node = {}

        for sub_graph, sub_graph_voltage_setters in zip(sub_graphs, sub_graphs_voltage_setters):
            # First and foremost, we get the value that will serve as the for the setting of the
            # voltage in this subgraph. If there are not setters in this subgraph we just pass along

            spl = dict(nx.all_pairs_shortest_path_length(sub_graph))
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
            number_of_cell_in_series = float(
                inputs[
                    PT_DATA_PREFIX + component_type + ":" + component_name + ":module:number_cells"
                ]
            )

        except RuntimeError as e:
            error_message = e.args[0]
            abs_names = re.findall(r"\[.*?\]", error_message)[0][1:-1].replace(" ", "").split(",")
            abs_name = abs_names[0]
            split_abs_name = abs_name.split(".")
            # Sometimes the number of cells is not one deep but two deep so we try both
            try:
                proper_abs_name = ".".join(split_abs_name[1:])
                number_of_cell_in_series = float(inputs[proper_abs_name])
            except KeyError:
                proper_abs_name = ".".join(split_abs_name[2:])
                number_of_cell_in_series = float(inputs[proper_abs_name])

        return number_of_cell_in_series

    def get_independent_sub_propulsion_chain(self):
        """
        This function returns a list of graphs of connected PT sub propulsion chain. As a
        prevision for next step all component will be split between their inputs and outputs to
        allow to include efficiency.
        """

        # TODO: very similar to self.get_graphs_connected_voltage(), think of refactoring ?

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

    def get_power_on_each_node(self, graph, inputs, propulsive_power_dict) -> dict:
        """
        Returns a dictionary which will contain the power at each node of a graph based on the
        propulsive power at its propulsor, on the assumed efficiencies of its components and the
        value of power split on its splitter.
        """

        copied_graph = copy.deepcopy(graph)
        name_to_id = dict(zip(self._components_name, self._components_id))
        name_to_eta = dict(zip(self._components_name, self._components_efficiency))

        # For each graph we attribute a priority to the nodes which is going to impose the order
        # in which we make the computation. Propulsive loads will have the highest priority (0)
        # and then each neighboring nodes will have the priority just below. Slight detail nodes
        # with more than one neighbor will have a priority which will be slightly below the
        # biggest priority among the neighbour. E.g:
        # Node 1 (0) --- Node 2 (1) --- Node 3 (2) \
        #                                           Node 6 (3)
        #                Node 4 (0) --- Node 5 (1) /

        propulsive_loads_proper_name = list(propulsive_power_dict.keys())

        # So we start, for each propulsive load by exploring the branch until someone has more
        # than two neighbor

        nodes_with_power = {}

        for propulsive_load_name in propulsive_loads_proper_name:
            nodes_with_power[propulsive_load_name] = propulsive_power_dict[propulsive_load_name]

        problematic_nodes = copy.deepcopy(propulsive_loads_proper_name)

        while problematic_nodes:

            problematic_nodes, copied_graph = self.set_priority_in_graph(
                copied_graph, nodes_with_power, problematic_nodes
            )

            # At this point, we painted the priority as mush as we could but we have encountered
            # some "problematic nodes" which we must dealt with. They can be of two types: buses
            # (multiple outputs but one input) or splitter (one output but two inputs)

            new_problematic_nodes = copy.deepcopy(problematic_nodes)

            for problematic_node in problematic_nodes:
                associated_component_name = problematic_node.replace("_in", "")
                associated_component_name = associated_component_name.replace("_out", "")

                associated_component_type = name_to_id[associated_component_name]

                # If the associated component is a bus it means it has multiple outputs so we
                # need to look for their priority take the highest among them and add 1
                if associated_component_type == "fastga_he.pt_component.dc_bus":

                    problematic_nodes_to_add = []

                    power = 0
                    for neighbor in graph.adj[problematic_node]:
                        if neighbor in list(nodes_with_power.keys()):
                            power += nodes_with_power[neighbor]

                    nodes_with_power[problematic_node] = power

                    # Then we add the bus inputs (there may be more than 1) in the list of
                    # "problematic nodes" and we add their priority
                    for neighbor in graph.adj[problematic_node]:
                        if neighbor not in list(nodes_with_power.keys()):

                            nodes_with_power[neighbor] = (
                                power / name_to_eta[associated_component_name]
                            )
                            problematic_nodes_to_add.append(neighbor)

                    new_problematic_nodes.remove(problematic_node)
                    new_problematic_nodes += problematic_nodes_to_add

                # If the associated component is a plain gearbox it means it has multiple
                # outputs so we need to look for their priority take the highest among them
                # and add 1
                if associated_component_type == "fastga_he.pt_component.gearbox":

                    problematic_nodes_to_add = []

                    power = 0
                    for neighbor in graph.adj[problematic_node]:
                        if neighbor in list(nodes_with_power.keys()):
                            power += nodes_with_power[neighbor]

                    nodes_with_power[problematic_node] = power

                    # Then we add the gearbox inputs (there may be more than 1) in the list of
                    # "problematic nodes" and we add their priority
                    for neighbor in graph.adj[problematic_node]:
                        if neighbor not in list(nodes_with_power.keys()):
                            nodes_with_power[neighbor] = (
                                power / name_to_eta[associated_component_name]
                            )
                            problematic_nodes_to_add.append(neighbor)

                    # Then we add the input as a problematic node
                    new_problematic_nodes.remove(problematic_node)
                    new_problematic_nodes += problematic_nodes_to_add

                # If it is a planetary gearbox
                elif associated_component_type == "fastga_he.pt_component.planetary_gear":

                    problematic_nodes_to_add = []

                    # This node does not have a priority just yet, so we'll first need to take a
                    # look at it
                    power = None

                    for neighbor in graph.adj[problematic_node]:
                        if neighbor in list(nodes_with_power.keys()):
                            power = nodes_with_power[neighbor]

                    for neighbor in graph.adj[problematic_node]:

                        neighbor_counter = 1

                        if neighbor not in list(nodes_with_power.keys()):
                            # If we haven't treated the neighbor yet, it means its an input of the
                            # gearbox. We will add a fake input node and do the connection to
                            # that neighbor. We should also take the opportunity to see whether
                            # or not this is the gearbox "priority" input

                            # First we identify which input of the splitter in matches, first
                            # find the name of the component
                            neighbor_components_name = neighbor.replace("_in", "")
                            neighbor_components_name = neighbor_components_name.replace("_out", "")

                            input_name = neighbor_components_name + ".shaft_power_out"

                            # We look at the number of the corresponding gearbox input. Will be a
                            # bit farfetched as we need to find using the current neighbor. For
                            # gearboxes, the two connexion are rpm and shaft power and they are
                            # both output of the input side of the gearbox, so we'll have to use
                            # the input list
                            # We look at the number of the corresponding splitter input
                            index = self._components_connection_inputs.index(input_name)
                            gearbox_input_name = self._components_connection_outputs[index]

                            (
                                primary_input_power,
                                secondary_power_output,
                            ) = self.gearbox_power_inputs(
                                inputs=inputs,
                                components_name=associated_component_name,
                                power_output=power,
                            )

                            # If it ends with a "1", its the priority input. We add the node as a
                            # problematic node, we set its priority and we add the node to the
                            # graph as well as the proper edge.
                            if gearbox_input_name[-1] == "1":
                                node_name = associated_component_name + "_in_1"
                                problematic_nodes_to_add.append(node_name)
                                nodes_with_power[node_name] = primary_input_power

                                copied_graph.add_edge(neighbor, node_name)

                            else:
                                node_name = (
                                    associated_component_name + "_in_" + str(neighbor_counter + 1)
                                )
                                problematic_nodes_to_add.append(node_name)
                                nodes_with_power[node_name] = secondary_power_output

                                copied_graph.add_edge(neighbor, node_name)

                                neighbor_counter += 1

                    # Now we remove the splitter and instead add its two neighbor, which will
                    # serve as new branch start

                    new_problematic_nodes.remove(problematic_node)
                    new_problematic_nodes += problematic_nodes_to_add

                # If it is a splitter
                elif associated_component_type == "fastga_he.pt_component.dc_splitter":

                    problematic_nodes_to_add = []

                    # This node does not have a priority just yet, so we'll first need to take a
                    # look at it
                    power = None

                    for neighbor in graph.adj[problematic_node]:
                        if neighbor in list(nodes_with_power.keys()):
                            power = nodes_with_power[neighbor]

                    for neighbor in graph.adj[problematic_node]:

                        neighbor_counter = 1

                        if neighbor not in list(nodes_with_power.keys()):
                            # If we haven't treated the neighbor yet, it means its an input of the
                            # splitter. We will add a fake input node and do the connection to
                            # that neighbor. We should also take the opportunity to see whether
                            # or not this is the splitter "priority" input

                            # First we identify which input of the splitter in matches, first
                            # find the name of the component
                            neighbor_components_name = neighbor.replace("_in", "")
                            neighbor_components_name = neighbor_components_name.replace("_out", "")

                            # Then identify which current output it correspond to. Here if the
                            # component in question is not an sspc, the following should work.
                            # Else we will try both way

                            output_name = neighbor_components_name + ".dc_current_out"
                            if (
                                name_to_id[neighbor_components_name]
                                == "fastga_he.pt_component.dc_sspc"
                            ):
                                if output_name not in self._components_connection_outputs:
                                    output_name = neighbor_components_name + ".dc_current_in"

                            # We look at the number of the corresponding splitter input
                            index = self._components_connection_outputs.index(output_name)
                            splitter_input_name = self._components_connection_inputs[index]

                            (
                                primary_input_power,
                                secondary_power_output,
                            ) = self.splitter_power_inputs(
                                inputs=inputs,
                                components_name=associated_component_name,
                                power_output=power,
                            )

                            # If it ends with a "1", its the priority input. We add the node as a
                            # problematic node, we set its priority and we add the node to the
                            # graph as well as the proper edge.
                            if splitter_input_name[-1] == "1":
                                node_name = associated_component_name + "_in_1"
                                problematic_nodes_to_add.append(node_name)
                                nodes_with_power[node_name] = primary_input_power

                                copied_graph.add_edge(node_name, neighbor)

                            else:
                                node_name = (
                                    associated_component_name + "_in_" + str(neighbor_counter + 1)
                                )
                                problematic_nodes_to_add.append(node_name)
                                nodes_with_power[node_name] = secondary_power_output

                                copied_graph.add_edge(node_name, neighbor)

                                neighbor_counter += 1

                    # Now we remove the splitter and instead add its two neighbor, which will
                    # serve as new branch start

                    new_problematic_nodes.remove(problematic_node)
                    new_problematic_nodes += problematic_nodes_to_add

                # Should be a fuel system. They are a bit specific because they can have multiple
                # inputs and multiple outputs
                elif associated_component_type == "fastga_he.pt_component.fuel_system":

                    problematic_nodes_to_add = []

                    # This node does not have a priority just yet, so we'll first need to take a
                    # look at it. Fuel system can have multiple output, so we'll simply sum the
                    # one whose power we know
                    power = 0.0
                    neighbor_counter = 0

                    for neighbor in graph.adj[problematic_node]:
                        if neighbor in list(nodes_with_power.keys()):
                            power += nodes_with_power[neighbor]
                            neighbor_counter += 1

                    input_power_dict = self.fuel_system_power_inputs(
                        inputs=inputs,
                        components_name=associated_component_name,
                        power_output=power,
                    )

                    # Once we have the power at the output, not that, we check how many engine we
                    # had at the output, it there was only one, the code can proceed. Else,
                    # we need to move to the input of the fuel system and then we can proceed
                    if neighbor_counter > 1:
                        # We add the out not to the list of stuff to remove and move the
                        # problematic node to the input.
                        nodes_with_power[problematic_node] = power
                        fuel_system_input_name = problematic_node.replace("out", "in")

                        new_problematic_nodes.remove(problematic_node)

                        problematic_node = fuel_system_input_name

                        new_problematic_nodes.append(problematic_node)

                    for neighbor in graph.adj[problematic_node]:

                        if neighbor not in list(nodes_with_power.keys()):
                            # If we haven't treated the neighbor yet, it means its an input of the
                            # fuel system. We will add a fake input node and do the connection to
                            # that neighbor.

                            # First we identify which input of the fuel system it matches, first
                            # find the name of the component
                            neighbor_components_name = neighbor.replace("_in", "")
                            neighbor_components_name = neighbor_components_name.replace("_out", "")

                            # Then we check among the connected tanks component, which inputs
                            # number it correspond to

                            tank_output_name = neighbor_components_name + ".fuel_consumed_t"
                            index = self._components_connection_inputs.index(tank_output_name)
                            corresponding_fs_input = self._components_connection_outputs[index]

                            node_number = corresponding_fs_input[-1]

                            # From the way we built the dict, we should have the power like this
                            neighbor_power = input_power_dict[corresponding_fs_input.split(".")[-1]]

                            node_name = associated_component_name + "_in_" + node_number
                            problematic_nodes_to_add.append(node_name)
                            nodes_with_power[node_name] = neighbor_power

                            copied_graph.add_edge(node_name, neighbor)

                    # Now we remove the fuel system and instead add its neighbors, which will
                    # serve as new branch start

                    new_problematic_nodes.remove(problematic_node)
                    new_problematic_nodes += problematic_nodes_to_add

                copied_graph.remove_node(problematic_node)

            problematic_nodes = new_problematic_nodes

        return nodes_with_power

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

    def set_priority_in_graph(
        self, graph: nx.Graph, nodes_with_power: dict, starting_points_name: list
    ) -> tuple:
        """
        Explore a graph and set the priority of its node base on the priority of the starting
        points.
        """

        name_to_eta = dict(zip(self._components_name, self._components_efficiency))

        problematic_nodes = []
        another_copied_graph = copy.deepcopy(graph)
        sub_graphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

        # First we must associate each starting point with its subgraphs (To prevent errors in
        # case one subgraph has multiple starting points or there are multiple subgraphs in,
        # each with one starting point)

        ordered_sub_graphs = []

        for starting_point in starting_points_name:
            for subgraph in sub_graphs:
                if subgraph.has_node(starting_point):
                    ordered_sub_graphs.append(subgraph)
                    continue

        for starting_point, subgraph in zip(starting_points_name, ordered_sub_graphs):

            current_node = starting_point

            # The inputs of the end components (usually sources, are not really necessary,
            # so we will start the number of explored nodes at 1. The other reason for that being
            # that it crashes if I start at 0
            node_explored = 1

            initial_number_of_nodes = subgraph.number_of_nodes()
            # We look at how many neighbor the current node has, and if it more than 2 we stop
            # exploring :) Also we'll put a failsafe to avoid infinite while

            while len(graph.adj[current_node]) <= 2 and node_explored < initial_number_of_nodes:

                for adj_node in graph.adj[current_node]:

                    # A previously explored neighbor
                    if adj_node in list(nodes_with_power.keys()):

                        # Here depending on whether we are going from a component's input to
                        # output or from a component to the other, setting the power will be
                        # different

                        current_components_name = current_node.replace("_in", "")
                        current_components_name = current_components_name.replace("_out", "")

                        adj_components_name = adj_node.replace("_in", "")
                        adj_components_name = adj_components_name.replace("_out", "")

                        # if name is the same, we compute the power below base on the efficiency,
                        # else its the same power, to refactor, we'll just say the efficiency is 1
                        if current_components_name == adj_components_name:
                            eta = name_to_eta[current_components_name]
                        else:
                            eta = 1.0

                        nodes_with_power[current_node] = nodes_with_power[adj_node] / eta

                    # Not a previously explored neighbor
                    else:
                        next_node = adj_node
                        node_explored += 1

                # When we are done with the node, we remove it :)
                another_copied_graph.remove_node(current_node)

                current_node = next_node

            if len(graph.adj[current_node]) > 2 and current_node not in problematic_nodes:
                problematic_nodes.append(current_node)

        graph = another_copied_graph

        return problematic_nodes, graph

    def get_power_to_set(
        self, inputs, propulsive_power_dict: dict
    ) -> Tuple[List[dict], List[dict]]:
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

        power_in_each_subgraph = []
        final_list = []
        power_at_each_node = {}

        # First step is to identify the independent sub-propulsion chain
        sub_graphs = self.get_independent_sub_propulsion_chain()

        # Need to be put here else the _get_component hasn't triggered yet
        name_to_id = dict(zip(self._components_name, self._components_id))

        # Then for each subgraph we get the power on each node
        for sub_graph in sub_graphs:

            power_dict_subgraph = {}

            # First we reconstruct the right propulsive load dict to ensure that we only take the
            # load we are interested in
            propulsive_power_dict_this_subgraph = {}
            for propulsive_load_name in list(proper_propulsive_power_dict.keys()):
                if propulsive_load_name in sub_graph.nodes:
                    propulsive_power_dict_this_subgraph[
                        propulsive_load_name
                    ] = proper_propulsive_power_dict[propulsive_load_name]

            power_in_this_subgraph = self.get_power_on_each_node(
                sub_graph, inputs, propulsive_power_dict_this_subgraph
            )
            power_in_each_subgraph.append(power_in_this_subgraph)

            nodes_list = list(sub_graph.nodes)
            for node in nodes_list:
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
                            power_dict_subgraph[variable_name] = power_in_this_subgraph[
                                node + "_" + variable_name[-1]
                            ]
                        else:
                            power_dict_subgraph[variable_name] = power_in_this_subgraph[node]

                    elif power[1] == "out" and node.endswith("_out"):
                        variable_name = component_name + "." + power[0]
                        power_dict_subgraph[variable_name] = power_in_this_subgraph[node]

            final_list.append(power_dict_subgraph)
            power_at_each_node = dict(power_at_each_node, **power_in_this_subgraph)

        self._power_at_each_node = power_at_each_node

        return power_in_each_subgraph, final_list

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
        for component_name, component_type_class in zip(
            self._components_name, self._components_type_class
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
                voltage_node = all_voltage_dict[node]
                current = all_power_dict[node] / voltage_node
                variable_name = component_name + ".dc_current_out"

                all_current_dict[variable_name] = current

        return all_current_dict


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
