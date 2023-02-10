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
from typing import Tuple

from jsonschema import validate
from ruamel.yaml import YAML

from .exceptions import (
    FASTGAHEUnknownComponentID,
    FASTGAHEUnknownOption,
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

            components_name_list.append(component_name)
            components_name_id_list.append(resources.DICTIONARY_CN_ID[component_id])
            components_type_list.append(resources.DICTIONARY_CT[component_id])
            components_om_type_list.append(resources.DICTIONARY_CN[component_id])
            components_promote_list.append(resources.DICTIONARY_PT[component_id])
            components_type_class_list.append(resources.DICTIONARY_CTC[component_id])
            components_perf_watchers_list.append(resources.DICTIONARY_MP[component_id])

            if "options" in component.keys():
                components_options_list.append(component["options"])

                # While we are at it, we also check that we have the right options and with the
                # right names

                if set(component["options"].keys()) != set(resources.DICTIONARY_ATT[component_id]):
                    raise FASTGAHEUnknownOption(
                        "Component " + component_id + " does not have all options declare or they "
                        "have an erroneous name. The following options should be declared: "
                        + ", ".join(resources.DICTIONARY_ATT[component_id])
                    )

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
                source_number = ""
                source_inputs = resources.DICTIONARY_IN[translator[source_name]]
            else:
                source_name = connection["source"][0]
                source_number = str(connection["source"][1])
                source_inputs = resources.DICTIONARY_IN[translator[source_name]]

            if type(connection["target"]) is str:
                target_name = connection["target"]
                target_number = ""
                target_outputs = resources.DICTIONARY_OUT[translator[target_name]]
            else:
                target_name = connection["target"][0]
                target_number = str(connection["target"][1])
                target_outputs = resources.DICTIONARY_OUT[translator[target_name]]

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

        return (
            self._components_name,
            self._components_name_id,
            self._components_type,
            self._components_om_type,
            self._components_options,
            self._components_connection_outputs,
            self._components_connection_inputs,
            self._components_promotes,
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
            if component_type_class == "propulsor":
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
            if component_type_class == "propulsive_load":
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
            if component_type_class == "source":
                components_names.append(component_name)

        return components_names

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

    def get_network_elements_list(self) -> tuple:
        """
        Returns the name of the components and their connections for the visualisation of the
        power train as a network graph.
        """

        self._get_components()
        self._get_connections()

        curated_connection_list = []
        for connections in self._connection_list:
            curated_connection_list.append((connections["source"], connections["target"]))

        return self._components_name, curated_connection_list


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
