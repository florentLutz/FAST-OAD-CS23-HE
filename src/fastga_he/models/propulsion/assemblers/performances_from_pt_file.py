# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import (
    FASTGAHEPowerTrainConfigurator,
)
from fastga_he.models.propulsion.assemblers.energy_consumption_from_pt_file import (
    EnergyConsumptionFromPTFile,
)

import fastga_he.models.propulsion.components as he_comp

from .constants import SUBMODEL_POWER_TRAIN_PERF, SUBMODEL_THRUST_DISTRIBUTOR


@oad.RegisterSubmodel(
    SUBMODEL_POWER_TRAIN_PERF, "fastga_he.submodel.propulsion.performances.from_pt_file"
)
class PowerTrainPerformancesFromFile(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

    def initialize(self):
        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the power",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="add_solver",
            default=True,
            desc="Boolean to add solvers to the power train performance group. Default is False "
            "it can be turned off when used jointly with the mission to save computation time",
            allow_none=False,
        )
        self.options.declare(
            name="pre_condition_pt",
            default=False,
            desc="Boolean to pre_condition the different components of the PT, "
            "can save some time in specific cases",
            allow_none=False,
        )
        self.options.declare(
            name="sort_component",
            default=True,
            desc="Boolean to sort the component with proper order for adding subsystem operations",
            allow_none=False,
        )
        self.options.declare(
            name="sspc_closed_list",
            default=[],
            types=list,
            desc="List of the states of the SSPC specified in the sspc_names_list option, "
            "each element must be either True or False, when nothing is specified, "
            "the default state from the PT file is used",
            allow_none=True,
        )
        self.options.declare(
            name="sspc_names_list",
            default=[],
            types=list,
            desc="Contains the list of the SSPC name which state need to be changed. If this list "
            "is empty, nothing will be done.",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.configurator.load(self.options["power_train_file_path"])

        propulsor_names = self.configurator.get_thrust_element_list()
        source_names = self.configurator.get_energy_consumption_list()

        (
            components_name,
            components_name_id,
            _,
            components_om_type,
            components_options,
            components_connection_outputs,
            components_connection_inputs,
            components_promotes,
            sspc_list,
            sspc_state,
        ) = self.configurator.get_performances_element_lists()

        # We decide on the SSPCs state, we take the default state unless the options specify
        # otherwise
        if self.options["sspc_names_list"]:
            for sspc_name, sspc_closed in zip(
                self.options["sspc_names_list"], self.options["sspc_closed_list"]
            ):
                sspc_state[sspc_name] = sspc_closed

        # We check the value the resulting states to see if it agrees with the logic and change
        # it if it is not the case
        sspc_state = self.configurator.check_sspc_states(sspc_state)

        options = {
            "power_train_file_path": self.options["power_train_file_path"],
            "number_of_points": number_of_points,
        }
        self.add_subsystem(
            name="thrust_splitter",
            subsys=oad.RegisterSubmodel.get_submodel(SUBMODEL_THRUST_DISTRIBUTOR, options=options),
            promotes=["data:*", "thrust"],
        )

        if self.options["sort_component"]:
            prop_order_dict, sort_map = self.create_propeller_order_map(components_name)
            (
                components_name,
                components_name_id,
                components_om_type,
                components_options,
                components_promotes,
            ) = self.reorder_components(
                sort_map,
                components_name,
                components_name_id,
                components_om_type,
                components_options,
                components_promotes,
            )

        # Enforces SSPC are added last, not done before because it might breaks the connections
        # necessary to ensure the coherence of SSPC states when connected to both end of a cable
        (
            components_name,
            components_name_id,
            components_om_type,
            components_options,
            components_promotes,
        ) = self.configurator.enforce_sspc_last(
            components_name,
            components_name_id,
            components_om_type,
            components_options,
            components_promotes,
        )

        for (
            component_name,
            component_name_id,
            component_om_type,
            component_option,
            component_promote,
        ) in zip(
            components_name,
            components_name_id,
            components_om_type,
            components_options,
            components_promotes,
        ):
            local_sub_sys = he_comp.__dict__["Performances" + component_om_type]()
            local_sub_sys.options[component_name_id] = component_name
            local_sub_sys.options["number_of_points"] = number_of_points

            if component_name in sspc_list.keys():
                local_sub_sys.options["at_bus_output"] = sspc_list[component_name]
                local_sub_sys.options["closed"] = sspc_state[component_name]

            if component_option:
                for option_name in component_option:
                    local_sub_sys.options[option_name] = component_option[option_name]

            self.add_subsystem(
                name=component_name,
                subsys=local_sub_sys,
                promotes=["data:*"] + component_promote,
            )

        self.add_subsystem(
            name="energy_consumption",
            subsys=EnergyConsumptionFromPTFile(
                number_of_points=number_of_points,
                power_train_file_path=self.options["power_train_file_path"],
            ),
            promotes=["non_consumable_energy_t_econ", "fuel_consumed_t_econ"],
        )

        for propulsor_name in propulsor_names:
            self.connect(
                "thrust_splitter." + propulsor_name + "_thrust", propulsor_name + ".thrust"
            )

        for om_output, om_input in zip(components_connection_outputs, components_connection_inputs):
            self.connect(om_output, om_input)

        for source_name in source_names:
            self.connect(
                source_name + ".non_consumable_energy_t",
                "energy_consumption." + source_name + "_non_consumable_energy_t",
            )
            self.connect(
                source_name + ".fuel_consumed_t",
                "energy_consumption." + source_name + "_fuel_consumed_t",
            )

        if self.options["add_solver"]:
            # Solvers setup
            self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
            self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
            self.nonlinear_solver.options["iprint"] = 2
            self.nonlinear_solver.options["maxiter"] = 200
            self.nonlinear_solver.options["rtol"] = 1e-4
            self.linear_solver = om.DirectSolver()

        # The performances watcher was moved at the same level as the mission performances
        # watcher so that it is not opened as much, they could be merged eventually

    def guess_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        # We need to reinstate this check on the coherence of voltage because if we run it on its
        # own we prefer having a warning as well. Though it begs the question of pre
        # conditioning, voltage power and current here as well even if it is faster at mission
        # level
        # TODO: Think about that

        # This one will be passed in before going into the first pt components
        number_of_points = self.options["number_of_points"]

        # Let's first check the coherence of the voltage
        self.configurator.check_voltage_coherence(inputs=inputs, number_of_points=number_of_points)

    def create_propeller_order_map(self, key_list=None):
        """
        Reorders dictionary elements by their values and assigns proper sequential indices.
        Optionally maps a list of keys to their corresponding proper indices.

        Args:
            key_list (list, optional): List of keys to be replaced with indices.
                                     If None, only returns the reindexed dictionary.

        Returns:
            tuple or dict:
                - If key_list is provided: (reindexed_dict, indexed_list)
                - If key_list is None: reindexed_dict only
        """
        # Sort items by value first, then by original key order to maintain consistency
        distance_from_prop = self.configurator.get_distance_from_propulsor()
        sorted_items = sorted(distance_from_prop.items(), key=lambda x: (x[1], x[0]))

        # Create new dictionary with proper sequential indices
        reindexed_dict = {}
        for index, (key, original_value) in enumerate(sorted_items):
            reindexed_dict[key] = index

        # If no key_list provided, return only the reindexed dictionary
        if key_list is None:
            return reindexed_dict

        # Replace keys in list with indices
        indexed_list = []
        for key in key_list:
            if key in reindexed_dict:
                indexed_list.append(reindexed_dict[key])
            else:
                print(f"Warning: Key '{key}' not found in re-indexed dictionary")
                indexed_list.append(None)

        return reindexed_dict, indexed_list

    def reorder_components(self, index_map, *lists):
        result = []
        for lst in lists:
            reordered = [None] * len(lst)
            for i, new_pos in enumerate(index_map):
                reordered[new_pos] = lst[i]
            result.append(reordered)
        return tuple(result)
