# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import (
    FASTGAHEPowerTrainConfigurator,
    PROMOTION_FROM_MISSION,
)
from fastga_he.models.propulsion.assemblers.energy_consumption_from_pt_file import (
    EnergyConsumptionFromPTFile,
)
from fastga_he.models.propulsion.assemblers.performances_watcher import (
    PowerTrainPerformancesWatcher,
)

# noinspection PyUnresolvedReferences
from fastga_he.models.propulsion.components import (
    PerformancesPropeller,
    PerformancesPMSM,
    PerformancesInverter,
    PerformancesDCBus,
    PerformancesHarness,
    PerformancesDCDCConverter,
    PerformancesBatteryPack,
    PerformancesDCSSPC,
    PerformancesDCSplitter,
)

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
            desc="Boolean to add solvers to the power train performance group. Default is true "
            "but can be turned off when used jointly with the mission to save computation time",
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
            desc="Contains the list of the SSPC name which state need to be changed. If this list is empty, nothing will be done.",
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

            klass = globals()["Performances" + component_om_type]
            local_sub_sys = klass()
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

        if self.configurator.get_watcher_file_path():

            self.add_subsystem(
                "performances_watcher",
                PowerTrainPerformancesWatcher(
                    power_train_file_path=self.options["power_train_file_path"],
                    number_of_points=number_of_points,
                ),
                promotes=list(PROMOTION_FROM_MISSION.keys()),
            )

            (
                components_name,
                components_performances_watchers_names,
                components_performances_watchers_units,
            ) = self.configurator.get_performance_watcher_elements_list()

            for (component_name, component_performances_watcher_name,) in zip(
                components_name,
                components_performances_watchers_names,
            ):

                self.connect(
                    component_name + "." + component_performances_watcher_name,
                    "performances_watcher"
                    + "."
                    + component_name
                    + "_"
                    + component_performances_watcher_name,
                )
