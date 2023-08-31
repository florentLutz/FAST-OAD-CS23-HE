# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import logging

import openmdao.api as om

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from .initialization.initialize import Initialize
from .mission.mission_core import MissionCore
from .to_csv import ToCSV
from fastga_he.models.weight.cg.cg_variation import InFlightCGVariation

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator
from fastga_he.models.propulsion.assemblers.performances_watcher import (
    PowerTrainPerformancesWatcher,
)

from fastga_he.models.performances.mission_vector.constants import (
    HE_SUBMODEL_ENERGY_CONSUMPTION,
    HE_SUBMODEL_DEP_EFFECT,
)
from fastga_he.models.propulsion.assemblers.energy_consumption_mission_vector import (
    ENERGY_CONSUMPTION_FROM_PT_FILE,
)
from fastga_he.models.propulsion.assemblers.delta_from_pt_file import DEP_EFFECT_FROM_PT_FILE

_LOGGER = logging.getLogger(__name__)


@oad.RegisterOpenMDAOSystem("fastga_he.performances.mission_vector", domain=ModelDomain.OTHER)
class MissionVector(om.Group):
    """Computes and potentially save mission based on options."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 100
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.LinearBlockGS()

        self.configurator = FASTGAHEPowerTrainConfigurator()

    def initialize(self):

        self.options.declare("out_file", default="", types=str)
        self.options.declare(
            "number_of_points_climb",
            default=100,
            desc="number of equilibrium to be treated in climb",
        )
        self.options.declare(
            "number_of_points_cruise",
            default=100,
            desc="number of equilibrium to be treated in cruise",
        )
        self.options.declare(
            "number_of_points_descent",
            default=50,
            desc="number of equilibrium to be treated in descent",
        )
        self.options.declare(
            "number_of_points_reserve",
            default=1,
            desc="number of equilibrium to be treated in reserve",
        )
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare(
            name="power_train_file_path",
            default="",
            desc="Path to the file containing the description of the power",
        )
        self.options.declare(
            "use_linesearch",
            default=True,
            types=bool,
            desc="boolean to turn off the use of a linesearch algorithm during the mission."
            "Can be turned off to speed up the process but might not converge.",
        )
        self.options.declare(
            name="pre_condition_voltage",
            default=False,
            desc="Boolean to pre_condition the voltages of the different components of the PT, "
            "can save some time in specific cases",
            allow_none=False,
        )

    def setup(self):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        pt_file_path = self.options["power_train_file_path"]

        self.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
        self.add_subsystem(
            "initialization",
            Initialize(
                number_of_points_climb=number_of_points_climb,
                number_of_points_cruise=number_of_points_cruise,
                number_of_points_descent=number_of_points_descent,
                number_of_points_reserve=number_of_points_reserve,
            ),
            promotes_inputs=["data:*", "settings:*"],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "solve_equilibrium",
            MissionCore(
                number_of_points_climb=number_of_points_climb,
                number_of_points_cruise=number_of_points_cruise,
                number_of_points_descent=number_of_points_descent,
                number_of_points_reserve=number_of_points_reserve,
                propulsion_id=self.options["propulsion_id"],
                power_train_file_path=self.options["power_train_file_path"],
                use_linesearch=self.options["use_linesearch"],
                pre_condition_voltage=self.options["pre_condition_voltage"],
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "to_csv",
            ToCSV(
                number_of_points_climb=number_of_points_climb,
                number_of_points_cruise=number_of_points_cruise,
                number_of_points_descent=number_of_points_descent,
                number_of_points_reserve=number_of_points_reserve,
                out_file=self.options["out_file"],
            ),
            promotes_inputs=["data:*"],
            promotes_outputs=[],
        )

        self.connect(
            "initialization.initialize_engine_setting.engine_setting",
            [
                "solve_equilibrium.compute_dep_equilibrium.engine_setting",
                "to_csv.engine_setting",
            ],
        )

        self.connect(
            "initialization.initialize_temperature.exterior_temperature",
            [
                "solve_equilibrium.compute_dep_equilibrium.exterior_temperature",
                "to_csv.exterior_temperature",
            ],
        )

        self.connect(
            "initialization.initialize_center_of_gravity.x_cg",
            [
                "solve_equilibrium.compute_dep_equilibrium.x_cg",
                "to_csv.x_cg",
            ],
        )
        self.connect(
            "initialization.initialize_density.density",
            [
                "solve_equilibrium.compute_dep_equilibrium.density",
            ],
        )

        self.connect(
            "initialization.initialize_time_and_distance.position",
            ["solve_equilibrium.performance_per_phase.position", "to_csv.position"],
        )

        self.connect(
            "initialization.initialize_time_and_distance.time",
            [
                "solve_equilibrium.performance_per_phase.time",
                "to_csv.time",
            ],
        )

        self.connect(
            "initialization.initialize_time_step.time_step",
            ["to_csv.time_step", "solve_equilibrium.time_step"],
        )

        self.connect(
            "initialization.initialize_airspeed_time_derivatives.d_vx_dt",
            [
                "solve_equilibrium.compute_dep_equilibrium.d_vx_dt",
                "to_csv.d_vx_dt",
            ],
        )

        self.connect(
            "initialization.initialize_airspeed.true_airspeed",
            [
                "solve_equilibrium.compute_dep_equilibrium.true_airspeed",
                "to_csv.true_airspeed",
            ],
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.delta_Cl",
            "to_csv.delta_Cl",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.delta_Cd",
            "to_csv.delta_Cd",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.delta_Cm",
            "to_csv.delta_Cm",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.alpha",
            "to_csv.alpha",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.thrust",
            "to_csv.thrust",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.delta_m",
            "to_csv.delta_m",
        )

        self.connect(
            "initialization.initialize_airspeed.equivalent_airspeed", "to_csv.equivalent_airspeed"
        )

        self.connect(
            "solve_equilibrium.mass",
            [
                "to_csv.mass",
                "initialization.mass",
            ],
        )

        self.connect(
            "solve_equilibrium.fuel_consumed_t",
            [
                "to_csv.fuel_consumed_t",
                "initialization.initialize_center_of_gravity.fuel_consumed_t",
            ],
        )

        self.connect(
            "solve_equilibrium.performance_per_phase.non_consumable_energy_t",
            [
                "to_csv.non_consumable_energy_t",
            ],
        )

        self.connect(
            "solve_equilibrium.performance_per_phase.thrust_rate_t", "to_csv.thrust_rate_t"
        )

        self.connect(
            "initialization.initialize_gamma.gamma",
            [
                "to_csv.gamma",
                "solve_equilibrium.compute_dep_equilibrium.gamma",
            ],
        )

        self.connect(
            "initialization.altitude",
            [
                "to_csv.altitude",
                "solve_equilibrium.compute_dep_equilibrium.altitude",
            ],
        )

        # Add the powertrain watcher here to avoid opening and closing csv all the time. We will
        # add here a check to ensure that the module that computes the performances base on the
        # powertrain builder is used. Because if it is not used, no pt file will be provided
        # meaning the load instruction of the configurator will fail and so will the connects.

        if (
            oad.RegisterSubmodel.active_models[HE_SUBMODEL_ENERGY_CONSUMPTION]
            == ENERGY_CONSUMPTION_FROM_PT_FILE
        ):

            self.configurator.load(pt_file_path)

            if self.configurator.get_watcher_file_path():

                number_of_points = (
                    1
                    + number_of_points_climb
                    + number_of_points_cruise
                    + number_of_points_descent
                    + number_of_points_reserve
                    + 1
                )
                self.add_subsystem(
                    "performances_watcher",
                    PowerTrainPerformancesWatcher(
                        power_train_file_path=self.options["power_train_file_path"],
                        number_of_points=number_of_points,
                    ),
                )

                (
                    components_name,
                    components_performances_watchers_names,
                    _,
                ) = self.configurator.get_performance_watcher_elements_list()

                for (component_name, component_performances_watcher_name) in zip(
                    components_name, components_performances_watchers_names
                ):

                    self.connect(
                        "solve_equilibrium.compute_dep_equilibrium.compute_energy_consumed.power_train_performances."
                        + component_name
                        + "."
                        + component_performances_watcher_name,
                        "performances_watcher"
                        + "."
                        + component_name
                        + "_"
                        + component_performances_watcher_name,
                    )

                # This is starting to become confusing: The next bit should only be executed if
                # the right model for the computation of slipstream effect is used. BUT we want
                # to be able to "turn off" the slipstream effects and still register results in
                # the powertrain watcher ...

                if (
                    oad.RegisterSubmodel.active_models[HE_SUBMODEL_DEP_EFFECT]
                    == DEP_EFFECT_FROM_PT_FILE
                ):

                    (
                        components_slip_name,
                        components_slip_performances_watchers_names,
                        _,
                    ) = self.configurator.get_slipstream_performance_watcher_elements_list()

                    for (component_slip_name, component_slip_performances_watcher_name) in zip(
                        components_slip_name, components_slip_performances_watchers_names
                    ):
                        self.connect(
                            "solve_equilibrium.compute_dep_equilibrium.compute_dep_effect."
                            + component_slip_name
                            + "."
                            + component_slip_performances_watcher_name,
                            "performances_watcher"
                            + "."
                            + component_slip_name
                            + "_"
                            + component_slip_performances_watcher_name,
                        )

                self.connect(
                    "solve_equilibrium.compute_dep_equilibrium.thrust_econ",
                    "performances_watcher.thrust",
                )
                self.connect(
                    "solve_equilibrium.compute_dep_equilibrium.altitude_econ",
                    "performances_watcher.altitude",
                )
                self.connect(
                    "solve_equilibrium.compute_dep_equilibrium.time_step_econ",
                    "performances_watcher.time_step",
                )
                self.connect(
                    "solve_equilibrium.compute_dep_equilibrium.true_airspeed_econ",
                    "performances_watcher.true_airspeed",
                )
                self.connect(
                    "solve_equilibrium.compute_dep_equilibrium.exterior_temperature_econ",
                    "performances_watcher.exterior_temperature",
                )

        else:

            _LOGGER.warning(
                "Power train builder is not used for the performances computation. If "
                "this was intended, you can ignore this warning. Else, make sure to select the "
                + ENERGY_CONSUMPTION_FROM_PT_FILE
                + " submodel for the "
                + HE_SUBMODEL_ENERGY_CONSUMPTION
                + " service"
            )
