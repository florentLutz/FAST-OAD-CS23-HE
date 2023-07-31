# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import openmdao.api as om
import fastoad.api as oad

from ..constants import (
    HE_SUBMODEL_DEP_EFFECT,
    HE_SUBMODEL_EQUILIBRIUM,
    HE_SUBMODEL_ENERGY_CONSUMPTION,
)
from ..mission.energy_consumption_preparation import PrepareForEnergyConsumption
from .equilibrium_alpha import EquilibriumAlpha
from .equilibrium_thrust import EquilibriumThrust
from .equilibrium_delta_m import EquilibriumDeltaM


@oad.RegisterSubmodel(HE_SUBMODEL_EQUILIBRIUM, "fastga_he.submodel.performances.equilibrium.legacy")
class DEPEquilibrium(om.Group):
    """Find the conditions necessary for the aircraft equilibrium."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup and configuration
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 30
        self.nonlinear_solver.options["rtol"] = 1e-6
        self.nonlinear_solver.options["atol"] = 1e-6
        self.nonlinear_solver.options["stall_limit"] = 5
        self.nonlinear_solver.options["stall_tol"] = 1e-6
        self.linear_solver = om.DirectSolver()

    def initialize(self):

        # We have to declare them even if not used to preserve compatibility
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare(
            name="power_train_file_path",
            default="",
            desc="Path to the file containing the description of the power",
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "promotes_all_variables",
            default=False,
            desc="Set to True if we need to be able to see the flight conditions variables, "
            "not needed in the mission",
        )
        self.options.declare(
            "flaps_position",
            default="cruise",
            desc="position of the flaps for the computation of the equilibrium",
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

        number_of_points = self.options["number_of_points"]

        if self.options["use_linesearch"]:
            self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

        if self.options["promotes_all_variables"]:
            self.add_subsystem(
                "preparation_for_energy_consumption",
                PrepareForEnergyConsumption(number_of_points=number_of_points),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )
            self.add_subsystem(
                "compute_equilibrium_alpha",
                EquilibriumAlpha(
                    number_of_points=number_of_points, flaps_position=self.options["flaps_position"]
                ),
                promotes_inputs=["*"],
                promotes_outputs=[],
            )
            self.add_subsystem(
                "compute_equilibrium_thrust",
                EquilibriumThrust(
                    number_of_points=number_of_points, flaps_position=self.options["flaps_position"]
                ),
                promotes_inputs=["*"],
                promotes_outputs=[],
            )
            self.add_subsystem(
                "compute_equilibrium_delta_m",
                EquilibriumDeltaM(
                    number_of_points=number_of_points, flaps_position=self.options["flaps_position"]
                ),
                promotes_inputs=["*"],
                promotes_outputs=[],
            )
            options_dep = {
                "number_of_points": number_of_points,
                "flaps_position": self.options["flaps_position"],
                "power_train_file_path": self.options["power_train_file_path"],
            }
            self.add_subsystem(
                "compute_dep_effect",
                oad.RegisterSubmodel.get_submodel(HE_SUBMODEL_DEP_EFFECT, options=options_dep),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )
            options_propulsion = {
                "number_of_points": number_of_points,
                "propulsion_id": self.options["propulsion_id"],
                "power_train_file_path": self.options["power_train_file_path"],
                "pre_condition_voltage": self.options["pre_condition_voltage"],
            }
            self.add_subsystem(
                "compute_energy_consumed",
                oad.RegisterSubmodel.get_submodel(
                    HE_SUBMODEL_ENERGY_CONSUMPTION, options=options_propulsion
                ),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )
        else:
            self.add_subsystem(
                "preparation_for_energy_consumption",
                PrepareForEnergyConsumption(number_of_points=number_of_points),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_equilibrium_alpha",
                EquilibriumAlpha(
                    number_of_points=number_of_points, flaps_position=self.options["flaps_position"]
                ),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_equilibrium_thrust",
                EquilibriumThrust(
                    number_of_points=number_of_points, flaps_position=self.options["flaps_position"]
                ),
                promotes=["*"],
            )
            self.add_subsystem(
                "compute_equilibrium_delta_m",
                EquilibriumDeltaM(
                    number_of_points=number_of_points, flaps_position=self.options["flaps_position"]
                ),
                promotes=["*"],
            )
            options_dep = {
                "number_of_points": number_of_points,
                "flaps_position": self.options["flaps_position"],
                "power_train_file_path": self.options["power_train_file_path"],
            }
            self.add_subsystem(
                "compute_dep_effect",
                oad.RegisterSubmodel.get_submodel(HE_SUBMODEL_DEP_EFFECT, options=options_dep),
                promotes_inputs=["data:*", "alpha", "thrust", "density", "true_airspeed"],
                promotes_outputs=["delta_Cl", "delta_Cd", "delta_Cm"],
            )
            options_propulsion = {
                "number_of_points": number_of_points,
                "propulsion_id": self.options["propulsion_id"],
                "power_train_file_path": self.options["power_train_file_path"],
                "pre_condition_voltage": self.options["pre_condition_voltage"],
            }
            self.add_subsystem(
                "compute_energy_consumed",
                oad.RegisterSubmodel.get_submodel(
                    HE_SUBMODEL_ENERGY_CONSUMPTION, options=options_propulsion
                ),
                promotes=[
                    "data:*",
                    "fuel_consumed_t_econ",
                    "non_consumable_energy_t_econ",
                    "thrust_rate_t_econ",
                    "true_airspeed_econ",
                    "true_airspeed_econ",
                    "time_step_econ",
                    "exterior_temperature_econ",
                    "altitude_econ",
                    "density_econ",
                    "thrust_econ",
                    "engine_setting_econ",
                ],
            )
