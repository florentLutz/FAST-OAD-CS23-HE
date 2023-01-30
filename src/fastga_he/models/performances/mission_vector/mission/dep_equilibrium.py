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
from ..mission.equilibrium import Equilibrium


@oad.RegisterSubmodel(HE_SUBMODEL_EQUILIBRIUM, "fastga_he.submodel.performances.equilibrium.legacy")
class DEPEquilibrium(om.Group):
    """Find the conditions necessary for the aircraft equilibrium."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup and configuration
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50
        self.nonlinear_solver.options["rtol"] = 1e-4
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
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
            "number_of_points_climb", default=1, desc="number of equilibrium to be treated in climb"
        )
        self.options.declare(
            "number_of_points_cruise",
            default=1,
            desc="number of equilibrium to be treated in cruise",
        )
        self.options.declare(
            "number_of_points_descent",
            default=1,
            desc="number of equilibrium to be treated in descent",
        )
        self.options.declare(
            "number_of_points_reserve",
            default=1,
            desc="number of equilibrium to be treated in reserve",
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

    def setup(self):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        if self.options["promotes_all_variables"]:
            self.add_subsystem(
                "preparation_for_energy_consumption",
                PrepareForEnergyConsumption(
                    number_of_points_climb=number_of_points_climb,
                    number_of_points_cruise=number_of_points_cruise,
                    number_of_points_descent=number_of_points_descent,
                    number_of_points_reserve=number_of_points_reserve,
                ),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )
            self.add_subsystem(
                "compute_equilibrium",
                Equilibrium(
                    number_of_points=number_of_points, flaps_position=self.options["flaps_position"]
                ),
                promotes_inputs=["*"],
                promotes_outputs=["*"],
            )
            options_dep = {
                "number_of_points": number_of_points,
                "flaps_position": self.options["flaps_position"],
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
                PrepareForEnergyConsumption(
                    number_of_points_climb=number_of_points_climb,
                    number_of_points_cruise=number_of_points_cruise,
                    number_of_points_descent=number_of_points_descent,
                    number_of_points_reserve=number_of_points_reserve,
                ),
                promotes_inputs=["data:*"],
                promotes_outputs=[],
            )
            self.add_subsystem(
                "compute_equilibrium",
                Equilibrium(
                    number_of_points=number_of_points, flaps_position=self.options["flaps_position"]
                ),
                promotes_inputs=["data:*"],
                promotes_outputs=[],
            )
            options_dep = {
                "number_of_points": number_of_points,
                "flaps_position": self.options["flaps_position"],
            }
            self.add_subsystem(
                "compute_dep_effect",
                oad.RegisterSubmodel.get_submodel(HE_SUBMODEL_DEP_EFFECT, options=options_dep),
                promotes_inputs=["data:*"],
                promotes_outputs=[],
            )
            options_propulsion = {
                "number_of_points": number_of_points,
                "propulsion_id": self.options["propulsion_id"],
                "power_train_file_path": self.options["power_train_file_path"],
            }
            self.add_subsystem(
                "compute_energy_consumed",
                oad.RegisterSubmodel.get_submodel(
                    HE_SUBMODEL_ENERGY_CONSUMPTION, options=options_propulsion
                ),
                promotes=["data:*"],
            )

            self.connect("compute_dep_effect.delta_Cl", "compute_equilibrium.delta_Cl")

            self.connect("compute_dep_effect.delta_Cd", "compute_equilibrium.delta_Cd")

            self.connect("compute_dep_effect.delta_Cm", "compute_equilibrium.delta_Cm")

            self.connect("compute_equilibrium.alpha", "compute_dep_effect.alpha")

            self.connect(
                "compute_equilibrium.thrust",
                "compute_dep_effect.thrust",
            )

            self.connect(
                "preparation_for_energy_consumption.engine_setting_econ",
                "compute_energy_consumed.engine_setting_econ",
            )
            self.connect(
                "preparation_for_energy_consumption.thrust_econ",
                "compute_energy_consumed.thrust_econ",
            )
            self.connect(
                "preparation_for_energy_consumption.altitude_econ",
                "compute_energy_consumed.altitude_econ",
            )
            self.connect(
                "preparation_for_energy_consumption.exterior_temperature_econ",
                "compute_energy_consumed.exterior_temperature_econ",
            )
            self.connect(
                "preparation_for_energy_consumption.time_step_econ",
                "compute_energy_consumed.time_step_econ",
            )
            self.connect(
                "preparation_for_energy_consumption.true_airspeed_econ",
                "compute_energy_consumed.true_airspeed_econ",
            )

            self.connect(
                "compute_equilibrium.thrust",
                "preparation_for_energy_consumption.thrust",
            )
