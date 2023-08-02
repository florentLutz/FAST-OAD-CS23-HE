# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np

import openmdao.api as om
import fastoad.api as oad

from ..constants import HE_SUBMODEL_EQUILIBRIUM
from ..mission.performance_per_phase import PerformancePerPhase
from ..mission.sizing_energy import SizingEnergy
from ..mission.thrust_taxi import ThrustTaxi
from ..mission.update_mass import UpdateMass


class MissionCore(om.Group):
    """Find the conditions necessary for the aircraft equilibrium."""

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

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        self.add_subsystem(
            "compute_taxi_thrust",
            ThrustTaxi(),
            promotes=["*"],
        )
        options_equilibrium = {
            "number_of_points": number_of_points,
            "propulsion_id": self.options["propulsion_id"],
            "power_train_file_path": self.options["power_train_file_path"],
            "use_linesearch": self.options["use_linesearch"],
            "pre_condition_voltage": self.options["pre_condition_voltage"],
        }
        self.add_subsystem(
            "compute_dep_equilibrium",
            oad.RegisterSubmodel.get_submodel(HE_SUBMODEL_EQUILIBRIUM, options=options_equilibrium),
            promotes=[
                "data:*",
                "mass",
                "time_step",
                "thrust_rate_t_econ",
                "non_consumable_energy_t_econ",
                "fuel_consumed_t_econ",
            ],
        )
        self.add_subsystem(
            "performance_per_phase",
            PerformancePerPhase(
                number_of_points_climb=number_of_points_climb,
                number_of_points_cruise=number_of_points_cruise,
                number_of_points_descent=number_of_points_descent,
                number_of_points_reserve=number_of_points_reserve,
            ),
            promotes_inputs=[
                "thrust_rate_t_econ",
                "non_consumable_energy_t_econ",
                "fuel_consumed_t_econ",
            ],
            promotes_outputs=["data:*", "fuel_consumed_t"],
        )
        self.add_subsystem("sizing_fuel", SizingEnergy(), promotes=["*"])
        self.add_subsystem(
            "update_mass",
            UpdateMass(number_of_points=number_of_points),
            promotes=["*"],
        )

    def guess_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]
        number_of_points_total = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        mtow = inputs["update_mass.data:weight:aircraft:MTOW"]

        dummy_fuel_consumed = np.linspace(0, 10.0, number_of_points_total)
        outputs["update_mass.mass"] = np.full(number_of_points_total, mtow) - np.cumsum(
            dummy_fuel_consumed
        )
        outputs["compute_dep_equilibrium.compute_equilibrium_alpha.alpha"] = np.concatenate(
            (
                np.full(number_of_points_climb, 3.0),
                np.full(number_of_points_cruise, 2.0),
                np.full(number_of_points_descent, 1.0),
                np.full(number_of_points_reserve, 7.0),
            )
        )
        outputs["compute_dep_equilibrium.compute_equilibrium_delta_m.delta_m"] = np.concatenate(
            (
                np.full(number_of_points_climb, -10.0),
                np.full(number_of_points_cruise, -2.0),
                np.full(number_of_points_descent, -5.0),
                np.full(number_of_points_reserve, -2.0),
            )
        )
        outputs["compute_dep_equilibrium.compute_equilibrium_thrust.thrust"] = np.concatenate(
            (
                np.full(number_of_points_climb, 2.0 * mtow),
                np.full(number_of_points_cruise, mtow / 1.3),
                np.full(number_of_points_descent, 0.5 * mtow),
                np.full(number_of_points_reserve, mtow / 1.3),
            )
        )
