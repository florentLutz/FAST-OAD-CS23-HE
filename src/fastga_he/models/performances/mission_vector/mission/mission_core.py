# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO.


import openmdao.api as om
import fastoad.api as oad

from ..constants import HE_SUBMODEL_EQUILIBRIUM
from ..mission.performance_per_phase import PerformancePerPhase
from ..mission.sizing_energy import SizingEnergy
from ..mission.sizing_time import SizingDuration
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
            name="pre_condition_pt",
            default=False,
            desc="Boolean to pre_condition the different components of the PT, "
            "can save some time in specific cases",
            allow_none=False,
        )
        self.options.declare(
            name="sort_component",
            default=False,
            desc="Boolean to enable automatic sorting to improve robustness of the convergence by "
            "ensuring components are executed in the right order",
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

        # This enables the components in the tanks and batteries that compute the energy/fuel
        # EXCLUDING reserve, which is what we need for the LCA.
        self.model_options["*.compute_dep_equilibrium.*"] = {
            "number_of_points_reserve": number_of_points_reserve
        }

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
            "pre_condition_pt": self.options["pre_condition_pt"],
            "sort_component": self.options["sort_component"],
        }
        self.add_subsystem(
            "compute_dep_equilibrium",
            oad.RegisterSubmodel.get_submodel(HE_SUBMODEL_EQUILIBRIUM, options=options_equilibrium),
            promotes=[
                "data:*",
                "settings:*",
                "convergence:*",
                "mass",
                "time_step",
                "thrust_rate_t_econ",
                "non_consumable_energy_t_econ",
                "fuel_consumed_t_econ",
                "fuel_mass_t_econ",
                "fuel_lever_arm_t_econ",
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
                "fuel_mass_t_econ",
                "fuel_lever_arm_t_econ",
            ],
            promotes_outputs=["data:*", "fuel_consumed_t", "fuel_mass_t", "fuel_lever_arm_t"],
        )
        self.add_subsystem("sizing_fuel", SizingEnergy(), promotes=["*"])
        self.add_subsystem("sizing_duration", SizingDuration(), promotes=["*"])
        self.add_subsystem(
            "update_mass",
            UpdateMass(number_of_points=number_of_points),
            promotes=["*"],
        )
