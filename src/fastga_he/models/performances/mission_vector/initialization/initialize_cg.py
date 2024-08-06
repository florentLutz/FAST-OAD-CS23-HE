# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class InitializeCoG(om.ExplicitComponent):
    """Computes the center of gravity at each time step."""

    def initialize(self):
        self.options.declare(
            "number_of_points_climb", default=1, desc="number of equilibrium to be treated in climb"
        )
        self.options.declare(
            "number_of_points_cruise",
            default=1,
            desc="number of equilibrium to be treated in " "cruise",
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

        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment",
            val=np.nan,
            units="kg*m",
        )
        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass", val=np.nan, units="kg"
        )
        self.add_input(
            "fuel_mass_t", shape=number_of_points, val=np.full(number_of_points, np.nan), units="kg"
        )
        self.add_input(
            "fuel_lever_arm_t",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="kg*m",
        )

        self.add_output("x_cg", shape=number_of_points, units="m")

        self.declare_partials(
            of="x_cg",
            wrt=["fuel_mass_t", "fuel_lever_arm_t"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt=[
                "data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass",
                "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_mass_t = inputs["fuel_mass_t"]
        fuel_mass_t = np.clip(fuel_mass_t, 0.0, None)

        fuel_lever_arm_t = inputs["fuel_lever_arm_t"]
        fuel_lever_arm_t = np.clip(fuel_lever_arm_t, 0.0, None)

        equivalent_moment = inputs[
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"
        ]
        equivalent_mass = inputs["data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"]

        x_cg = (equivalent_moment + fuel_lever_arm_t) / (equivalent_mass + fuel_mass_t)

        outputs["x_cg"] = x_cg

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fuel_mass_t = inputs["fuel_mass_t"]
        fuel_mass_t = np.clip(fuel_mass_t, 0.0, None)

        fuel_lever_arm_t = inputs["fuel_lever_arm_t"]
        fuel_lever_arm_t = np.clip(fuel_lever_arm_t, 0.0, None)

        equivalent_moment = inputs[
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"
        ]
        equivalent_mass = inputs["data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"]

        partials[
            "x_cg", "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"
        ] = 1.0 / (fuel_mass_t + equivalent_mass)
        partials["x_cg", "fuel_lever_arm_t"] = 1.0 / (fuel_mass_t + equivalent_mass)

        partials["x_cg", "data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"] = (
            -(equivalent_moment + fuel_lever_arm_t) / (equivalent_mass + fuel_mass_t) ** 2.0
        )
        partials["x_cg", "fuel_mass_t"] = (
            -(equivalent_moment + fuel_lever_arm_t) / (equivalent_mass + fuel_mass_t) ** 2.0
        )
