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
            "fuel_consumed_t",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="kg",
        )

        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", val=np.nan, units="kg")

        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment",
            val=np.nan,
            units="kg*m",
        )
        self.add_input("data:weight:propulsion:tank:CG:x", val=np.nan, units="m")
        self.add_input(
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass", val=np.nan, units="kg"
        )

        self.add_output("x_cg", shape=number_of_points, units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_taxi_out = inputs["data:mission:sizing:taxi_out:fuel"]
        fuel_takeoff = inputs["data:mission:sizing:takeoff:fuel"]
        fuel_initial_climb = inputs["data:mission:sizing:initial_climb:fuel"]

        # We need the fuel remaining at each time step in the aircraft, so if we assume that the
        # sizing mission is stocked with sizing fuel we can do the following
        fuel = (
            inputs["data:mission:sizing:fuel"]
            - fuel_taxi_out
            - fuel_takeoff
            - fuel_initial_climb
            - np.cumsum(np.concatenate((np.zeros(1), inputs["fuel_consumed_t"][:-1])))
        )

        if np.any(fuel) < 0.0:
            print("Negative fuel consumed for a point, consumption replaced")

        fuel = np.where(fuel >= 0.0, fuel, np.zeros_like(fuel))

        equivalent_moment = inputs[
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"
        ]
        cg_tank = inputs["data:weight:propulsion:tank:CG:x"]
        equivalent_mass = inputs["data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"]
        x_cg = (equivalent_moment + cg_tank * fuel) / (equivalent_mass + fuel)

        outputs["x_cg"] = x_cg

    def compute_partials(self, inputs, partials, discrete_inputs=None):
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

        fuel_taxi_out = inputs["data:mission:sizing:taxi_out:fuel"]
        fuel_takeoff = inputs["data:mission:sizing:takeoff:fuel"]
        fuel_initial_climb = inputs["data:mission:sizing:initial_climb:fuel"]

        fuel = (
            inputs["data:mission:sizing:fuel"]
            - fuel_taxi_out
            - fuel_takeoff
            - fuel_initial_climb
            - np.cumsum(np.concatenate((np.zeros(1), inputs["fuel_consumed_t"][:-1])))
        )

        equivalent_moment = inputs[
            "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"
        ]
        cg_tank = inputs["data:weight:propulsion:tank:CG:x"]
        equivalent_mass = inputs["data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"]

        d_cg_d_fuel = (cg_tank * equivalent_mass - equivalent_moment) / (
            equivalent_mass + fuel
        ) ** 2.0

        partials["x_cg", "data:mission:sizing:fuel"] = d_cg_d_fuel
        partials["x_cg", "data:mission:sizing:taxi_out:fuel"] = -d_cg_d_fuel
        partials["x_cg", "data:mission:sizing:takeoff:fuel"] = -d_cg_d_fuel
        partials["x_cg", "data:mission:sizing:initial_climb:fuel"] = -d_cg_d_fuel

        d_cg_d_fuel_t = np.tile(-d_cg_d_fuel, (number_of_points, 1))
        d_cg_d_fuel_t = np.tril(np.transpose(d_cg_d_fuel_t), -1)

        partials["x_cg", "fuel_consumed_t"] = d_cg_d_fuel_t

        partials["x_cg", "data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass"] = (
            -(equivalent_moment + cg_tank * fuel) / (equivalent_mass + fuel) ** 2.0
        )
        partials[
            "x_cg", "data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment"
        ] = 1.0 / (equivalent_mass + fuel)
        partials["x_cg", "data:weight:propulsion:tank:CG:x"] = fuel / (equivalent_mass + fuel)
