# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class InitializeCoG(om.ExplicitComponent):
    """Computes the center of gravity at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

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

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_taxi_out = inputs["data:mission:sizing:taxi_out:fuel"]
        fuel_takeoff = inputs["data:mission:sizing:takeoff:fuel"]
        fuel_initial_climb = inputs["data:mission:sizing:initial_climb:fuel"]

        # We need the fuel remaining at each time step in the aircraft, so if we assume that the
        # sizing mission is stocked with sizing fuel we can do the following
        fuel = (
            inputs["data:mission:sizing:fuel"]
            - np.cumsum(inputs["fuel_consumed_t"])
            - fuel_taxi_out
            - fuel_takeoff
            - fuel_initial_climb
            + inputs["fuel_consumed_t"][0]
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
