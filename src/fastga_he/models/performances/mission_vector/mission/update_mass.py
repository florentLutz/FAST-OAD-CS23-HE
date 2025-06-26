# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class UpdateMass(om.ExplicitComponent):
    """Update mass for next iteration."""

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input(
            "fuel_consumed_t",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="kg",
        )
        self.add_input("data:mission:sizing:taxi_out:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", val=np.nan, units="kg")

        self.add_output(
            "mass", shape=number_of_points, val=np.full(number_of_points, 1500.0), units="kg"
        )

        self.declare_partials(
            of="mass",
            wrt="data:weight:aircraft:MTOW",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
            val=1.0,
        )
        self.declare_partials(
            of="mass",
            wrt=[
                "data:mission:sizing:taxi_out:fuel",
                "data:mission:sizing:initial_climb:fuel",
                "data:mission:sizing:takeoff:fuel",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
            val=1.0,
        )

        partials_fuel_consumed = -(
            np.tri(number_of_points, number_of_points) - np.eye(number_of_points)
        )
        self.declare_partials(
            of="mass",
            wrt="fuel_consumed_t",
            method="exact",
            val=np.ones(len(np.where(partials_fuel_consumed != 0)[0])),
            rows=np.where(partials_fuel_consumed != 0)[0],
            cols=np.where(partials_fuel_consumed != 0)[1],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mtow = inputs["data:weight:aircraft:MTOW"]
        fuel_taxi_out = inputs["data:mission:sizing:taxi_out:fuel"]
        fuel_takeoff = inputs["data:mission:sizing:takeoff:fuel"]
        fuel_initial_climb = inputs["data:mission:sizing:initial_climb:fuel"]
        number_of_points = self.options["number_of_points"]

        # The + inputs["fuel_consumed_t"][0] is due to the fact that the mass is the one we use
        # to compute the equilibrium so, in a sense it is before the fuel is burned so the first
        # point of the mass vector is gonna be a t the very start of climb which means the first
        # kg of fuel will not have been consumed. Only the fuel for taxi out, takeoff and initial
        # climb is considered
        a = inputs["fuel_consumed_t"][:-1]
        outputs["mass"] = (
            np.full(number_of_points, mtow)
            - fuel_taxi_out
            - fuel_takeoff
            - fuel_initial_climb
            - np.cumsum(np.concatenate((np.zeros(1), inputs["fuel_consumed_t"][:-1])))
        )

