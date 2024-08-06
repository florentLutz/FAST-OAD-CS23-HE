# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesFuelRemainingMission(om.ExplicitComponent):
    """
    Computation of the amount of the amount of fuel remaining inside the tank.
    """

    def initialize(self):
        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        fuel_tank_id = self.options["fuel_tank_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_consumed_mission",
            units="kg",
            val=np.nan,
            desc="Total amount of fuel loaded in the tank",
        )
        self.add_input(
            "fuel_consumed_t",
            units="kg",
            val=np.full(number_of_points, np.nan),
            desc="Fuel from this tank consumed at each time step",
        )

        self.add_output(
            "fuel_remaining_t",
            units="kg",
            val=np.linspace(50.5, 0.5, number_of_points),
            desc="Fuel remaining inside the tank at each time step",
        )

        self.declare_partials(
            of="fuel_remaining_t",
            wrt="data:propulsion:he_power_train:fuel_tank:"
            + fuel_tank_id
            + ":fuel_consumed_mission",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
            val=np.ones(number_of_points),
        )

        partials_fuel_consumed = -(
            np.tri(number_of_points, number_of_points) - np.eye(number_of_points)
        )

        self.declare_partials(
            of="fuel_remaining_t",
            wrt="fuel_consumed_t",
            method="exact",
            rows=np.where(partials_fuel_consumed != 0)[0],
            cols=np.where(partials_fuel_consumed != 0)[1],
            val=-np.ones(len(np.where(partials_fuel_consumed != 0)[0])),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]
        number_of_points = self.options["number_of_points"]

        total_fuel = inputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_consumed_mission"
        ]

        outputs["fuel_remaining_t"] = np.full(number_of_points, total_fuel) - np.cumsum(
            np.concatenate((np.zeros(1), inputs["fuel_consumed_t"][:-1]))
        )
