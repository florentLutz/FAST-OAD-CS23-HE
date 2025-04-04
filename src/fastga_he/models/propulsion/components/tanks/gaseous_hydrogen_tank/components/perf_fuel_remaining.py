# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesGaseousHydrogenRemainingMission(om.ExplicitComponent):
    """
    Computation of the amount of the amount of hydrogen remaining inside the tank.
    """

    def initialize(self):
        self.options.declare(
            name="gaseous_hydrogen_tank_id",
            default=None,
            desc="Identifier of the gaseous hydrogen tank",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission",
            units="kg",
            val=np.nan,
            desc="Total amount of hydrogen loaded in the tank",
        )

        self.add_input(
            "fuel_consumed_t",
            units="kg",
            val=np.full(number_of_points, np.nan),
            desc="Hydrogen from this tank consumed at each time step",
        )

        self.add_output(
            "fuel_remaining_t",
            units="kg",
            val=np.linspace(15.15, 0.15, number_of_points),
            desc="Hydrogen remaining inside the tank at each time step",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        number_of_points = self.options["number_of_points"]

        total_fuel = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission"
        ]

        outputs["fuel_remaining_t"] = np.full(number_of_points, total_fuel) - np.cumsum(
            np.concatenate((np.zeros(1), inputs["fuel_consumed_t"][:-1]))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        number_of_points = self.options["number_of_points"]

        partials[
            "fuel_remaining_t",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission",
        ] = np.ones(number_of_points)

        partials["fuel_remaining_t", "fuel_consumed_t"] = -(
            np.tri(number_of_points, number_of_points) - np.eye(number_of_points)
        )
