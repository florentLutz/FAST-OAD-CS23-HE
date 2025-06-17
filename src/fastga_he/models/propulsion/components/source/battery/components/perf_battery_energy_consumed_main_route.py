# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesEnergyConsumedMainRoute(om.ExplicitComponent):
    """
    Summation over all the main route of the energy drawn for that battery. We have a computation at
    aircraft level but we need one at component level as well.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        # Default is set as None, because at group level if the option isn't set this component
        # shouldn't be added so it is a second safety
        self.options.declare(
            "number_of_points_reserve",
            default=None,
            desc="number of equilibrium to be treated in reserve",
            types=int,
        )
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        number_of_points_reserve = self.options["number_of_points_reserve"]
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input(
            "non_consumable_energy_t",
            val=np.full(number_of_points, np.nan),
            desc="fuel consumed at each time step in the battery",
            units="W*h",
        )
        self.add_output(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":energy_consumed_main_route",
            units="W*h",
            val=50e3,
            desc="Energy drawn from the battery for the mission",
        )

        val_partial = np.ones(number_of_points)
        val_partial[-number_of_points_reserve - 1 : -1] = np.zeros(number_of_points_reserve)

        self.declare_partials(
            of="*",
            wrt="*",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=val_partial,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        # TODO: this way of "extracting" the reserve will only work with the current "format" for
        #  the points in the mission (1pt taxi_out -> climb -> cruise -> descent -> 1pt taxi_in)
        outputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":energy_consumed_main_route"
        ] = np.sum(inputs["non_consumable_energy_t"]) - np.sum(
            inputs["non_consumable_energy_t"][-number_of_points_reserve - 1 : -1]
        )
