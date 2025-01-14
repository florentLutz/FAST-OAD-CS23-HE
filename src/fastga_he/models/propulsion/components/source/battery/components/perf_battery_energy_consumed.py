# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesEnergyConsumed(om.ExplicitComponent):
    """
    Summation over all the mission of the energy drawn for that battery. We have a computation at
    aircraft level but we need one at component level as well.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
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
            + ":energy_consumed_mission",
            units="W*h",
            val=50e3,
            desc="Energy drawn from the battery for the mission",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":energy_consumed_mission"
        ] = np.sum(inputs["non_consumable_energy_t"])
