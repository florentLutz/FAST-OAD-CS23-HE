# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingFuelTankUnusableFuel(om.ExplicitComponent):
    """
    Computation of the amount of trapped fuel in that particular tank. Based on a simple ratio
    from :cite:`raymer:2012`.
    """

    def initialize(self):
        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )

    def setup(self):
        fuel_tank_id = self.options["fuel_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":fuel_consumed_mission",
            units="kg",
            val=np.nan,
            desc="Amount of fuel from that tank which will be consumed during mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":unusable_fuel_mission",
            units="kg",
            val=0.5,
            desc="Amount of trapped fuel in the tank",
        )

        self.declare_partials(of="*", wrt="*", val=0.01)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]

        outputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":unusable_fuel_mission"
        ] = (
            0.01
            * inputs[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":fuel_consumed_mission"
            ]
        )
