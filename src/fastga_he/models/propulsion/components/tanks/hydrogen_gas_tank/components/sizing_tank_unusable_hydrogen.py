# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
from utils.filter_residuals import filter_residuals


class SizingHydrogenGasTankUnusableHydrogen(om.ExplicitComponent):
    """
    Computation of the amount of trapped fuel in that particular tank.
    """

    def initialize(self):
        self.options.declare(
            name="hydrogen_gas_tank_id",
            default=None,
            desc="Identifier of the hydrogen gas tank",
            allow_none=False,
        )

    def setup(self):
        # To modify based on the minimum pressure for the output hydrogen mass flow
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":fuel_consumed_mission",
            units="kg",
            val=15.0,
            desc="Amount of hydrogen from that tank which will be consumed during mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":unusable_fuel_mission",
            units="kg",
            val=0.15,
            desc="Amount of trapped hydrogen in the tank",
        )

        self.declare_partials(of="*", wrt="*", val=0.01)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # To modify based on the minimum pressure for the output hydrogen mass flow
        hydrogen_gas_tank_id = self.options["hydrogen_gas_tank_id"]

        outputs[
            "data:propulsion:he_power_train:hydrogen_gas_tank:"
            + hydrogen_gas_tank_id
            + ":unusable_fuel_mission"
        ] = (
            0.03
            * inputs[
                "data:propulsion:he_power_train:hydrogen_gas_tank:"
                + hydrogen_gas_tank_id
                + ":fuel_consumed_mission"
            ]
        )
