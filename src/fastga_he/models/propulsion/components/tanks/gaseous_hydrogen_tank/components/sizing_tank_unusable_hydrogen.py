# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingGaseousHydrogenTankUnusableHydrogen(om.ExplicitComponent):
    """
    Computation of the amount of trapped hydrogen in the tank.
    """

    def initialize(self):
        self.options.declare(
            name="gaseous_hydrogen_tank_id",
            default=None,
            desc="Identifier of the gas hydrogen tank",
            allow_none=False,
        )

        self.options.declare(
            name="trapped_ratio",
            default=0.03,
            desc="Ratio between typical empty pressure and filling pressure of hydrogen tank.",
            allow_none=False,
        )
        # "The default value is set slightly higher to prevent underestimation."
        # :cite:`ahluwalia:2010`

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission",
            units="kg",
            val=15.0,
            desc="Amount of hydrogen from that tank which will be consumed during mission",
        )

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_total_mission",
            units="kg",
            val=np.nan,
            desc="Total amount of hydrogen loaded in the tank for the mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":unusable_fuel_mission",
            units="kg",
            val=0.15,
            desc="Amount of trapped hydrogen in the tank",
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_total_mission",
            val=1.0,
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission",
            val=-1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # To modify based on the minimum pressure for the output hydrogen mass flow
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        outputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":unusable_fuel_mission"
        ] = (
            inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":fuel_total_mission"
            ]
            - inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":fuel_consumed_mission"
            ]
        )
