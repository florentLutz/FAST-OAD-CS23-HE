# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


class SizingGaseousHydrogenTankUnusableHydrogen(om.ExplicitComponent):
    """
    Computation of the amount of trapped hydrogen in the tank. The default value of trapped_ratio is
    set slightly higher to prevent underestimation (3%). :cite:`ahluwalia:2010`
    """

    def initialize(self):
        self.options.declare(
            name="gaseous_hydrogen_tank_id",
            default=None,
            desc="Identifier of the gas hydrogen tank",
            allow_none=False,
        )

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
            + ":trapped_ratio",
            val=0.03,
            desc="Ratio between the hydrogen reactant pressure for fuel cell or the injection "
            "pressure for hydrogen engines and filling pressure of hydrogen tank",
        )

        self.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":unusable_fuel_mission",
            units="kg",
            val=0.15,
            desc="Amount of trapped hydrogen in the tank",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

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
                + ":trapped_ratio"
            ]
            * inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":fuel_consumed_mission"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":unusable_fuel_mission",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":trapped_ratio",
        ] = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission"
        ]

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":unusable_fuel_mission",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission",
        ] = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":trapped_ratio"
        ]
