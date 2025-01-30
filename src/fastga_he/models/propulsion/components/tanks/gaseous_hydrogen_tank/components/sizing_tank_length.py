# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingGaseousHydrogenTankLength(om.ExplicitComponent):
    """
    Computation of the length of the tank, which includes the cap from both end.
    """

    def initialize(self):
        self.options.declare(
            name="gaseous_hydrogen_tank_id",
            default=None,
            desc="Identifier of the gaseous hydrogen tank",
            allow_none=False,
        )

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume",
            units="m**3",
            val=np.nan,
            desc="Capacity of the tank in terms of volume",
        )

        self.add_input(
            name="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:inner_diameter",
            units="m",
            val=np.nan,
            desc="Inner diameter of the gaseous hydrogen tanks",
        )

        self.add_input(
            name="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the gaseous hydrogen tanks",
        )

        self.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length",
            val=1.0,
            units="m",
            desc="Value of the cylindrical length of the tank in the x-direction, computed differently based "
            "on the location of the tank",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:outer_diameter",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        d_inner = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:inner_diameter"
        ]

        d_outer = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:outer_diameter"
        ]

        length = (
            inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":inner_volume"
            ]
            - np.pi * d_inner**3 / 6
        ) / (np.pi * d_inner**2 / 4)

        outputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length"
        ] = length + d_outer

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        d_inner = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:inner_diameter"
        ]

        inner_volume = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume"
        ]

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume",
        ] = 1 / (d_inner**2 * np.pi / 4)

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:inner_diameter",
        ] = -8 * inner_volume / (np.pi * d_inner**3) - 2 / 3
