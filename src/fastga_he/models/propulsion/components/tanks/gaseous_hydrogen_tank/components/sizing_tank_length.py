# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
from ..constants import POSSIBLE_POSITION


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
        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the gaseous hydrogen tank, "
            "possible position include " + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]
        in_fuselage = not (position == "wing_pod" or position == "underbelly")

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
        if in_fuselage:
            self.add_input(
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":number_of_tank",
                val=1.0,
                desc="Number of gaseous hydrogen tank in a stack in fuselage. "
                "Default set 1.0 for single tank in fuselage and outside fuselage uses.",
            )

        self.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length",
            val=1.0,
            units="m",
            desc="Value of the cylindrical length of the tank in the x-direction, "
            "computed differently based on the location of the tank.",
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
        position = self.options["position"]
        in_fuselage = position == "in_the_cabin" or position == "in_the_back"

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

        inner_volume = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume"
        ]
        if in_fuselage:
            nb_tank = inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":number_of_tank"
            ]
        else:
            nb_tank = 1.0

        length = (inner_volume - np.pi * d_inner**3 / 6 * nb_tank) / (
            np.pi * nb_tank * d_inner**2 / 4
        )

        outputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length"
        ] = length + d_outer

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]
        in_fuselage = position == "in_the_cabin" or position == "in_the_back"

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

        if in_fuselage:
            nb_tank = inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":number_of_tank"
            ]
            partials[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length",
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":number_of_tank",
            ] = -4 * inner_volume / (np.pi * d_inner**2 * nb_tank**2)
        else:
            nb_tank = 1.0

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume",
        ] = 1 / (d_inner**2 * np.pi * nb_tank / 4)

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:inner_diameter",
        ] = -(2 * np.pi * nb_tank * d_inner**3 + 24 * inner_volume) / (
            3 * np.pi * nb_tank * d_inner**3
        )
