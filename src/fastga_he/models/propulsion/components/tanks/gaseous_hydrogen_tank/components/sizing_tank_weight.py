# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import POSSIBLE_POSITION


class SizingGaseousHydrogenTankWeight(om.ExplicitComponent):
    """
    Computation of the weight of the tank. The very simplistic approach we will use is to say
    that weight of tank is the weight of unused fuel and the weight of the tank itself
    :cite:`colozza:2002`.
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
        in_fuselage = position == "in_the_cabin" or position == "in_the_back"

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume",
            units="m**3",
            val=np.nan,
            desc="Capacity of the tank in terms of volume",
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

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length",
            val=np.nan,
            units="m",
            desc="Value of the length of the tank in the x-direction",
        )
        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the gaseous hydrogen tank",
        )

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":material:density",
            units="kg/m**3",
            val=7860.0,
            desc="Choice of the tank material,Some reference (in kg/m**3): "
            "Steel(ASTM-A514):7860, Aluminum(2014-T6):2800, Titanium(6%Al,4%V):4460, "
            "Carbon Composite:1530",
        )

        self.add_output(
            name="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass",
            units="kg",
            val=20.0,
            desc="Weight of the gaseous hydrogen tanks",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]
        in_fuselage = position == "in_the_cabin" or position == "in_the_back"

        wall_density = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":material:density"
        ]

        d_tank = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:outer_diameter"
        ]

        length = (
            inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length"
            ]
            - inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ]
        )

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

        outputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass"
        ] = wall_density * (
            nb_tank * (np.pi * d_tank**3 / 6 + np.pi * d_tank**2 * length / 4) - inner_volume
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]
        in_fuselage = position == "in_the_cabin" or position == "in_the_back"

        d_tank = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:outer_diameter"
        ]

        length = (
            inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length"
            ]
            - inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:outer_diameter"
            ]
        )

        wall_density = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":material:density"
        ]

        inner_volume = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume"
        ]

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume",
        ] = -wall_density

        if in_fuselage:
            nb_tank = inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":number_of_tank"
            ]

            partials[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":mass",
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":number_of_tank",
            ] = wall_density * (np.pi * d_tank**3 / 6 + np.pi * d_tank**2 * length / 4)

        else:
            nb_tank = 1.0

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":material:density",
        ] = nb_tank * (np.pi * d_tank**3 / 6 + np.pi * d_tank**2 * length / 4) - inner_volume

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length",
        ] = nb_tank * wall_density * np.pi * d_tank**2 / 4

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:outer_diameter",
        ] = (
            nb_tank
            * wall_density
            * (-np.pi * d_tank**2 / 4 + np.pi * d_tank * (length + d_tank) / 2)
        )
