# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)


class SizingGaseousHydrogenTankWeight(om.ExplicitComponent):
    """
    Computation of the weight of the tank. The very simplistic approach we will use is to say
    that weight of tank is the weight of unused fuel and the weight of the tank itself.
    :cite:`colozza:2002`
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
            "Steel(ASTM-A514):7860, Aluminum(2014-T6):2800, Titanium(6%Al,4%V):4460, Carbon Composite:1530",
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

        wall_density = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":material:density"
        ]

        d = inputs[
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

        outputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass"
        ] = wall_density * (np.pi * d**3 / 6 + np.pi * d**2 * length / 4 - inner_volume)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        d = inputs[
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
            + ":material:density",
        ] = np.pi * d**3 / 6 + np.pi * d**2 * length / 4 - inner_volume

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length",
        ] = wall_density * np.pi * d**2 / 4

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":inner_volume",
        ] = -wall_density

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:outer_diameter",
        ] = wall_density * (-np.pi * d**2 / 4 + np.pi * d * (length + d) / 2)
