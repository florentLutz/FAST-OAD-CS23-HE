# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingFuelTankWidth(om.ExplicitComponent):
    """
    Computation of the reference width.
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
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume",
            units="m**3",
            val=np.nan,
            desc="Capacity of the tank in terms of volume",
        )
        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height",
            val=np.nan,
            units="m",
            desc="Value of the length of the tank in the z-direction, computed differently based "
            "on the location of the tank",
        )
        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length",
            val=np.nan,
            units="m",
            desc="Value of the length of the tank in the x-direction, computed differently based "
            "on the location of the tank",
        )

        self.add_output(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width",
            val=np.nan,
            units="m",
            desc="Value of the length of the tank in the y-direction, computed differently based "
            "on the location of the tank",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuel_tank_id = self.options["fuel_tank_id"]

        tank_volume = inputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume"]
        tank_length = inputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length"
        ]
        tank_height = inputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height"
        ]

        outputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width"
        ] = tank_volume / (tank_length * tank_height)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        fuel_tank_id = self.options["fuel_tank_id"]

        tank_volume = inputs["data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume"]
        tank_length = inputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length"
        ]
        tank_height = inputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height"
        ]

        partials[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width",
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume",
        ] = 1.0 / (tank_length * tank_height)
        partials[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width",
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:length",
        ] = -tank_volume / (tank_length ** 2.0 * tank_height)
        partials[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width",
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height",
        ] = -tank_volume / (tank_length * tank_height ** 2.0)
