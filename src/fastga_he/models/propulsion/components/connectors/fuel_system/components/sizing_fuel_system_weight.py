# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingFuelSystemWeight(om.ExplicitComponent):
    """
    Computation of the fuel system weight, based on a formula from :cite:`gudmundsson:2013` for
    Torenbeek approach. Include the weight of fuel tanks, pipes, pumps, vents, ...
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.exponent = None
        self.factor = None

    def initialize(self):

        self.options.declare(
            name="fuel_system_id",
            default=None,
            desc="Identifier of the fuel system",
            types=str,
            allow_none=False,
        )

    def setup(self):

        fuel_system_id = self.options["fuel_system_id"]

        self.add_input(
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":connected_volume",
            units="galUS",
            val=np.nan,
            desc="Capacity of the connected tank in terms of volume",
        )
        self.add_input(
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":number_engine",
            val=np.nan,
            desc="Number of engine connected to this fuel system",
        )
        self.add_input(
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":fuel_type",
            val=1.0,
            desc="Type of fuel flowing in the system, 1.0 - gasoline, 2.0 - Diesel, 3.0 - Jet A1",
        )

        self.add_output(
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass",
            units="lbm",
            val=20.0,
            desc="Weight of the fuel system",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass",
            wrt="data:propulsion:he_power_train:fuel_system:"
            + fuel_system_id
            + ":connected_volume",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuel_system_id = self.options["fuel_system_id"]

        volume = inputs[
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":connected_volume"
        ]
        number_of_engine = float(
            inputs[
                "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":number_engine"
            ]
        )
        fuel_type = inputs[
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":fuel_type"
        ]

        if fuel_type == 3.0:
            self.exponent = 1.0
            self.factor = 0.40
        else:
            self.exponent = 0.667 if number_of_engine == 1.0 else 0.6
            self.factor = 2.0 if number_of_engine == 1.0 else 4.5

        outputs["data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass"] = (
            self.factor * volume ** self.exponent
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        fuel_system_id = self.options["fuel_system_id"]

        volume = inputs[
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":connected_volume"
        ]

        partials[
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":mass",
            "data:propulsion:he_power_train:fuel_system:" + fuel_system_id + ":connected_volume",
        ] = (
            self.factor * self.exponent * volume ** (self.exponent - 1.0)
        )
