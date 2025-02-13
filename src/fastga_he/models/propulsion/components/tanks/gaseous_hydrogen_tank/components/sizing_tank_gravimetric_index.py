# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingGaseousHydrogenTankGravimetricIndex(om.ExplicitComponent):
    """
    Computation of the gravimetric index of gaseous hydrogen tank, ratio between hydrogen
    capacity and overall system weight :cite:`mukhopadhaya:2022`.
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
            name="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass",
            units="kg",
            val=np.nan,
            desc="Weight of the gaseous hydrogen tanks",
        )

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity",
            units="kg",
            val=np.nan,
            desc="Capacity of the tank in terms of weight",
        )

        self.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":gravimetric_index",
            val=0.5,
            desc="Ratio between the maximum capacity and overall system weight",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        mass = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass"
        ]
        capacity = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity"
        ]

        outputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":gravimetric_index"
        ] = capacity / (capacity + mass)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        mass = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass"
        ]
        capacity = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity"
        ]
        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":gravimetric_index",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":capacity",
        ] = mass / (capacity + mass) ** 2

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":gravimetric_index",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":mass",
        ] = -capacity / (capacity + mass) ** 2
