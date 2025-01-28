# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import logging

_LOGGER = logging.getLogger(__name__)


class SizingGaseousHydrogenTankGravimetricIndex(om.ExplicitComponent):
    """
    Computation of the gravimetric index of gaseous hydrogen tank,
    ratio between total mission fuel weight and overall weight.
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
            + ":fuel_total_mission",
            units="kg",
            val=np.nan,
            desc="Total amount of hydrogen loaded in the tank for the mission",
        )

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission",
            units="kg",
            val=15.0,
            desc="Amount of hydrogen from that tank which will be consumed during mission",
        )

        self.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":gravimetric_index",
            val=10.0,
            desc="Ratio between the mission used weight and overall weight",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        outputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":gravimetric_index"
        ] = inputs[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission"
        ] / (
            inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":fuel_total_mission"
            ]
            + inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:" + gaseous_hydrogen_tank_id + ":mass"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":gravimetric_index",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_consumed_mission",
        ] = 1 / (
            inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":fuel_total_mission"
            ]
            + inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:" + gaseous_hydrogen_tank_id + ":mass"
            ]
        )

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":gravimetric_index",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:" + gaseous_hydrogen_tank_id + ":mass",
        ] = (
            -inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":fuel_consumed_mission"
            ]
            / (
                inputs[
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                    + gaseous_hydrogen_tank_id
                    + ":fuel_total_mission"
                ]
                + inputs[
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                    + gaseous_hydrogen_tank_id
                    + ":mass"
                ]
            )
            ** 2
        )

        partials[
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":gravimetric_index",
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":fuel_total_mission",
        ] = (
            -inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":fuel_consumed_mission"
            ]
            / (
                inputs[
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                    + gaseous_hydrogen_tank_id
                    + ":fuel_total_mission"
                ]
                + inputs[
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                    + gaseous_hydrogen_tank_id
                    + ":mass"
                ]
            )
            ** 2
        )
