# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
from ..constants import POSSIBLE_POSITION


class SizingGaseousHydrogenTankLengthFuselageConstraints(om.ExplicitComponent):
    """
    Computation to check the overall length of the tank will fit in the fuselage or not.
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

        self.add_input(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length",
            val=np.nan,
            units="m",
            desc="Value of the length of the tank in the x-direction",
        )

        if position == "in_the_cabin" or position == "underbelly":
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        if position == "in_the_back":
            self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
            self.add_input(
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:rear_length_ratio",
                val=0.5,
                desc="The ratio between the usable length of the rear fuselage "
                "and the the whole rear fuselage length.",
            )

        self.add_output(
            "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:"
            + gaseous_hydrogen_tank_id
            + ":dimension:length",
            val=0.0,
            units="m",
            desc="Constraints on the tank length w.r.t "
            "the cabin/rear_fuselage length,  respected if <0",
        )

        if position != "wing_pod":
            self.declare_partials(
                of="*",
                wrt="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length",
                val=1.0,
            )

        else:
            self.declare_partials(
                of="*",
                wrt="data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length",
                val=0.0,
            )

        if position == "in_the_cabin" or position == "underbelly":
            self.declare_partials(
                of="*",
                wrt="data:geometry:cabin:length",
                val=-1.0,
            )

        elif position == "in_the_back":
            self.declare_partials(
                of="*",
                wrt=[
                    "data:geometry:fuselage:rear_length",
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                    + gaseous_hydrogen_tank_id
                    + ":dimension:rear_length_ratio",
                    "data:geometry:fuselage:rear_length",
                ],
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]

        if position == "in_the_cabin" or position == "underbelly":
            outputs[
                "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length"
            ] = (
                inputs[
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                    + gaseous_hydrogen_tank_id
                    + ":dimension:length"
                ]
                - inputs["data:geometry:cabin:length"]
            )

        elif position == "in_the_back":
            outputs[
                "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length"
            ] = (
                inputs[
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                    + gaseous_hydrogen_tank_id
                    + ":dimension:length"
                ]
                - inputs[
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                    + gaseous_hydrogen_tank_id
                    + ":dimension:rear_length_ratio"
                ]
                * inputs["data:geometry:fuselage:rear_length"]
            )

        else:
            outputs[
                "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length"
            ] = -1.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gaseous_hydrogen_tank_id = self.options["gaseous_hydrogen_tank_id"]
        position = self.options["position"]

        if position == "in_the_back":
            partials[
                "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length",
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:rear_length_ratio",
            ] = -inputs["data:geometry:fuselage:rear_length"]

            partials[
                "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:length",
                "data:geometry:fuselage:rear_length",
            ] = -inputs[
                "data:propulsion:he_power_train:gaseous_hydrogen_tank:"
                + gaseous_hydrogen_tank_id
                + ":dimension:rear_length_ratio"
            ]
