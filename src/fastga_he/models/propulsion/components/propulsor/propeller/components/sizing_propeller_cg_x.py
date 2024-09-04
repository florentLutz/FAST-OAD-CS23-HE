# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np

import openmdao.api as om

from ..constants import POSSIBLE_POSITION


class SizingPropellerCGX(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the propeller, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        propeller_id = self.options["propeller_id"]
        position = self.options["position"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":depth",
            val=np.nan,
            units="m",
            desc="Depth of the propeller",
        )

        if position == "on_the_wing":
            self.add_input(
                name="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_LE",
                val=np.nan,
                units="m",
                desc="Distance between the propeller and the leading edge",
            )
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
            self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the propeller center of gravity",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]
        position = self.options["position"]

        prop_depth = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":depth"]

        if position == "on_the_wing":
            distance_from_le = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_LE"
            ]
            l0_wing = inputs["data:geometry:wing:MAC:length"]
            fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]

            outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":CG:x"] = (
                fa_length - 0.25 * l0_wing - distance_from_le - 0.5 * prop_depth
            )

        else:
            # In the nose
            outputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":CG:x"] = (
                0.5 * prop_depth
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]
        position = self.options["position"]

        if position == "on_the_wing":
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":CG:x",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":depth",
            ] = -0.5
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":CG:x",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_LE",
            ] = -1.0
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":CG:x",
                "data:geometry:wing:MAC:length",
            ] = -0.25
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":CG:x",
                "data:geometry:wing:MAC:at25percent:x",
            ] = 1.0

        else:
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":CG:x",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":depth",
            ] = 0.5
