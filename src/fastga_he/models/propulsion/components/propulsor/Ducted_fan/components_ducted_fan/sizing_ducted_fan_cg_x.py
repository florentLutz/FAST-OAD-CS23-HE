# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
sizing_ducted_fan_cg_x.py
=========================

The ducted fan is always mounted on the wing leading edge (see
constants.py), so CG_x is computed directly from the duct depth and the
distance to the leading edge.
"""

import numpy as np

import openmdao.api as om

from .constants import POSSIBLE_POSITION


class SizingDuctedFanCGX(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="ducted_fan_id", default=None, desc="Identifier of the ducted fan", allow_none=False
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the ducted fan, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        ducted_fan_id = self.options["ducted_fan_id"]

        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":depth",
            val=np.nan,
            units="m",
            desc="Depth of the ducted fan",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":from_LE",
            val=np.nan,
            units="m",
            desc="Distance between the ducted fan and the leading edge",
        )
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:x",
            units="m",
            val=2.5,
            desc="X position of the ducted fan center of gravity",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]

        depth = inputs["data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":depth"]
        distance_from_le = inputs[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":from_LE"
        ]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]

        outputs["data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:x"] = (
            fa_length - 0.25 * l0_wing - distance_from_le - 0.5 * depth
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]

        partials[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:x",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":depth",
        ] = -0.5
        partials[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:x",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":from_LE",
        ] = -1.0
        partials[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:x",
            "data:geometry:wing:MAC:length",
        ] = -0.25
        partials[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:x",
            "data:geometry:wing:MAC:at25percent:x",
        ] = 1.0
