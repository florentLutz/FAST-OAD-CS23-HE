# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
sizing_ducted_fan_cg_y.py
=========================

"""

import openmdao.api as om
import numpy as np

from .constants import POSSIBLE_POSITION


class SizingDuctedFanCGY(om.ExplicitComponent):
    """
    Class that computes the Y-CG of the ducted fan based on its position. Will be based on simple
    geometric ratios, no consideration of volume will be implemented for now.
    """

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

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y_ratio",
            val=np.nan,
            desc="Y position of the ducted fan center of gravity as a ratio of the wing "
            "half-span",
        )

        self.add_output(
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y",
            units="m",
            val=0.0,
            desc="Y position of the ducted fan center of gravity",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]

        outputs["data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y"] = (
            inputs["data:geometry:wing:span"]
            * inputs["data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y_ratio"]
            / 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ducted_fan_id = self.options["ducted_fan_id"]

        partials[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y",
            "data:geometry:wing:span",
        ] = (
            inputs["data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y_ratio"]
            / 2.0
        )
        partials[
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y",
            "data:propulsion:he_power_train:ducted_fan:" + ducted_fan_id + ":CG:y_ratio",
        ] = inputs["data:geometry:wing:span"] / 2.0
