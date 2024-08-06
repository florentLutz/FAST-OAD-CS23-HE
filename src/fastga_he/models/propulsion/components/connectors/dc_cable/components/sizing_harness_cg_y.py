# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingHarnessCGY(om.ExplicitComponent):
    """
    Class that computes the CG of the DC cable based on the position of its source and target.
    Will be based on simple geometric ratios, no consideration of volume will be implemented for
    now. To make things simpler and because the individual contribution of each cable won't amount
    to too much, we will consider the Y-CG to be 0 all the time except when inside the wing.
    """

    def initialize(self):
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="from_rear_to_front",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the cable harness, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        harness_id = self.options["harness_id"]
        position = self.options["position"]

        # At least one input is needed regardless of the case
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:y",
            units="m",
            val=0.0,
            desc="Y position of the DC bus center of gravity",
        )

        if position == "inside_the_wing":
            self.add_input(
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:y_ratio",
                val=np.nan,
                desc="X position of the DC bus center of gravity as a ratio of the wing half-span",
            )

            self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        position = self.options["position"]
        harness_id = self.options["harness_id"]

        if position == "inside_the_wing":
            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:y"] = (
                inputs["data:geometry:wing:span"]
                * inputs[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:y_ratio"
                ]
                / 2.0
            )

        else:
            outputs["data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:y"] = 0.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]
        position = self.options["position"]

        if position == "inside_the_wing":
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:y",
                "data:geometry:wing:span",
            ] = (
                inputs[
                    "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:y_ratio"
                ]
                / 2.0
            )
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:y",
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":CG:y_ratio",
            ] = inputs["data:geometry:wing:span"] / 2.0
