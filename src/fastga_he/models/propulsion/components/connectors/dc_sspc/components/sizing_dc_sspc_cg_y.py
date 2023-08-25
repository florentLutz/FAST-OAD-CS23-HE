# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingDCSSPCCGY(om.ExplicitComponent):
    """
    Class that computes the Y-CG of the DC SSPC based on its position. Will be based on simple
    geometric ratios, no consideration of volume will be implemented for now.
    """

    def initialize(self):

        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the DC SSPC, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        position = self.options["position"]
        dc_sspc_id = self.options["dc_sspc_id"]

        # At least one input is needed regardless of the case
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.add_output(
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":CG:y",
            units="m",
            val=0.0,
            desc="Y position of the DC bus center of gravity",
        )

        if position == "inside_the_wing":

            self.add_input(
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":CG:y_ratio",
                val=np.nan,
                desc="Y position of the DC SSPC center of gravity as a ratio of the wing half-span",
            )

            self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        position = self.options["position"]
        dc_sspc_id = self.options["dc_sspc_id"]

        if position == "inside_the_wing":

            outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":CG:y"] = (
                inputs["data:geometry:wing:span"]
                * inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":CG:y_ratio"]
                / 2.0
            )

        else:

            outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":CG:y"] = 0.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_sspc_id = self.options["dc_sspc_id"]
        position = self.options["position"]

        if position == "inside_the_wing":

            partials[
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":CG:y",
                "data:geometry:wing:span",
            ] = (
                inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":CG:y_ratio"] / 2.0
            )
            partials[
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":CG:y",
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":CG:y_ratio",
            ] = (
                inputs["data:geometry:wing:span"] / 2.0
            )
