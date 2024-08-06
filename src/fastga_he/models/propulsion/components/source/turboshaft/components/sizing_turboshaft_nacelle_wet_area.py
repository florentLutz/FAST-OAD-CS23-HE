# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np



class SizingTurboshaftNacelleWetArea(om.ExplicitComponent):
    """
    Computation of the wet area of the turboshaft nacelle. Based on some very simple geometric
    considerations.
    """

    def initialize(self):
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):
        turboshaft_id = self.options["turboshaft_id"]

        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:length",
            val=np.nan,
            desc="Length of the turboshaft nacelle",
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:width",
            val=np.nan,
            desc="Width of the turboshaft nacelle",
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:height",
            val=np.nan,
            desc="Height of the turboshaft nacelle",
            units="m",
        )

        self.add_output(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:wet_area",
            val=0.5,
            desc="Wet area of the turboshaft nacelle",
            units="m**2",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        outputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:wet_area"
        ] = (
            2.0
            * (
                inputs[
                    "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:height"
                ]
                + inputs[
                    "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:width"
                ]
            )
            * inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:length"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:wet_area",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:height",
        ] = (
            2.0
            * inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:length"
            ]
        )
        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:wet_area",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:width",
        ] = (
            2.0
            * inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:length"
            ]
        )
        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:wet_area",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:length",
        ] = 2.0 * (
            inputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:height"]
            + inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":nacelle:width"
            ]
        )
