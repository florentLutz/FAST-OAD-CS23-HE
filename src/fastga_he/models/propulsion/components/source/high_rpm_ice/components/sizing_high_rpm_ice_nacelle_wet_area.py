# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHighRPMICENacelleWetArea(om.ExplicitComponent):
    """
    Computation of the wet area of the ICE nacelle. Based on some very simple geometric
    considerations.
    """

    def initialize(self):
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        self.add_input(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:length",
            val=np.nan,
            desc="Length of the ICE nacelle",
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:width",
            val=np.nan,
            desc="Width of the ICE nacelle",
            units="m",
        )
        self.add_input(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:height",
            val=np.nan,
            desc="Height of the ICE nacelle",
            units="m",
        )

        self.add_output(
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:wet_area",
            val=0.5,
            desc="Wet area of the ICE nacelle",
            units="m**2",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        outputs[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:wet_area"
        ] = (
            2.0
            * (
                inputs[
                    "data:propulsion:he_power_train:high_rpm_ICE:"
                    + high_rpm_ice_id
                    + ":nacelle:height"
                ]
                + inputs[
                    "data:propulsion:he_power_train:high_rpm_ICE:"
                    + high_rpm_ice_id
                    + ":nacelle:width"
                ]
            )
            * inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:length"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        partials[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:wet_area",
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:height",
        ] = (
            2.0
            * inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:length"
            ]
        )
        partials[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:wet_area",
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:width",
        ] = (
            2.0
            * inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:length"
            ]
        )
        partials[
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:wet_area",
            "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:length",
        ] = 2.0 * (
            inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:height"
            ]
            + inputs[
                "data:propulsion:he_power_train:high_rpm_ICE:" + high_rpm_ice_id + ":nacelle:width"
            ]
        )
