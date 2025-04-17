# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingHighRPMICEDimensions(om.ExplicitComponent):
    """
    Computation of the dimensions of the ICE. Based on the scaling factor and the dimensions of
    the reference ICE, the Rotax 912-A.
    """

    def initialize(self):
        self.options.declare(
            name="high_rpm_ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine for high RPM engine",
            allow_none=False,
        )
        self.options.declare(
            "length_ref",
            default=0.561,
            desc="Length of the reference motor in [m]",
        )
        self.options.declare(
            "width_ref",
            default=0.576,
            desc="Width of the reference motor in [m]",
        )
        self.options.declare(
            "height_ref",
            default=0.576,
            desc="Height of the reference motor in [m]",
        )

    def setup(self):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        for dimension in ["length", "width", "height"]:
            self.add_input(
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":scaling:"
                + dimension,
                val=np.nan,
                desc="Scaling factor for the " + dimension + " of the ICE",
            )

            self.add_output(
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":engine:"
                + dimension,
                val=self.options[dimension + "_ref"],
                desc=dimension.capitalize() + " of the ICE",
                units="m",
            )

            self.declare_partials(
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":engine:"
                + dimension,
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":scaling:"
                + dimension,
                val=self.options[dimension + "_ref"],
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        high_rpm_ice_id = self.options["high_rpm_ice_id"]

        for dimension in ["length", "width", "height"]:
            outputs[
                "data:propulsion:he_power_train:high_rpm_ICE:"
                + high_rpm_ice_id
                + ":engine:"
                + dimension
            ] = (
                inputs[
                    "data:propulsion:he_power_train:high_rpm_ICE:"
                    + high_rpm_ice_id
                    + ":scaling:"
                    + dimension
                ]
                * self.options[dimension + "_ref"]
            )
