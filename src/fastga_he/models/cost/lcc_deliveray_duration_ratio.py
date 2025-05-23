# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDeliveryDurationRatio(om.ExplicitComponent):
    """
    Computation of the ratio between the flight delivery duration and the sizing mission duration.
    """

    def setup(self):
        self.add_input(
            name="data:cost:production:delivery_duration",
            val=np.nan,
            units="h",
        )
        self.add_input(
            name="data:mission:sizing:duration",
            val=np.nan,
            units="h",
        )

        self.add_output(
            name="data:cost:production:delivery:mission_ratio",
            val=0.5,
            desc="Ratio between the delivery flight duration and the sizing mission duration",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:delivery:mission_ratio"] = (
            inputs["data:cost:production:delivery_duration"]
            / inputs["data:mission:sizing:duration"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:cost:production:delivery:mission_ratio", "data:cost:production:delivery_duration"
        ] = 1.0 / inputs["data:mission:sizing:duration"]

        partials["data:cost:production:delivery:mission_ratio", "data:mission:sizing:duration"] = (
            -inputs["data:cost:production:delivery_duration"]
            / inputs["data:mission:sizing:duration"] ** 2.0
        )
