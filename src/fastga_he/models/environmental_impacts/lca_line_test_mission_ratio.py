# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCARatioTestFlightMission(om.ExplicitComponent):
    """
    Compute the ratio of fuel needed for the flight tests wrt to the fuel needed for the mission.
    This will serve to compute the impact of manufacturing we'll assume the emission profiles are
    similar.
    """

    def setup(self):
        self.add_input("data:mission:sizing:duration", units="h", val=np.nan)
        self.add_input(
            "data:environmental_impact:line_test:duration",
            units="h",
            val=np.nan,
            desc="Duration of line tests",
        )

        self.add_output(
            "data:environmental_impact:line_test:mission_ratio",
            val=5.0,
            desc="Ratio of fuel used during the line tests to fuel used during cruise",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:environmental_impact:line_test:mission_ratio"] = (
            inputs["data:environmental_impact:line_test:duration"]
            / inputs["data:mission:sizing:duration"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:environmental_impact:line_test:mission_ratio",
            "data:environmental_impact:line_test:duration",
        ] = 1.0 / inputs["data:mission:sizing:duration"]
        partials[
            "data:environmental_impact:line_test:mission_ratio",
            "data:mission:sizing:duration",
        ] = (
            -inputs["data:environmental_impact:line_test:duration"]
            / inputs["data:mission:sizing:duration"] ** 2.0
        )
