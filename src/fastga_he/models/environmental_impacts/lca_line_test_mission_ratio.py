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

    def initialize(self):
        self.options.declare(
            name="use_operational_mission",
            default=False,
            types=bool,
            desc="The characteristics and consumption of the operational mission will be used",
        )

    def setup(self):
        if not self.options["use_operational_mission"]:
            self.add_input("data:mission:sizing:duration", units="h", val=np.nan)
        else:
            self.add_input("data:mission:operational:duration", units="h", val=np.nan)

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
        if not self.options["use_operational_mission"]:
            mission_duration = inputs["data:mission:sizing:duration"]
        else:
            mission_duration = inputs["data:mission:operational:duration"]

        outputs["data:environmental_impact:line_test:mission_ratio"] = (
            inputs["data:environmental_impact:line_test:duration"] / mission_duration
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if not self.options["use_operational_mission"]:
            mission_duration_name = "data:mission:sizing:duration"
        else:
            mission_duration_name = "data:mission:operational:duration"

        partials[
            "data:environmental_impact:line_test:mission_ratio",
            "data:environmental_impact:line_test:duration",
        ] = 1.0 / inputs[mission_duration_name]
        partials[
            "data:environmental_impact:line_test:mission_ratio",
            mission_duration_name,
        ] = (
            -inputs["data:environmental_impact:line_test:duration"]
            / inputs[mission_duration_name] ** 2.0
        )
