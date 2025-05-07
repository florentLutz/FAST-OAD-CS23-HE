# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCFlightMission(om.ExplicitComponent):
    """
    Computation of the aircraft flight mission per year.
    """

    def setup(self):
        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_input(
            name="data:mission:sizing:duration",
            units="h",
            val=np.nan,
        )

        self.add_output(
            "data:cost:operation:mission_per_year",
            val=100.0,
            units="1/yr",
            desc="Flight mission per year",
        )
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:mission_per_year"] = (
            inputs["data:TLAR:flight_hours_per_year"] / inputs["data:mission:sizing:duration"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:cost:operation:mission_per_year", "data:TLAR:flight_hours_per_year"] = (
            1.0 / inputs["data:mission:sizing:duration"]
        )

        partials["data:cost:operation:mission_per_year", "data:mission:sizing:duration"] = (
            -inputs["data:TLAR:flight_hours_per_year"]
            / inputs["data:mission:sizing:duration"] ** 2.0
        )
