# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCFlightMission(om.ExplicitComponent):
    """
    Computation of the aircraft flight mission per year.
    """

    def initialize(self):
        self.options.declare(
            name="use_operational_mission",
            default=False,
            types=bool,
            desc="The characteristics and consumption of the operational mission will be used",
        )

    def setup(self):
        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )

        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        self.add_input(
            name=duration_mission_name,
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
        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        outputs["data:cost:operation:mission_per_year"] = (
            inputs["data:TLAR:flight_hours_per_year"] / inputs[duration_mission_name]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        partials["data:cost:operation:mission_per_year", "data:TLAR:flight_hours_per_year"] = (
            1.0 / inputs[duration_mission_name]
        )

        partials["data:cost:operation:mission_per_year", duration_mission_name] = (
            -inputs["data:TLAR:flight_hours_per_year"] / inputs[duration_mission_name] ** 2.0
        )
