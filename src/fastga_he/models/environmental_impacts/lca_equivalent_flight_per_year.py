# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCAEquivalentFlightsPerYear(om.ExplicitComponent):
    """
    The models have been implemented in a way were the input are the number of year the aircraft
    is expected to operate and the number of flights per year. In practice data rather give the
    time in hours an airframe is expected to live (airframe hours) and the average number of flight
    hours in a year.

    To avoid complicate rework of existing component, we'll simply compute an equivalent of the
    former based on the latter. The default value will be the average for a 1 engine turboprop AC
    as computed based on the data of the GA survey of the FAA.
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
            self.add_input("data:mission:sizing:duration", units="h", val=np.nan)
        else:
            self.add_input("data:mission:operational:duration", units="h", val=np.nan)

        self.add_output(
            name="data:TLAR:flight_per_year",
            val=100.0,
            desc="Average number of flight per year",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if not self.options["use_operational_mission"]:
            mission_duration = inputs["data:mission:sizing:duration"]
        else:
            mission_duration = inputs["data:mission:operational:duration"]

        outputs["data:TLAR:flight_per_year"] = (
            inputs["data:TLAR:flight_hours_per_year"] / mission_duration
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if not self.options["use_operational_mission"]:
            mission_duration_name = "data:mission:sizing:duration"
        else:
            mission_duration_name = "data:mission:operational:duration"

        partials[
            "data:TLAR:flight_per_year",
            "data:TLAR:flight_hours_per_year",
        ] = 1.0 / inputs[mission_duration_name]
        partials[
            "data:TLAR:flight_per_year",
            mission_duration_name,
        ] = -inputs["data:TLAR:flight_hours_per_year"] / inputs[mission_duration_name] ** 2.0
