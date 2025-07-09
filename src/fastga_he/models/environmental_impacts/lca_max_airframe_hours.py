#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCAEquivalentMaxAirframeHours(om.ExplicitComponent):
    """
    The models have been implemented in a way were the input are the number of year the aircraft
    is expected to operate and the number of flights per year. Some models however, required the
    expected number of hours the aircraft will live.

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
            name="data:TLAR:aircraft_lifespan",
            val=np.nan,
            units="yr",
            desc="Expected lifetime of the aircraft",
        )
        self.add_input(
            name="data:TLAR:flight_per_year",
            val=np.nan,
            desc="Average number of flight per year",
        )

        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:main_route:duration"

        else:
            duration_mission_name = "data:mission:operational:main_route:duration"

        self.add_input(
            name=duration_mission_name,
            units="h",
            val=np.nan,
        )

        self.add_output(
            name="data:TLAR:max_airframe_hours",
            val=3524.9,
            units="h",
            desc="Expected lifetime of the aircraft expressed in airframe hours",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:main_route:duration"
        else:
            duration_mission_name = "data:mission:operational:main_route:duration"

        outputs["data:TLAR:max_airframe_hours"] = (
            inputs["data:TLAR:aircraft_lifespan"]
            * inputs["data:TLAR:flight_per_year"]
            * inputs[duration_mission_name]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:main_route:duration"
        else:
            duration_mission_name = "data:mission:operational:main_route:duration"

        partials["data:TLAR:max_airframe_hours", "data:TLAR:aircraft_lifespan"] = (
            inputs["data:TLAR:flight_per_year"] * inputs[duration_mission_name]
        )
        partials["data:TLAR:max_airframe_hours", "data:TLAR:flight_per_year"] = (
            inputs["data:TLAR:aircraft_lifespan"] * inputs[duration_mission_name]
        )
        partials["data:TLAR:max_airframe_hours", duration_mission_name] = (
            inputs["data:TLAR:flight_per_year"] * inputs["data:TLAR:aircraft_lifespan"]
        )
