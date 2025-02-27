# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


# TODO: Rename class
class LCAUseFlightPerFU(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="use_operational_mission",
            default=False,
            types=bool,
            desc="The characteristics and consumption of the operational mission will be used",
        )

    def setup(self):
        if not self.options["use_operational_mission"]:
            range_mission_name = "data:TLAR:range"
            payload_name = "data:weight:aircraft:payload"

        else:
            range_mission_name = "data:mission:operational:range"
            payload_name = "data:mission:operational:payload:mass"

        self.add_input(
            name=range_mission_name,
            units="km",
            val=np.nan,
        )
        self.add_input(payload_name, val=np.nan, units="kg")

        self.add_output(
            "data:environmental_impact:flight_per_fu",
            val=1e-8,
            desc="Number of flight required to perform a functionnal unit, defined here as "
            "carrying 85kg over one km",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if not self.options["use_operational_mission"]:
            range_mission_name = "data:TLAR:range"
            payload_name = "data:weight:aircraft:payload"

        else:
            range_mission_name = "data:mission:operational:range"
            payload_name = "data:mission:operational:payload:mass"

        outputs["data:environmental_impact:flight_per_fu"] = 1.0 / (
            inputs[range_mission_name] * inputs[payload_name] / 85.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if not self.options["use_operational_mission"]:
            range_mission_name = "data:TLAR:range"
            payload_name = "data:weight:aircraft:payload"

        else:
            range_mission_name = "data:mission:operational:range"
            payload_name = "data:mission:operational:payload:mass"

        partials["data:environmental_impact:flight_per_fu", range_mission_name] = -1.0 / (
            inputs[range_mission_name] ** 2.0 * inputs[payload_name] / 85.0
        )
        partials["data:environmental_impact:flight_per_fu", payload_name] = -1.0 / (
            inputs[range_mission_name] * inputs[payload_name] ** 2.0 / 85.0
        )


class LCAUseFlightPerFUFlightHours(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="use_operational_mission",
            default=False,
            types=bool,
            desc="The characteristics and consumption of the operational mission will be used",
        )

    def setup(self):
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
            "data:environmental_impact:flight_per_fu",
            val=1e-8,
            desc="Number of flight required to perform a functionnal unit, defined here as "
            "flying one hour",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        outputs["data:environmental_impact:flight_per_fu"] = 1.0 / inputs[duration_mission_name]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        partials["data:environmental_impact:flight_per_fu", duration_mission_name] = -1.0 / (
            inputs[duration_mission_name] ** 2.0
        )
