# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCAAircraftPerFU(om.ExplicitComponent):
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
            "data:environmental_impact:aircraft_per_fu",
            val=1e-8,
            desc="Number of aircraft required to perform a functional unit, defined here as "
            "carrying 85kg over one km",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if not self.options["use_operational_mission"]:
            range_mission = inputs["data:TLAR:range"]
            payload = inputs["data:weight:aircraft:payload"]
        else:
            range_mission = inputs["data:mission:operational:range"]
            payload = inputs["data:mission:operational:payload:mass"]

        outputs["data:environmental_impact:aircraft_per_fu"] = 1.0 / (
            inputs["data:TLAR:aircraft_lifespan"]
            * inputs["data:TLAR:flight_per_year"]
            * range_mission
            * payload
            / 85.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if not self.options["use_operational_mission"]:
            range_mission_name = "data:TLAR:range"
            payload_name = "data:weight:aircraft:payload"

        else:
            range_mission_name = "data:mission:operational:range"
            payload_name = "data:mission:operational:payload:mass"

        range_mission = inputs[range_mission_name]
        payload = inputs[payload_name]

        partials["data:environmental_impact:aircraft_per_fu", "data:TLAR:aircraft_lifespan"] = (
            -1.0
            / (
                inputs["data:TLAR:aircraft_lifespan"] ** 2.0
                * inputs["data:TLAR:flight_per_year"]
                * range_mission
                * payload
                / 85.0
            )
        )
        partials["data:environmental_impact:aircraft_per_fu", "data:TLAR:flight_per_year"] = (
            -1.0
            / (
                inputs["data:TLAR:aircraft_lifespan"]
                * inputs["data:TLAR:flight_per_year"] ** 2.0
                * range_mission
                * payload
                / 85.0
            )
        )
        partials["data:environmental_impact:aircraft_per_fu", range_mission_name] = -1.0 / (
            inputs["data:TLAR:aircraft_lifespan"]
            * inputs["data:TLAR:flight_per_year"]
            * range_mission**2.0
            * payload
            / 85.0
        )
        partials["data:environmental_impact:aircraft_per_fu", payload_name] = -1.0 / (
            inputs["data:TLAR:aircraft_lifespan"]
            * inputs["data:TLAR:flight_per_year"]
            * range_mission
            * payload**2.0
            / 85.0
        )


class LCAAircraftPerFUFlightHours(om.ExplicitComponent):
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
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        self.add_input(
            name=duration_mission_name,
            units="h",
            val=np.nan,
        )

        self.add_output(
            "data:environmental_impact:aircraft_per_fu",
            val=1e-8,
            desc="Number of aircraft required to perform a functional unit, defined here as "
            "flying one hour",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        outputs["data:environmental_impact:aircraft_per_fu"] = 1.0 / (
            inputs["data:TLAR:aircraft_lifespan"]
            * inputs["data:TLAR:flight_per_year"]
            * inputs[duration_mission_name]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        if not self.options["use_operational_mission"]:
            duration_mission_name = "data:mission:sizing:duration"

        else:
            duration_mission_name = "data:mission:operational:duration"

        duration_mission = inputs[duration_mission_name]

        partials["data:environmental_impact:aircraft_per_fu", "data:TLAR:aircraft_lifespan"] = (
            -1.0
            / (
                inputs["data:TLAR:aircraft_lifespan"] ** 2.0
                * inputs["data:TLAR:flight_per_year"]
                * duration_mission
            )
        )
        partials["data:environmental_impact:aircraft_per_fu", "data:TLAR:flight_per_year"] = (
            -1.0
            / (
                inputs["data:TLAR:aircraft_lifespan"]
                * inputs["data:TLAR:flight_per_year"] ** 2.0
                * duration_mission
            )
        )
        partials["data:environmental_impact:aircraft_per_fu", duration_mission_name] = -1.0 / (
            inputs["data:TLAR:aircraft_lifespan"]
            * inputs["data:TLAR:flight_per_year"]
            * duration_mission ** 2.0
        )
