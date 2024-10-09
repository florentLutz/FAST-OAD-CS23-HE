# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCAAircraftPerFU(om.ExplicitComponent):
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
        self.add_input(
            name="data:TLAR:range",
            units="km",
            val=np.nan,
        )
        self.add_input("data:weight:aircraft:payload", val=np.nan, units="kg")

        self.add_output(
            "data:environmental_impact:aircraft_per_fu",
            val=1e-8,
            desc="Number of aircraft required to perform a functionnal unit, defined here as "
            "carrying 85kg over on km",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:environmental_impact:aircraft_per_fu"] = 1.0 / (
            inputs["data:TLAR:aircraft_lifespan"]
            * inputs["data:TLAR:flight_per_year"]
            * inputs["data:TLAR:range"]
            * inputs["data:weight:aircraft:payload"]
            / 85.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:environmental_impact:aircraft_per_fu", "data:TLAR:aircraft_lifespan"] = (
            -1.0
            / (
                inputs["data:TLAR:aircraft_lifespan"] ** 2.0
                * inputs["data:TLAR:flight_per_year"]
                * inputs["data:TLAR:range"]
                * inputs["data:weight:aircraft:payload"]
                / 85.0
            )
        )
        partials["data:environmental_impact:aircraft_per_fu", "data:TLAR:flight_per_year"] = (
            -1.0
            / (
                inputs["data:TLAR:aircraft_lifespan"]
                * inputs["data:TLAR:flight_per_year"] ** 2.0
                * inputs["data:TLAR:range"]
                * inputs["data:weight:aircraft:payload"]
                / 85.0
            )
        )
        partials["data:environmental_impact:aircraft_per_fu", "data:TLAR:range"] = -1.0 / (
            inputs["data:TLAR:aircraft_lifespan"]
            * inputs["data:TLAR:flight_per_year"]
            * inputs["data:TLAR:range"] ** 2.0
            * inputs["data:weight:aircraft:payload"]
            / 85.0
        )
        partials["data:environmental_impact:aircraft_per_fu", "data:weight:aircraft:payload"] = (
            -1.0
            / (
                inputs["data:TLAR:aircraft_lifespan"]
                * inputs["data:TLAR:flight_per_year"]
                * inputs["data:TLAR:range"]
                * inputs["data:weight:aircraft:payload"] ** 2.0
                / 85.0
            )
        )
