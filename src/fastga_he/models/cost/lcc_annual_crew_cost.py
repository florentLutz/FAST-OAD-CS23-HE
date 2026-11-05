# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


class LCCAnnualCrewCost(om.ExplicitComponent):
    """
    Computation of the annual flight crew cost in operation, annual salary obtained from
    https://bizjetjobs.com/pilot-salary-survey and
    https://simpleflying.com/private-jet-flight-attendant-salary/. The default social benefits rate
    is an average between western-european countries.
    """

    def setup(self):
        self.add_input(
            "data:cost:operation:number_of_pilot",
            val=0.0,
        )
        self.add_input(
            "data:cost:operation:number_of_cabin_crew",
            val=0.0,
        )
        self.add_input(
            "data:cost:operation:social_benefits_rate",
            val=0.3,
            desc="Social benefits rate for the flight crew",
        )

        self.add_output(
            "data:cost:operation:annual_crew_cost",
            val=1.5e5,
            units="USD/yr",
            desc="Annual flight crew cost of the aircraft",
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_pilot = inputs["data:cost:operation:number_of_pilot"]
        number_of_cabin_crew = inputs["data:cost:operation:number_of_cabin_crew"]
        social_benefits_rate = inputs["data:cost:operation:social_benefits_rate"]

        outputs["data:cost:operation:annual_crew_cost"] = (
            113556.7 * number_of_pilot + 25200.0 * number_of_cabin_crew
        ) * (1.0 + social_benefits_rate)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_pilot = inputs["data:cost:operation:number_of_pilot"]
        number_of_cabin_crew = inputs["data:cost:operation:number_of_cabin_crew"]
        social_benefits_rate = inputs["data:cost:operation:social_benefits_rate"]

        partials["data:cost:operation:annual_crew_cost", "data:cost:operation:number_of_pilot"] = (
            113556.7 * (1.0 + social_benefits_rate)
        )

        partials[
            "data:cost:operation:annual_crew_cost", "data:cost:operation:number_of_cabin_crew"
        ] = 25200.0 * (1.0 + social_benefits_rate)

        partials[
            "data:cost:operation:annual_crew_cost", "data:cost:operation:social_benefits_rate"
        ] = 113556.7 * number_of_pilot + 25200.0 * number_of_cabin_crew
