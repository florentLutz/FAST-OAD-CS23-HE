# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


class LCCAnnualCrewCost(om.ExplicitComponent):
    """
    Computation of the annual flight crew cost in operation, annual salary obtained from
    https://bizjetjobs.com/pilot-salary-survey and
    https://simpleflying.com/private-jet-flight-attendant-salary/.
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

        self.add_output(
            "data:cost:operation:annual_crew_cost",
            val=1.5e5,
            units="USD/yr",
            desc="Annual flight crew cost of the aircraft",
        )
        self.declare_partials(
            "data:cost:operation:annual_crew_cost",
            "data:cost:operation:number_of_pilot",
            val=113556.7,
        )
        self.declare_partials(
            "data:cost:operation:annual_crew_cost",
            "data:cost:operation:number_of_cabin_crew",
            val=25200.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:annual_crew_cost"] = (
            113556.7 * inputs["data:cost:operation:number_of_pilot"]
            + 25200.0 * inputs["data:cost:operation:number_of_cabin_crew"]
        )
