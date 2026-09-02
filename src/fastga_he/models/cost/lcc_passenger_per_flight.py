# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCOperationalPassengerPerFlight(om.ExplicitComponent):
    """
    Computation of the number of passenger carried per flight mission.
    """

    def setup(self):
        self.add_input(
            "data:TLAR:NPAX_design",
            val=np.nan,
            desc="Number of passengers for the sizing mission",
        )
        self.add_input(
            "data:cost:operation:passenger_per_flight_baseline",
            val=np.nan,
            desc="The passenger per flight mission for the baseline aircraft, computed with "
            "passenger load factor of 0.7 and round down to the nearest integer.",
        )

        self.add_output("data:cost:operation:passenger_per_flight", val=4)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        npax_design = inputs["data:TLAR:NPAX_design"]
        passenger_per_flight_baseline = inputs["data:cost:operation:passenger_per_flight_baseline"]

        outputs["data:cost:operation:passenger_per_flight"] = (
            passenger_per_flight_baseline
            if npax_design >= passenger_per_flight_baseline
            else npax_design
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        npax_design = inputs["data:TLAR:NPAX_design"]
        passenger_per_flight_baseline = inputs["data:cost:operation:passenger_per_flight_baseline"]

        partials["data:cost:operation:passenger_per_flight", "data:TLAR:NPAX_design"] = (
            1.0 if npax_design < passenger_per_flight_baseline else 0.0
        )

        partials[
            "data:cost:operation:passenger_per_flight",
            "data:cost:operation:passenger_per_flight_baseline",
        ] = 1.0 if npax_design >= passenger_per_flight_baseline else 0.0
