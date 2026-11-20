# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCOperationalPassengerPerFlight(om.ExplicitComponent):
    """
    Computation of the number of passenger carried per flight mission.
    """

    def initialize(self):
        self.options.declare(
            name="trainer",
            default=False,
            types=bool,
            desc="True if the aircraft is used as a trainer",
        )

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
        trainer = self.options["trainer"]

        npax_design = inputs["data:TLAR:NPAX_design"]
        passenger_per_flight_baseline = inputs["data:cost:operation:passenger_per_flight_baseline"]

        if trainer or npax_design < 1:
            passenger_per_flight = 1
        elif npax_design >= passenger_per_flight_baseline:
            passenger_per_flight = passenger_per_flight_baseline
        else:
            passenger_per_flight = npax_design

        outputs["data:cost:operation:passenger_per_flight"] = passenger_per_flight

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        trainer = self.options["trainer"]

        npax_design = inputs["data:TLAR:NPAX_design"]
        passenger_per_flight_baseline = inputs["data:cost:operation:passenger_per_flight_baseline"]

        if trainer or npax_design < 1:
            d_pax_per_flight_d_npax_design = 0.0
            d_pax_per_flight_d_npax_baseline = 0.0
        elif npax_design >= passenger_per_flight_baseline:
            d_pax_per_flight_d_npax_design = 0.0
            d_pax_per_flight_d_npax_baseline = 1.0
        else:
            d_pax_per_flight_d_npax_design = 1.0
            d_pax_per_flight_d_npax_baseline = 0.0

        partials["data:cost:operation:passenger_per_flight", "data:TLAR:NPAX_design"] = (
            d_pax_per_flight_d_npax_design
        )

        partials[
            "data:cost:operation:passenger_per_flight",
            "data:cost:operation:passenger_per_flight_baseline",
        ] = d_pax_per_flight_d_npax_baseline
