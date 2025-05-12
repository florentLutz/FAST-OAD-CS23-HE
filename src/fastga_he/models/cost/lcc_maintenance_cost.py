# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCMaintenanceCost(om.ExplicitComponent):
    """
    Computation of  the annual maintenance cost of the aircraft. The calculation is adjusted
    based on the cost rate from https://www.guardianjet.com/jet-aircraft-online-tools.
    """

    def setup(self):
        self.add_input(
            "data:weight:aircraft:OWE",
            units="kg",
            val=np.nan,
        )

        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_output(
            "data:cost:operation:maintenance_cost",
            val=2.0e4,
            units="USD/yr",
            desc="Annual maintenance cost per aircraft",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        flight_hour = inputs["data:TLAR:flight_hours_per_year"]
        owe = inputs["data:weight:aircraft:OWE"]

        outputs["data:cost:operation:maintenance_cost"] = flight_hour * (
            331.0 - 0.072 * owe + 2.75e-5 * owe**2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        flight_hour = inputs["data:TLAR:flight_hours_per_year"]
        owe = inputs["data:weight:aircraft:OWE"]

        partials["data:cost:operation:maintenance_cost", "data:weight:aircraft:OWE"] = (
            flight_hour * (5.5e-5 * owe - 0.072)
        )

        partials["data:cost:operation:maintenance_cost", "data:TLAR:flight_hours_per_year"] = (
            331.0 - 0.072 * owe + 2.75e-5 * owe**2.0
        )
