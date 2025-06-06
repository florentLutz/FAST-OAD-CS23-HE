# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCMaintenanceCost(om.ExplicitComponent):
    """
    Computation of the annual maintenance cost of the aircraft. The calculation is adjusted
    based on the cost rate from https://www.guardianjet.com/jet-aircraft-online-tools and
    https://planephd.com/wizard.
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
        owe_clipped = np.clip(inputs["data:weight:aircraft:OWE"], 500.0, None)

        outputs["data:cost:operation:maintenance_cost"] = inputs[
            "data:TLAR:flight_hours_per_year"
        ] * (-49.2 + 0.147 * owe_clipped)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        owe_clipped = np.clip(inputs["data:weight:aircraft:OWE"], 500.0, None)

        partials["data:cost:operation:maintenance_cost", "data:weight:aircraft:OWE"] = np.where(
            inputs["data:weight:aircraft:OWE"] == owe_clipped,
            0.147 * inputs["data:TLAR:flight_hours_per_year"],
            1e-6,
        )

        partials["data:cost:operation:maintenance_cost", "data:TLAR:flight_hours_per_year"] = (
            -49.2 + 0.147 * owe_clipped
        )
