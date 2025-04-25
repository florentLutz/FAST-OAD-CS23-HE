# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


class LCCMaintenanceMiscellaneousCost(om.ExplicitComponent):
    """
    Compute the annual miscellaneous cost of the aircraft. The calculation is adjusted based on
    the cost rate from https://www.guardianjet.com/jet-aircraft-online-tools.
    """

    def setup(self):
        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_output(
            "data:cost:operation:miscellaneous_cost",
            val=200.0,
            units="USD/yr",
            desc="Annual airframe maintenance material cost per aircraft",
        )

        self.declare_partials(of="*", wrt="*", val=80.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ft_year = inputs["data:TLAR:flight_hours_per_year"]

        outputs["data:cost:operation:miscellaneous_cost"] = 80.0 * ft_year
