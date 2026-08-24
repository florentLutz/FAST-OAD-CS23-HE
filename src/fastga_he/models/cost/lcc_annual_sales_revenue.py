# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAnnualSalesRevenue(om.ExplicitComponent):
    """
    Computation of the aircraft manufacturer annual sales revenue.
    """

    def setup(self):
        self.add_input(
            "data:cost:msp_per_unit",
            units="USD",
            val=np.nan,
            desc="Manufacturer suggested price of the aircraft",
        )
        self.add_input(
            "data:cost:production:number_aircraft_5_years",
            val=np.nan,
            desc="Number of planned aircraft to be produced over a 5-year period or 60 months",
        )

        self.add_output(
            "data:cost:annual_sales_revenue",
            val=1.0e5,
            units="USD",
            desc="Manufacturer suggested price of the aircraft",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:annual_sales_revenue"] = (
            inputs["data:cost:msp_per_unit"]
            * inputs["data:cost:production:number_aircraft_5_years"]
            / 5.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:cost:annual_sales_revenue", "data:cost:msp_per_unit"] = (
            inputs["data:cost:production:number_aircraft_5_years"] / 5.0
        )

        partials[
            "data:cost:annual_sales_revenue", "data:cost:production:number_aircraft_5_years"
        ] = inputs["data:cost:msp_per_unit"] / 5.0
