# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCGrossProfitPerUnit(om.ExplicitComponent):
    """
    Computation of the aircraft manufacturer unit sales gross profit.
    """

    def setup(self):
        self.add_input(
            "data:cost:msp_per_unit",
            units="USD",
            val=np.nan,
            desc="Manufacturer suggested price of the aircraft",
        )
        self.add_input("data:cost:recursive_cost_per_unit", units="USD", val=np.nan)

        self.add_output(
            "data:cost:unit_gross_profit",
            val=1.0e5,
            units="USD",
            desc="Gross profit per unit sales of the aircraft manufacturer",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:unit_gross_profit"] = (
            inputs["data:cost:msp_per_unit"] - inputs["data:cost:recursive_cost_per_unit"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:cost:unit_gross_profit", "data:cost:msp_per_unit"] = 1.0

        partials["data:cost:unit_gross_profit", "data:cost:recursive_cost_per_unit"] = -1.0
