# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCOperationalAnnualRevenue(om.ExplicitComponent):
    """
    Computation of the annual revenue of the aircraft. The profit margin is derived from the
    operating margin in Industry Statistics of IATA :cite:`iata:2025`.
    """

    def setup(self):
        self.add_input(
            "data:cost:operation:annual_cost_per_unit",
            val=np.nan,
            units="USD/yr",
            desc="Annual operational cost per unit of the aircraft",
        )
        self.add_input(
            "data:cost:operation:profit_margin",
            val=0.07,
            desc="Profit margin as a fraction of the annual revenue",
        )

        self.add_output("data:cost:operation:annual_revenue_per_unit", units="USD/yr", val=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:annual_revenue_per_unit"] = inputs[
            "data:cost:operation:annual_cost_per_unit"
        ] / (1.0 - inputs["data:cost:operation:profit_margin"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:cost:operation:annual_revenue_per_unit",
            "data:cost:operation:annual_cost_per_unit",
        ] = 1.0 / (1.0 - inputs["data:cost:operation:profit_margin"])

        partials[
            "data:cost:operation:annual_revenue_per_unit", "data:cost:operation:profit_margin"
        ] = inputs["data:cost:operation:annual_cost_per_unit"] / (
            (1.0 - inputs["data:cost:operation:profit_margin"]) ** 2.0
        )
