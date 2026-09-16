# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCMSP(om.ExplicitComponent):
    """
    Computation of the aircraft manufacturer suggested price (MSP). The profit margin is
    set to 10% based on :cite:`marciello:2024`. A 35% gross margin is assumed based on
    :cite:`trainelli:2020`. The spare parts factor of 1.1 is used in :cite:`trainelli:2020` .
    """

    def setup(self):
        self.add_input(
            "data:cost:production_cost_per_unit",
            units="USD",
            val=np.nan,
        )
        self.add_input(
            "data:cost:production:gross_margin",
            val=0.35,
        )
        self.add_input(
            "data:cost:production:spare_parts_factor",
            val=1.1,
        )

        self.add_output(
            "data:cost:msp_per_unit",
            val=1.0e5,
            units="USD",
            desc="Manufacturer suggested price of the aircraft",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:msp_per_unit"] = (
            (1.0 + inputs["data:cost:production:gross_margin"])
            * inputs["data:cost:production_cost_per_unit"]
            * inputs["data:cost:production:spare_parts_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:cost:msp_per_unit", "data:cost:production_cost_per_unit"] = (
            1.0 + inputs["data:cost:production:gross_margin"]
        ) * inputs["data:cost:production:spare_parts_factor"]

        partials["data:cost:msp_per_unit", "data:cost:production:gross_margin"] = (
            inputs["data:cost:production_cost_per_unit"]
            * inputs["data:cost:production:spare_parts_factor"]
        )

        partials["data:cost:msp_per_unit", "data:cost:production:spare_parts_factor"] = (
            1.0 + inputs["data:cost:production:gross_margin"]
        ) * inputs["data:cost:production_cost_per_unit"]
