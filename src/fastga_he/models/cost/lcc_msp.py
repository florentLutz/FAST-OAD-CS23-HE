# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCMSP(om.ExplicitComponent):
    """
    Computation of the aircraft manufacturer suggested price (MSP) . The profit margin is
    set to 10% based on :cite:`marciello:2024`.
    """

    def setup(self):
        self.add_input(
            "data:cost:production_cost_per_unit",
            units="USD",
            val=np.nan,
        )

        self.add_output(
            "data:cost:msp_per_unit",
            val=1.0e5,
            units="USD",
            desc="Manufacturer suggested price of the aircraft",
        )
        self.declare_partials(
            of="data:cost:msp_per_unit", wrt="data:cost:production_cost_per_unit", val=1.11
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:msp_per_unit"] = 1.11 * inputs["data:cost:production_cost_per_unit"]
