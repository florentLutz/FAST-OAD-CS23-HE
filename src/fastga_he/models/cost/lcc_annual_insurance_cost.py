# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAnnualInsuranceCost(om.ExplicitComponent):
    """
    Computation of the annual insurance cost in operation :cite:`gudmundsson:2013`. Percentage
    adjusted with cost provides from https://sunsetais.com/aircraft-hull-liability/.
    """

    def setup(self):
        self.add_input(
            "data:cost:msp_per_unit",
            units="USD",
            val=np.nan,
        )

        self.add_output(
            "data:cost:operation:annual_insurance_cost",
            val=1500.0,
            units="USD/yr",
            desc="Annual insurance cost of the aircraft",
        )

        self.declare_partials(
            of="data:cost:operation:annual_insurance_cost", wrt="data:cost:msp_per_unit", val=0.01
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:annual_insurance_cost"] = (
            500 + 0.01 * inputs["data:cost:msp_per_unit"]
        )
