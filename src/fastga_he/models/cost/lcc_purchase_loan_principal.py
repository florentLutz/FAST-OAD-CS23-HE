# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCPurchaseLoanPrincipal(om.ExplicitComponent):
    """
    Computation of the purchase loan principal of the aircraft operator. The loan principal is
    derived from the manufacturer suggested price of the aircraft and the loan-to-value ratio based
    on the rate approved by bank, typically between 80%-90%.
    """

    def setup(self):
        self.add_input(
            "data:cost:msp_per_unit",
            units="USD",
            val=np.nan,
        )
        self.add_input(
            "data:cost:operation:loan_to_value_ratio",
            val=0.85,
            desc="The loan-to-value ratio based on the rate approved by bank.",
        )

        self.add_output(
            "data:cost:operation:loan_principal",
            units="USD",
            val=np.nan,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:operation:loan_principal"] = (
            inputs["data:cost:msp_per_unit"] * inputs["data:cost:operation:loan_to_value_ratio"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:cost:operation:loan_principal", "data:cost:msp_per_unit"] = inputs[
            "data:cost:operation:loan_to_value_ratio"
        ]

        partials[
            "data:cost:operation:loan_principal", "data:cost:operation:loan_to_value_ratio"
        ] = inputs["data:cost:msp_per_unit"]
