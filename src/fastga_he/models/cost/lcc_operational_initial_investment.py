# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCOperationalInitialInvestment(om.ExplicitComponent):
    """
    Computation of the initial investment of the aircraft for the NPV and PI calculations.
    """

    def initialize(self):
        self.options.declare(
            name="loan",
            default=True,
            types=bool,
            desc="True if loan is taken for financing the aircraft",
        )

    def setup(self):
        loan = self.options["loan"]
        self.add_input(
            "data:cost:msp_per_unit",
            units="USD",
            val=np.nan,
        )

        if loan:
            self.add_input(
                "data:cost:operation:loan_principal",
                units="USD",
                val=0.0,
            )

        self.add_input(
            "data:cost:operation:other_upfront_costs",
            units="USD",
            val=0.0,
            desc="Other upfront costs for the aircraft financing",
        )
        self.add_input(
            "data:cost:operation:other_financial_support",
            units="USD",
            val=0.0,
            desc="Other financial support for the aircraft financing, e.g. subsidies, etc.",
        )

        self.add_output(
            "data:cost:operation:initial_investment",
            val=1e5,
            units="USD",
            desc="Initial investment of the aircraft",
        )

    def setup_partials(self):
        loan = self.options["loan"]

        self.declare_partials(
            "*",
            ["data:cost:msp_per_unit", "data:cost:operation:other_upfront_costs"],
            val=1.0,
        )
        self.declare_partials("*", "data:cost:operation:other_financial_support", val=-1.0)

        if loan:
            self.declare_partials("*", "data:cost:operation:loan_principal", val=-1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        loan_principal = (
            inputs["data:cost:operation:loan_principal"] if self.options["loan"] else 0.0
        )

        outputs["data:cost:operation:initial_investment"] = (
            inputs["data:cost:msp_per_unit"]
            - inputs["data:cost:operation:other_financial_support"]
            + inputs["data:cost:operation:other_upfront_costs"]
            - loan_principal
        )
