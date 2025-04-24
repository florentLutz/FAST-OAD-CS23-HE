# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAnnualLoanCost(om.ExplicitComponent):
    """
    Computation of the yearly loan payment of standard mortgage formula from
    :cite:`gudmundsson:2013`. The 7% interest rate is applied typically with down payment of
    10-20% and a good credit record.
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
            "data:cost:operation:loan_principal",
            units="USD",
            val=np.nan,
        )

        self.add_input(
            "data:cost:operation:annual_interest_rate",
            val=0.07,
            desc="Annual interest rate of aircraft financing loan",
        )

        self.add_input(
            "data:cost:operation:pay_period",
            val=15.0,
            units="yr",
            desc="Number of years to pay back, normally ranges between 10-20 years",
        )

        self.add_output(
            "data:cost:operation:annual_loan_cost",
            val=15.0,
            units="USD/yr",
            desc="Annual loan cost of the aircraft",
        )
        if loan:
            self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        principal = inputs["data:cost:operation:loan_principal"]
        r_interest = inputs["data:cost:operation:annual_interest_rate"]
        period = inputs["data:cost:operation:pay_period"]
        loan = self.options["loan"]

        if loan:
            outputs["data:cost:operation:annual_loan_cost"] = (
                principal * r_interest / (1.0 - (1.0 + r_interest) ** -period)
            )

        else:
            outputs["data:cost:operation:annual_loan_cost"] = 0.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        principal = inputs["data:cost:operation:loan_principal"]
        r_interest = inputs["data:cost:operation:annual_interest_rate"]
        period = inputs["data:cost:operation:pay_period"]
        loan = self.options["loan"]

        if loan:
            partials[
                "data:cost:operation:annual_loan_cost", "data:cost:operation:loan_principal"
            ] = r_interest / (1.0 - (1.0 + r_interest) ** -period)

            partials[
                "data:cost:operation:annual_loan_cost", "data:cost:operation:annual_interest_rate"
            ] = (
                principal
                * (r_interest + 1.0) ** (period - 1.0)
                * ((r_interest + 1.0) ** (period + 1.0) - (period + 1.0) * r_interest - 1.0)
                / ((r_interest + 1.0) ** period - 1.0) ** 2.0
            )

            partials["data:cost:operation:annual_loan_cost", "data:cost:operation:pay_period"] = (
                -principal
                * r_interest
                * (r_interest + 1.0) ** period
                * np.log(r_interest + 1.0)
                / ((r_interest + 1.0) ** period - 1.0) ** 2.0
            )
