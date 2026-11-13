# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAnnualLoanCost(om.Group):
    """
    Computation of the yearly loan payment of standard mortgage formula from
    :cite:`gudmundsson:2013`. The 7% interest rate is applied typically with down payment of
    10-20% and a good credit record.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]

        self.add_subsystem(
            name="loan_payment",
            subsys=_LoanPayment(years_of_service=years_of_service),
            promotes=["*"],
        )
        self.add_subsystem(
            name="loan_payment_with_balloon",
            subsys=_LoanPaymentWithBalloon(years_of_service=years_of_service),
            promotes=["*"],
        )


class _LoanPayment(om.ExplicitComponent):
    """
    Computation of the yearly loan payments.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]

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

        self.add_output(
            "loan_payments_before_adding_balloon",
            val=15.0,
            units="USD/yr",
            shape=years_of_service,
        )

    def setup_partials(self):
        years_of_service = self.options["years_of_service"]

        self.declare_partials(
            "*",
            "*",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.zeros(years_of_service),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        years_of_service = self.options["years_of_service"]

        principal = inputs["data:cost:operation:loan_principal"]
        r_interest = inputs["data:cost:operation:annual_interest_rate"]

        outputs["loan_payments_before_adding_balloon"] = (
            principal * r_interest / (1.0 - (1.0 + r_interest) ** -years_of_service)
        ) * np.ones(years_of_service)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        years_of_service = self.options["years_of_service"]

        principal = inputs["data:cost:operation:loan_principal"]
        r_interest = inputs["data:cost:operation:annual_interest_rate"]

        partials["loan_payments_before_adding_balloon", "data:cost:operation:loan_principal"] = (
            r_interest / (1.0 - (1.0 + r_interest) ** -years_of_service)
        )

        partials[
            "loan_payments_before_adding_balloon", "data:cost:operation:annual_interest_rate"
        ] = (
            principal
            * (r_interest + 1.0) ** (years_of_service - 1.0)
            * (
                (r_interest + 1.0) ** (years_of_service + 1.0)
                - (years_of_service + 1.0) * r_interest
                - 1.0
            )
            / ((r_interest + 1.0) ** years_of_service - 1.0) ** 2.0
        )


class _LoanPaymentWithBalloon(om.ExplicitComponent):
    """
    Computation of the yearly loan payments with balloon payment.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]

        self.add_input(
            "loan_payments_before_adding_balloon",
            val=np.nan,
            units="USD/yr",
            shape=years_of_service,
        )
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
            shape=years_of_service,
        )

    def setup_partials(self):
        years_of_service = self.options["years_of_service"]

        self.declare_partials(
            "*",
            "loan_payments_before_adding_balloon",
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.arange(years_of_service),
        )
        self.declare_partials(
            "*",
            [
                "data:cost:operation:loan_principal",
                "data:cost:operation:annual_interest_rate",
                "data:cost:operation:pay_period",
            ],
            method="exact",
            rows=np.arange(years_of_service),
            cols=np.zeros(years_of_service),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        years_of_service = self.options["years_of_service"]

        principal = inputs["data:cost:operation:loan_principal"]
        r_interest = inputs["data:cost:operation:annual_interest_rate"]
        period = int(inputs["data:cost:operation:pay_period"][0])

        if period > years_of_service:
            raise ValueError("Pay period cannot be longer than the years of service.")

        balloon_payment = (
            principal
            * ((1.0 + r_interest) ** years_of_service - (1.0 + r_interest) ** period)
            / ((1.0 + r_interest) ** period - 1.0)
        )

        # Copy so the input vector is not modified in place
        loan_payments = inputs["loan_payments_before_adding_balloon"].copy()
        loan_payments[period:] = 0.0
        loan_payments[period - 1] += balloon_payment[0]

        outputs["data:cost:operation:annual_loan_cost"] = loan_payments

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        years_of_service = self.options["years_of_service"]

        principal = inputs["data:cost:operation:loan_principal"]
        r_interest = inputs["data:cost:operation:annual_interest_rate"]
        period = int(inputs["data:cost:operation:pay_period"][0])

        one_plus_r = 1.0 + r_interest
        a_n = one_plus_r**years_of_service
        a_p = one_plus_r**period

        # Indices 0 .. period + 1 are passed through, the rest are set to 0
        partials["data:cost:operation:annual_loan_cost", "loan_payments_before_adding_balloon"] = (
            np.concatenate((np.ones(period), np.zeros(years_of_service - period)))
        )

        # The balloon only appears at index period + 1. Assign into a zero array rather than
        # np.insert, which would make the array one element longer than the sparsity pattern.
        d_balloon_d_principal = np.zeros(years_of_service)
        d_balloon_d_principal[period - 1] = ((a_n - a_p) / (a_p - 1.0))[0]
        partials["data:cost:operation:annual_loan_cost", "data:cost:operation:loan_principal"] = (
            d_balloon_d_principal
        )

        d_balloon_d_interest = np.zeros(years_of_service)
        d_balloon_d_interest[period - 1] = (
            principal * (years_of_service * a_n - period * a_p) / one_plus_r / (a_p - 1.0)
            - principal * (a_n - a_p) * period * a_p / one_plus_r / (a_p - 1.0) ** 2.0
        )[0]
        partials[
            "data:cost:operation:annual_loan_cost", "data:cost:operation:annual_interest_rate"
        ] = d_balloon_d_interest

        # compute() truncates pay_period to an integer, so the outputs are piecewise constant in
        # pay_period and their exact derivative is 0 almost everywhere.
        partials["data:cost:operation:annual_loan_cost", "data:cost:operation:pay_period"] = (
            np.zeros(years_of_service)
        )
