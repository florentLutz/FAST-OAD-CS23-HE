# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCOperationAnnualCashFlow(om.ExplicitComponent):
    """
    Computation of the annual cash flow of the aircraft operator. This includes the energy pricing
    correction with compound average growth rate over the aircraft's service life.
    """

    def initialize(self):
        self.options.declare(
            "years_of_service",
            types=int,
            default=30,
            desc="The total service life of the aircraft in years for NPV calculation.",
        )
        self.options.declare(
            name="loan",
            default=True,
            types=bool,
            desc="True if loan is taken for financing the aircraft",
        )

    def setup(self):
        years_of_service = self.options["years_of_service"]
        loan = self.options["loan"]

        if loan:
            self.add_input(
                "data:cost:operation:annual_loan_cost",
                val=np.nan,
                units="USD/yr",
                desc="Annual loan cost for the aircraft",
                shape=years_of_service + 1,
            )

        self.add_input(
            "data:cost:operation:initial_investment",
            val=np.nan,
            units="USD",
            desc="Initial investment of the aircraft",
        )
        self.add_input(
            "data:cost:operation:annual_cost_per_unit",
            val=np.nan,
            units="USD/yr",
            desc="Annual operational cost per unit of the aircraft",
        )
        self.add_input(
            name="data:cost:operation:annual_fuel_cost",
            val=0.0,
            units="USD/yr",
        )
        self.add_input(
            name="data:cost:operation:annual_electricity_cost",
            val=0.0,
            units="USD/yr",
        )
        self.add_input(
            name="data:cost:operation:annual_energy_cost_projection",
            val=np.nan,
            units="USD/yr",
            shape=years_of_service + 1,
        )
        self.add_input(
            "data:cost:operation:annual_revenue_projection",
            units="USD/yr",
            val=np.nan,
            shape=years_of_service + 1,
        )

        self.add_output(
            name="data:cost:operation:annual_cash_flow",
            val=np.nan,
            units="USD",
            shape=years_of_service + 1,
        )

    def setup_partials(self):
        # Year 0 (row 0) depends only on the initial investment; recurring terms start at year 1
        years_of_service = self.options["years_of_service"]
        loan = self.options["loan"]

        annual_cost_variables = (
            [
                "data:cost:operation:annual_energy_cost_projection",
                "data:cost:operation:annual_loan_cost",
            ]
            if loan
            else "data:cost:operation:annual_energy_cost_projection"
        )

        self.declare_partials(
            "data:cost:operation:annual_cash_flow",
            "data:cost:operation:initial_investment",
            rows=np.array([0]),
            cols=np.array([0]),
            method="exact",
            val=-1.0,
        )
        self.declare_partials(
            "data:cost:operation:annual_cash_flow",
            annual_cost_variables,
            rows=np.arange(1, years_of_service + 1),
            cols=np.arange(1, years_of_service + 1),
            method="exact",
            val=-1.0,
        )
        self.declare_partials(
            "data:cost:operation:annual_cash_flow",
            "data:cost:operation:annual_revenue_projection",
            rows=np.arange(1, years_of_service + 1),
            cols=np.arange(1, years_of_service + 1),
            method="exact",
            val=1.0,
        )
        self.declare_partials(
            "data:cost:operation:annual_cash_flow",
            "data:cost:operation:annual_cost_per_unit",
            rows=np.arange(1, years_of_service + 1),
            cols=np.zeros(years_of_service),
            method="exact",
            val=-1.0,
        )
        self.declare_partials(
            "data:cost:operation:annual_cash_flow",
            [
                "data:cost:operation:annual_fuel_cost",
                "data:cost:operation:annual_electricity_cost",
            ],
            rows=np.arange(1, years_of_service + 1),
            cols=np.zeros(years_of_service),
            method="exact",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        loan = self.options["loan"]
        annual_cost = inputs["data:cost:operation:annual_cost_per_unit"]
        annual_fuel_cost = inputs["data:cost:operation:annual_fuel_cost"]
        annual_electricity_cost = inputs["data:cost:operation:annual_electricity_cost"]
        annual_energy_cost_projection = inputs["data:cost:operation:annual_energy_cost_projection"]
        annual_revenue = inputs["data:cost:operation:annual_revenue_projection"]
        annual_loan_cost = inputs["data:cost:operation:annual_loan_cost"] if loan else 0.0

        first_payment = inputs["data:cost:operation:initial_investment"]

        recurring_part = (
            annual_revenue
            - annual_cost
            + annual_fuel_cost
            + annual_electricity_cost
            - annual_energy_cost_projection
            - annual_loan_cost
        )

        # Year 0 only holds the initial investment, so recurring terms have no effect on it
        recurring_part[0] = -first_payment[0]

        outputs["data:cost:operation:annual_cash_flow"] = recurring_part
