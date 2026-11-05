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
                "data:cost:operation:loan_principal",
                val=np.nan,
                units="USD",
                desc="The loan principal paid by the operator for the aircraft purchase.",
            )

        self.add_input(
            "data:cost:msp_per_unit",
            units="USD",
            val=np.nan,
            desc="The manufacturing suggested price of the aircraft.",
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
            shape=years_of_service,
        )
        self.add_input(
            "data:cost:operation:annual_revenue_projection",
            units="USD/yr",
            val=np.nan,
            shape=years_of_service,
        )

        self.add_output(
            name="data:cost:operation:annual_cash_flow",
            val=np.nan,
            units="USD",
            shape=years_of_service + 1,
        )

    def setup_partials(self):
        years_of_service = self.options["years_of_service"]
        loan = self.options["loan"]

        if loan:
            self.declare_partials(
                "data:cost:operation:annual_cash_flow",
                "data:cost:operation:loan_principal",
                rows=np.array([0]),
                cols=np.array([0]),
                method="exact",
                val=1.0,
            )

        self.declare_partials(
            "data:cost:operation:annual_cash_flow",
            "data:cost:msp_per_unit",
            rows=np.array([0]),
            cols=np.array([0]),
            method="exact",
            val=-1.0,
        )
        self.declare_partials(
            "data:cost:operation:annual_cash_flow",
            "data:cost:operation:annual_energy_cost_projection",
            rows=np.arange(1, years_of_service + 1),
            cols=np.arange(years_of_service),
            method="exact",
            val=-1.0,
        )
        self.declare_partials(
            "data:cost:operation:annual_cash_flow",
            "data:cost:operation:annual_revenue_projection",
            rows=np.arange(1, years_of_service + 1),
            cols=np.arange(years_of_service),
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

        if loan:
            first_payment = (
                inputs["data:cost:msp_per_unit"] - inputs["data:cost:operation:loan_principal"]
            )
        else:
            first_payment = inputs["data:cost:msp_per_unit"]

        recurring_part = (
            annual_revenue
            - annual_cost
            + annual_fuel_cost
            + annual_electricity_cost
            - annual_energy_cost_projection
        )

        outputs["data:cost:operation:annual_cash_flow"] = np.insert(
            recurring_part, 0, -first_payment
        )
