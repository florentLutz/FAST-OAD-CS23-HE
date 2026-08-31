# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAnnualSalesGrossProfit(om.ExplicitComponent):
    """
    Computation of the annual aircraft manufacturer sales gross profit.
    """

    def initialize(self):
        self.options.declare(
            "years_of_development",
            types=int,
            default=10,
            desc="The number of years of development for the aircraft, before the first delivery.",
        )
        self.options.declare(
            "years_of_program",
            types=int,
            default=30,
            desc="The total number of years of the aircraft program, including development and production.",
        )

    def setup(self):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        self.add_input(
            "data:cost:msp_per_unit",
            units="USD",
            val=np.nan,
            desc="Manufacturer suggested price of the aircraft",
        )
        self.add_input("data:cost:production:recursive_cost_per_unit", units="USD", val=np.nan)
        self.add_input(
            "data:cost:production:maturity_discount",
            val=np.nan,
            desc="The discount factor in manufacturing and tooling bsed on process maturity",
            shape=years_of_program - years_of_development + 1,
        )
        self.add_input(
            "data:cost:production:annual_delivery_count",
            val=np.nan,
            shape=years_of_program - years_of_development + 1,
            desc="The annual delivery count of aircraft over the program duration.",
        )

        self.add_output(
            "data:cost:production:annual_gross_profit",
            val=1.0e5,
            units="USD",
            desc="Gross profit per unit sales of the aircraft manufacturer",
            shape=years_of_program - years_of_development + 1,
        )

    def setup_partials(self):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        self.declare_partials(
            of="*",
            wrt=[
                "data:cost:production:annual_delivery_count",
                "data:cost:production:maturity_discount",
            ],
            method="exact",
            rows=np.arange(years_of_program - years_of_development + 1),
            cols=np.arange(years_of_program - years_of_development + 1),
        )
        self.declare_partials(
            of="*",
            wrt=[
                "data:cost:msp_per_unit",
                "data:cost:production:recursive_cost_per_unit",
            ],
            method="exact",
            rows=np.arange(years_of_program - years_of_development + 1),
            cols=np.zeros(years_of_program - years_of_development + 1),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aircraft_msp = inputs["data:cost:msp_per_unit"]
        aircraft_cost = inputs["data:cost:production:recursive_cost_per_unit"]
        annual_delivery_count = inputs["data:cost:production:annual_delivery_count"]
        maturity_discount = inputs["data:cost:production:maturity_discount"]

        outputs["data:cost:production:annual_gross_profit"] = (
            aircraft_msp - aircraft_cost * maturity_discount
        ) * annual_delivery_count

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aircraft_msp = inputs["data:cost:msp_per_unit"]
        aircraft_cost = inputs["data:cost:production:recursive_cost_per_unit"]
        annual_delivery_count = inputs["data:cost:production:annual_delivery_count"]
        maturity_discount = inputs["data:cost:production:maturity_discount"]

        partials["data:cost:production:annual_gross_profit", "data:cost:msp_per_unit"] = (
            annual_delivery_count
        )

        partials[
            "data:cost:production:annual_gross_profit",
            "data:cost:production:recursive_cost_per_unit",
        ] = -annual_delivery_count * maturity_discount

        partials[
            "data:cost:production:annual_gross_profit", "data:cost:production:maturity_discount"
        ] = -annual_delivery_count * aircraft_cost

        partials[
            "data:cost:production:annual_gross_profit", "data:cost:production:annual_delivery_count"
        ] = aircraft_msp - aircraft_cost * maturity_discount
