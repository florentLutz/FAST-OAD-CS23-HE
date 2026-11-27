# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCProductionAnnualCashFlow(om.ExplicitComponent):
    """
    Computation of the annual aircraft manufacturer free cash flow.
    """

    def initialize(self):
        self.options.declare(
            "years_of_program",
            types=int,
            default=30,
            desc="The total number of years of the aircraft program, including development and production.",
        )

    def setup(self):
        years_of_program = self.options["years_of_program"]

        self.add_input(
            "data:cost:production:project_development_cost_distribution",
            units="USD",
            val=np.nan,
            shape=years_of_program + 1,
            desc="Project development cost distribution over the development years",
        )
        self.add_input(
            "data:cost:production:annual_gross_profit",
            val=np.nan,
            units="USD",
            desc="Gross profit per unit sales of the aircraft manufacturer",
            shape=years_of_program + 1,
        )

        self.add_output(
            "data:cost:production:annual_cash_flow",
            val=1.0e5,
            units="USD",
            desc="Annual cash flow of the aircraft manufacturer",
            shape=years_of_program + 1,
        )

    def setup_partials(self):
        years_of_program = self.options["years_of_program"]
        self.declare_partials(
            of="*",
            wrt="data:cost:production:project_development_cost_distribution",
            method="exact",
            rows=np.arange(years_of_program + 1),
            cols=np.arange(years_of_program + 1),
            val=-1.0,
        )
        self.declare_partials(
            of="*",
            wrt="data:cost:production:annual_gross_profit",
            method="exact",
            rows=np.arange(years_of_program + 1),
            cols=np.arange(years_of_program + 1),
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:annual_cash_flow"] = (
            inputs["data:cost:production:annual_gross_profit"]
            - inputs["data:cost:production:project_development_cost_distribution"]
        )
