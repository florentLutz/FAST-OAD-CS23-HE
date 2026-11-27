# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCNetPresentValue(om.ExplicitComponent):
    """
    Computation of the annual net present value of the aircraft manufacturer/operator .
    """

    def initialize(self):
        self.options.declare(
            "duration_in_years",
            types=int,
            default=30,
            desc="The total number of years of the aircraft production/ service life.",
        )

    def setup(self):
        duration_in_years = self.options["duration_in_years"]

        self.add_input(
            "annual_discount_factor",
            val=np.nan,
            desc="Annual net cash flow of the aircraft manufacturer",
            shape=duration_in_years + 1,
        )
        self.add_input(
            "annual_net_cash_flow",
            val=np.nan,
            units="USD",
            desc="Annual net cash flow of the aircraft manufacturer/operator over the aircraft "
            "production / service life.",
            shape=duration_in_years + 1,
        )

        self.add_output(
            "net_present_value",
            val=0.0,
            units="USD",
            desc="Net Present Value of the aircraft manufacturer/operator over the aircraft "
            "production / service life.",
            shape=duration_in_years + 1,
        )

    def setup_partials(self):
        duration_in_years = self.options["duration_in_years"]

        # lower-triangular sparsity: output j depends on inputs 0..j
        rows, cols = np.tril_indices(duration_in_years + 1)

        self.declare_partials(
            of="net_present_value",
            wrt="annual_discount_factor",
            method="exact",
            rows=rows,
            cols=cols,
        )
        self.declare_partials(
            of="net_present_value",
            wrt="annual_net_cash_flow",
            method="exact",
            rows=rows,
            cols=cols,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["net_present_value"] = np.cumsum(
            inputs["annual_discount_factor"] * inputs["annual_net_cash_flow"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options["duration_in_years"] + 1
        rows, cols = np.tril_indices(n)

        partials["net_present_value", "annual_discount_factor"] = inputs["annual_net_cash_flow"][
            cols
        ]

        partials["net_present_value", "annual_net_cash_flow"] = inputs["annual_discount_factor"][
            cols
        ]
