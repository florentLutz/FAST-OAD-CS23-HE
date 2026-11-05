# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCNPVDiscountFactor(om.ExplicitComponent):
    """
    Computation of the annual discount factor of the NPV calculation.
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
            "discount_rate",
            val=np.nan,
            desc="The discount rate used to compute the NPV.",
        )

        self.add_output(
            "annual_discount_factor",
            val=0.5,
            desc="Annual net cash flow of the aircraft manufacturer",
            shape=duration_in_years + 1,
        )

    def setup_partials(self):
        duration_in_years = self.options["duration_in_years"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(duration_in_years + 1),
            cols=np.zeros(duration_in_years + 1),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        duration_in_years = self.options["duration_in_years"]

        outputs["annual_discount_factor"] = 1.0 / (1.0 + inputs["discount_rate"]) ** np.arange(
            duration_in_years + 1
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        duration_in_years = self.options["duration_in_years"]

        partials["annual_discount_factor", "discount_rate"] = -np.arange(duration_in_years + 1) / (
            1.0 + inputs["discount_rate"]
        ) ** (np.arange(duration_in_years + 1) + 1)
