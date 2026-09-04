# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCProfitabilityIndexOptimization(om.ExplicitComponent):
    """
    Extract the last profitability index factor for optimization.
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
            "profitability_index",
            val=0.0,
            desc="The profitability index of the aircraft manufacturer/operator.",
            shape=duration_in_years + 1,
        )

        self.add_output(
            "profitability_index_factor",
            val=0.0,
            desc="The profitability index factor for optimization.",
        )

    def setup_partials(self):
        duration_in_years = self.options["duration_in_years"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.zeros(duration_in_years + 1),
            cols=np.arange(duration_in_years + 1),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["profitability_index_factor"] = 1.0 / inputs["profitability_index"][-1]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["profitability_index_factor", "profitability_index"] = 0.0
        partials["profitability_index_factor", "profitability_index"][-1] = (
                -1.0 / inputs["profitability_index"][-1] ** 2.0
        )