# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCProfitabilityIndex(om.ExplicitComponent):
    """
    Computation of the profitability Index (PI) of the aircraft manufacturer/operator .
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
            "net_present_value",
            val=np.nan,
            units="USD",
            desc="Annual net cash flow of the aircraft manufacturer",
            shape=duration_in_years + 1,
        )
        self.add_input(
            "initial_investment",
            val=np.nan,
            units="USD",
            desc="The initial investment of the aircraft manufacturer/operator.",
        )

        self.add_output(
            "profitability_index",
            val=0.0,
            desc="The profitability index of the aircraft manufacturer/operator.",
            shape=duration_in_years + 1,
        )

    def setup_partials(self):
        duration_in_years = self.options["duration_in_years"]

        self.declare_partials(
            of="*",
            wrt="initial_investment",
            method="exact",
            rows=np.arange(duration_in_years + 1),
            cols=np.zeros(duration_in_years + 1),
        )
        self.declare_partials(
            of="*",
            wrt="net_present_value",
            method="exact",
            rows=np.arange(duration_in_years + 1),
            cols=np.arange(duration_in_years + 1),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["profitability_index"] = (
            1.0 + inputs["net_present_value"] / inputs["initial_investment"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        duration_in_years = self.options["duration_in_years"]

        partials["profitability_index", "initial_investment"] = -inputs["net_present_value"] / (
            inputs["initial_investment"] ** 2.0
        )

        partials["profitability_index", "net_present_value"] = (
            np.ones(duration_in_years + 1) / inputs["initial_investment"]
        )
