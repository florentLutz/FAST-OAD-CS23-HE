# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCProjectDevelopmentCostDistribution(om.ExplicitComponent):
    """
    Computation of cumulative distribution of the development cost based on the 60/40 distribution
    obtained from :cite:`brown:2015`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_development_years",
            default=5,
            desc="number of years for the aircraft development",
        )

    def setup(self):
        number_of_development_years = self.options["number_of_development_years"]

        self.add_input(
            "data:cost:production:total_non_recursive_project_cost",
            val=np.nan,
            units="USD",
            desc="Total non-recursive project cost for the aircraft development",
        )

        self.add_output(
            "data:cost:project_development_cost_cumulative_distribution",
            units="USD",
            val=0.0,
            shape=number_of_development_years + 1,
            desc="Project development cost cumulative distribution over the development years",
        )

    def setup_partials(self):
        number_of_development_years = self.options["number_of_development_years"]
        progression_percentage = np.linspace(0.0, 1.0, number_of_development_years + 1)

        self.declare_partials(
            "*",
            "*",
            method="exact",
            rows=np.arange(number_of_development_years + 1),
            cols=np.zeros(number_of_development_years + 1),
            val=((1.0 - np.exp(-3.52 * progression_percentage**2.0)) / 0.97),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_development_years = self.options["number_of_development_years"]
        progression_percentage = np.linspace(0.0, 1.0, number_of_development_years + 1)

        outputs["data:cost:project_development_cost_cumulative_distribution"] = (
            (1.0 - np.exp(-3.52 * progression_percentage**2.0)) / 0.97
        ) * inputs["data:cost:production:total_non_recursive_project_cost"]
