# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCProjectDevelopmentCostDistribution(om.Group):
    """
    Group collects all the project cost distribution calculation.
    """

    def initialize(self):
        self.options.declare(
            "years_of_development",
            types=int,
            default=10,
            desc="The number of years of development for the aircraft, before the first delivery.",
        )

    def setup(self):
        years_of_development = self.options["years_of_development"]

        self.add_subsystem(
            name="cumulative_project_development_cost_distribution",
            subsys=_CumulativeProjectDevelopmentCostDistribution(
                years_of_development=years_of_development
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="development_cost_distribution",
            subsys=_DevelopmentCostDistribution(years_of_development=years_of_development),
            promotes=["*"],
        )


class _CumulativeProjectDevelopmentCostDistribution(om.ExplicitComponent):
    """
    Computation of cumulative distribution of the development cost based on the 60/40 distribution
    obtained from :cite:`brown:2015`.
    """

    def initialize(self):
        self.options.declare(
            "years_of_development",
            types=int,
            default=10,
            desc="The number of years of development for the aircraft, before the first delivery.",
        )

    def setup(self):
        years_of_development = self.options["years_of_development"]

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
            shape=years_of_development + 1,
            desc="Project development cost cumulative distribution over the development years",
        )

    def setup_partials(self):
        years_of_development = self.options["years_of_development"]
        progression_percentage = np.linspace(0.0, 1.0, years_of_development + 1)

        self.declare_partials(
            "*",
            "*",
            method="exact",
            rows=np.arange(years_of_development + 1),
            cols=np.zeros(years_of_development + 1),
            val=((1.0 - np.exp(-3.52 * progression_percentage**2.0)) / 0.97),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        years_of_development = self.options["years_of_development"]
        progression_percentage = np.linspace(0.0, 1.0, years_of_development + 1)

        outputs["data:cost:project_development_cost_cumulative_distribution"] = (
            (1.0 - np.exp(-3.52 * progression_percentage**2.0)) / 0.97
        ) * inputs["data:cost:production:total_non_recursive_project_cost"]


class _DevelopmentCostDistribution(om.ExplicitComponent):
    """
    Computation of the development cost distribution based on the 60/40 distribution
    obtained from :cite:`brown:2015`.
    """

    def initialize(self):
        self.options.declare(
            "years_of_development",
            types=int,
            default=10,
            desc="The number of years of development for the aircraft, before the first delivery.",
        )

    def setup(self):
        years_of_development = self.options["years_of_development"]

        self.add_input(
            "data:cost:project_development_cost_cumulative_distribution",
            units="USD",
            val=np.nan,
            shape=years_of_development + 1,
        )

        self.add_output(
            "data:cost:project_development_cost_distribution",
            units="USD",
            val=0.0,
            shape=years_of_development + 1,
            desc="Project development cost distribution over the development years",
        )

    def setup_partials(self):
        years_of_development = self.options["years_of_development"]

        rows = np.concatenate(
            [np.arange(years_of_development + 1), np.arange(1, years_of_development + 1)]
        )
        cols = np.concatenate(
            [np.arange(years_of_development + 1), np.arange(0, years_of_development)]
        )

        self.declare_partials(
            "data:cost:project_development_cost_distribution",
            "data:cost:project_development_cost_cumulative_distribution",
            method="exact",
            rows=rows,
            cols=cols,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        years_of_development = self.options["years_of_development"]
        cumulative_distribution = inputs[
            "data:cost:project_development_cost_cumulative_distribution"
        ]
        distribution = np.zeros(years_of_development + 1)

        distribution[0] = cumulative_distribution[0]
        for i in range(1, years_of_development + 1):
            distribution[i] = cumulative_distribution[i] - cumulative_distribution[i - 1]

        outputs["data:cost:project_development_cost_distribution"] = distribution

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        n = self.options["years_of_development"]
        partials[
            "data:cost:project_development_cost_distribution",
            "data:cost:project_development_cost_cumulative_distribution",
        ] = np.concatenate([np.ones(n + 1), -np.ones(n)])
