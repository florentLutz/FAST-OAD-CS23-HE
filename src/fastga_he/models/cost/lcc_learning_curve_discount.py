# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCLearningCurveDiscount(om.ExplicitComponent):
    """
    Computation of the aircraft production learning curve discount factor for tooling and
    manufacturing. The computation is obtained from
    http://www.ae.metu.edu.tr/~ae452sc2/lecture8_cost.pdf. The learning curve percentage falls
    between 80% to 90% based on the results from :cite:`bongers:2017`.
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
            "data:cost:production:learning_curve_percentage",
            val=95.0,
            units="percent",
            desc="The percentage decrease in unit production cost after extensive learning",
        )
        self.add_input(
            "data:cost:production:similar_aircraft_made",
            val=1.0,
            desc="The cumulative number of similar aircraft produced over the production",
            shape=years_of_program - years_of_development,
        )

        self.add_output(
            "data:cost:production:maturity_discount",
            val=1.0,
            desc="The discount factor in manufacturing and tooling bsed on process maturity",
            shape=years_of_program - years_of_development,
        )

    def setup_partials(self):
        years_of_development = self.options["years_of_development"]
        years_of_program = self.options["years_of_program"]

        self.declare_partials(
            of="*",
            wrt="data:cost:production:similar_aircraft_made",
            method="exact",
            rows=np.arange(years_of_program - years_of_development),
            cols=np.arange(years_of_program - years_of_development),
        )
        self.declare_partials(
            of="*",
            wrt="data:cost:production:learning_curve_percentage",
            method="exact",
            rows=np.arange(years_of_program - years_of_development),
            cols=np.zeros(years_of_program - years_of_development),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:maturity_discount"] = (
            inputs["data:cost:production:similar_aircraft_made"]
        ) ** (np.log2(0.02 * inputs["data:cost:production:learning_curve_percentage"]) - 1.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aircraft_made = inputs["data:cost:production:similar_aircraft_made"]
        factor = np.log2(0.02 * inputs["data:cost:production:learning_curve_percentage"])

        partials[
            "data:cost:production:maturity_discount",
            "data:cost:production:similar_aircraft_made",
        ] = (factor - 1.0) * aircraft_made ** (factor - 1.0) / aircraft_made

        partials[
            "data:cost:production:maturity_discount",
            "data:cost:production:learning_curve_percentage",
        ] = (
            aircraft_made ** (factor - 1.0)
            * np.log(aircraft_made)
            / (inputs["data:cost:production:learning_curve_percentage"] * np.log(2.0))
        )
