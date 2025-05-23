# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCLearningCurveDiscount(om.ExplicitComponent):
    """
    Computation of the aircraft production learning curve discount factor for tooling and
    manufacturing. The computation is obtained from
    http://www.ae.metu.edu.tr/~ae452sc2/lecture8_cost.pdf. The learning curve percentage falls
    between 80% to 90% based on the results from :cite:`bongers:2017`.
    """

    def setup(self):
        self.add_input(
            "data:cost:production:learning_curve_percentage",
            val=85.0,
            units="percent",
            desc="The percentage decrease in unit production cost after extensive learning",
        )
        self.add_input(
            "data:cost:production:similar_aircraft_made",
            val=1.0,
            desc="The number of similar models of aircraft produced by the manufacturer",
        )
        self.add_input(
            "data:cost:production:number_aircraft_5_years",
            val=np.nan,
            desc="Number of planned aircraft to be produced over a 5-year period or 60 months",
        )

        self.add_output(
            "data:cost:production:maturity_discount",
            val=1.0,
            desc="The discount factor in manufacturing and tooling bsed on process maturity",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:maturity_discount"] = (
            inputs["data:cost:production:similar_aircraft_made"]
            / inputs["data:cost:production:number_aircraft_5_years"]
        ) ** (np.log2(0.02 * inputs["data:cost:production:learning_curve_percentage"]) - 1.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aircraft_made = inputs["data:cost:production:similar_aircraft_made"]
        aircraft_planned = inputs["data:cost:production:number_aircraft_5_years"]
        factor = np.log2(0.02 * inputs["data:cost:production:learning_curve_percentage"])

        partials[
            "data:cost:production:maturity_discount",
            "data:cost:production:number_aircraft_5_years",
        ] = -(factor - 1.0) * (aircraft_made / aircraft_planned) ** factor / aircraft_made

        partials[
            "data:cost:production:maturity_discount",
            "data:cost:production:similar_aircraft_made",
        ] = (factor - 1.0) * (aircraft_made / aircraft_planned) ** (factor - 1.0) / aircraft_made

        partials[
            "data:cost:production:maturity_discount",
            "data:cost:production:learning_curve_percentage",
        ] = (
            (aircraft_made / aircraft_planned) ** (factor - 1.0)
            * np.log(aircraft_made / aircraft_planned)
            / (inputs["data:cost:production:learning_curve_percentage"] * np.log(2.0))
        )
