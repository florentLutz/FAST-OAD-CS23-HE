# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCTotalNonRecursiveProjectCost(om.ExplicitComponent):
    """
    Computation of summing all the non-recursive costs and reductions for production phase.
    """

    def setup(self):
        self.add_input(
            "data:cost:production:number_aircraft_5_years",
            val=np.nan,
            desc="Number of planned aircraft to be produced over a 5-year period or 60 months",
        )
        self.add_input(
            "data:cost:production:engineering_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Engineering adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:production:tooling_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Tooling adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:production:flight_test_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Development flight test adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:production:dev_support_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Development support adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:production:certification_cost_per_unit",
            val=0.0,
            units="USD",
            desc="Certification adjusted cost per aircraft",
        )

        self.add_output(
            "data:cost:production:total_non_recursive_project_cost", units="USD", val=0.0
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_aircraft_5_years = inputs["data:cost:production:number_aircraft_5_years"]
        engineering_cost_per_unit = inputs["data:cost:production:engineering_cost_per_unit"]
        tooling_cost_per_unit = inputs["data:cost:production:tooling_cost_per_unit"]
        flight_test_cost_per_unit = inputs["data:cost:production:flight_test_cost_per_unit"]
        dev_support_cost_per_unit = inputs["data:cost:production:dev_support_cost_per_unit"]
        certification_cost_per_unit = inputs["data:cost:production:certification_cost_per_unit"]

        outputs["data:cost:production:total_non_recursive_project_cost"] = (
            number_aircraft_5_years
            * (
                engineering_cost_per_unit
                + tooling_cost_per_unit
                + flight_test_cost_per_unit
                + dev_support_cost_per_unit
                + certification_cost_per_unit
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_aircraft_5_years = inputs["data:cost:production:number_aircraft_5_years"]
        engineering_cost_per_unit = inputs["data:cost:production:engineering_cost_per_unit"]
        tooling_cost_per_unit = inputs["data:cost:production:tooling_cost_per_unit"]
        flight_test_cost_per_unit = inputs["data:cost:production:flight_test_cost_per_unit"]
        dev_support_cost_per_unit = inputs["data:cost:production:dev_support_cost_per_unit"]
        certification_cost_per_unit = inputs["data:cost:production:certification_cost_per_unit"]

        partials[
            "data:cost:production:total_non_recursive_project_cost",
            "data:cost:production:number_aircraft_5_years",
        ] = (
            engineering_cost_per_unit
            + tooling_cost_per_unit
            + flight_test_cost_per_unit
            + dev_support_cost_per_unit
            + certification_cost_per_unit
        )

        partials[
            "data:cost:production:total_non_recursive_project_cost",
            "data:cost:production:engineering_cost_per_unit",
        ] = number_aircraft_5_years

        partials[
            "data:cost:production:total_non_recursive_project_cost",
            "data:cost:production:tooling_cost_per_unit",
        ] = number_aircraft_5_years

        partials[
            "data:cost:production:total_non_recursive_project_cost",
            "data:cost:production:flight_test_cost_per_unit",
        ] = number_aircraft_5_years

        partials[
            "data:cost:production:total_non_recursive_project_cost",
            "data:cost:production:dev_support_cost_per_unit",
        ] = number_aircraft_5_years

        partials[
            "data:cost:production:total_non_recursive_project_cost",
            "data:cost:production:certification_cost_per_unit",
        ] = number_aircraft_5_years
