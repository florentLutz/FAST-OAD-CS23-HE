# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCToolingCost(om.ExplicitComponent):
    """
    Computation of the airframe cost per aircraft of tooling labor obtained from
    :cite:`gudmundsson:2013`. Default tooling cost per hour is provided by :cite:`stefana:2024`.
    """

    def setup(self):
        self.add_input(
            "data:cost:production:tooling_man_hours",
            val=np.nan,
            units="h",
            desc="Number of tooling man-hours required for a certain amount of aircraft been "
            "produced in a 5-year or 60 month period",
        )
        self.add_input(
            "data:cost:production:tooling_cost_per_hour",
            val=83.97,
            units="USD/h",
            desc="Tooling labor cost per hour",
        )
        self.add_input(
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )

        self.add_output(
            "data:cost:production:tooling_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Tooling adjusted cost per aircraft",
        )

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:tooling_cost_per_unit"] = (
            2.0969
            * inputs["data:cost:production:tooling_man_hours"]
            * inputs["data:cost:production:tooling_cost_per_hour"]
            * inputs["data:cost:cpi_2012"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        mh_tooling = inputs["data:cost:production:tooling_man_hours"]
        cost_rate_tooling = inputs["data:cost:production:tooling_cost_per_hour"]
        cpi_2012 = inputs["data:cost:cpi_2012"]

        partials[
            "data:cost:production:tooling_cost_per_unit",
            "data:cost:production:tooling_man_hours",
        ] = 2.0969 * cost_rate_tooling * cpi_2012

        partials[
            "data:cost:production:tooling_cost_per_unit",
            "data:cost:production:tooling_cost_per_hour",
        ] = 2.0969 * mh_tooling * cpi_2012

        partials["data:cost:production:tooling_cost_per_unit", "data:cost:cpi_2012"] = (
            2.0969 * mh_tooling * cost_rate_tooling
        )
