# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCEngineeringCost(om.ExplicitComponent):
    """
    Computation of the airframe engineering labor cost as obtained from :cite:`gudmundsson:2013`.
    Default engineering cost per hour is provided by :cite:`stefana:2024`.
    """

    def setup(self):
        self.add_input(
            "data:cost:production:engineering_man_hours_5_years",
            val=np.nan,
            units="h",
            desc="Number of engineering man-hours required for a certain number of aircraft to be"
            "produced in a 5-year or 60 month period",
        )
        self.add_input(
            "data:cost:production:engineering_cost_per_hour",
            val=126.54,
            units="USD/h",
            desc="Engineering cost per hour",
        )
        self.add_input(
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )

        self.add_output(
            "data:cost:production:engineering_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Engineering adjusted cost per aircraft",
        )

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:engineering_cost_per_unit"] = (
            2.0969
            * inputs["data:cost:production:engineering_man_hours_5_years"]
            * inputs["data:cost:production:engineering_cost_per_hour"]
            * inputs["data:cost:cpi_2012"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:cost:production:engineering_cost_per_unit",
            "data:cost:production:engineering_man_hours_5_years",
        ] = (
            2.0969
            * inputs["data:cost:production:engineering_cost_per_hour"]
            * inputs["data:cost:cpi_2012"]
        )

        partials[
            "data:cost:production:engineering_cost_per_unit",
            "data:cost:production:engineering_cost_per_hour",
        ] = (
            2.0969
            * inputs["data:cost:production:engineering_man_hours_5_years"]
            * inputs["data:cost:cpi_2012"]
        )

        partials["data:cost:production:engineering_cost_per_unit", "data:cost:cpi_2012"] = (
            2.0969
            * inputs["data:cost:production:engineering_man_hours_5_years"]
            * inputs["data:cost:production:engineering_cost_per_hour"]
        )
