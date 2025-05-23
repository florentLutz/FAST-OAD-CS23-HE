# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCManufacturingCost(om.ExplicitComponent):
    """
    Computation of the manufacturing labor cost for the airframe as obtained from
    :cite:`gudmundsson:2013`. Default manufacturing cost per hour is provided by
    :cite:`stefana:2024`.
    """

    def setup(self):
        self.add_input(
            "data:cost:production:manufacturing_man_hours_5_years",
            val=np.nan,
            units="h",
            desc="Number of tooling man-hours required for a certain number of aircraft to be"
            "produced in a 5-year or 60 month period",
        )
        self.add_input(
            "data:cost:production:manufacturing_cost_per_hour",
            val=72.97,
            units="USD/h",
            desc="Manufacturing labor cost per hour",
        )
        self.add_input(
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )
        self.add_input(
            "data:cost:production:maturity_discount",
            val=1.0,
            desc="The discount factor in manufacturing and tooling bsed on process maturity",
        )

        self.add_output(
            "data:cost:production:manufacturing_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Manufacturing adjusted cost per aircraft",
        )

        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:manufacturing_cost_per_unit"] = (
            2.0969
            * inputs["data:cost:production:manufacturing_man_hours_5_years"]
            * inputs["data:cost:production:manufacturing_cost_per_hour"]
            * inputs["data:cost:cpi_2012"]
            * inputs["data:cost:production:maturity_discount"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cost_per_hour = inputs["data:cost:production:manufacturing_cost_per_hour"]
        cpi_2022 = inputs["data:cost:cpi_2012"]
        discount = inputs["data:cost:production:maturity_discount"]
        man_hours = inputs["data:cost:production:manufacturing_man_hours_5_years"]

        partials[
            "data:cost:production:manufacturing_cost_per_unit",
            "data:cost:production:manufacturing_man_hours_5_years",
        ] = 2.0969 * cost_per_hour * cpi_2022 * discount

        partials[
            "data:cost:production:manufacturing_cost_per_unit",
            "data:cost:production:manufacturing_cost_per_hour",
        ] = 2.0969 * man_hours * cpi_2022 * discount

        partials["data:cost:production:manufacturing_cost_per_unit", "data:cost:cpi_2012"] = (
            2.0969 * man_hours * cost_per_hour * discount
        )

        partials[
            "data:cost:production:manufacturing_cost_per_unit",
            "data:cost:production:maturity_discount",
        ] = 2.0969 * man_hours * cost_per_hour * cpi_2022
