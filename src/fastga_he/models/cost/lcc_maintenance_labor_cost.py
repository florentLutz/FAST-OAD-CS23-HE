# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCMaintenanceLaborCost(om.ExplicitComponent):
    """
    Compute the annual labor cost of the airframe maintenance, obtained from :cite:`salgas:2025`.
    The calculation is adjusted based on the cost rate from https://elixir-aircraft.com/.
    """

    def setup(self):
        self.add_input("data:weight:airframe:mass", units="t", val=np.nan)

        self.add_input(
            "data:cost:operation:maintenance_labor_rate",
            val=54.0,
            units="USD/h",
            desc="overhead costs from management, training and others for airframe maintenance",
        )

        self.add_input(
            "data:cost:operation:mission_per_year",
            val=np.nan,
            units="1/yr",
            desc="Flight mission per year",
        )

        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_output(
            "data:cost:operation:airframe_labor_cost",
            val=2.0e4,
            units="USD/yr",
            desc="Annual airframe maintenance labor cost per aircraft",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ft_year = inputs["data:TLAR:flight_hours_per_year"]
        labor_rate = inputs["data:cost:operation:maintenance_labor_rate"]
        mass = inputs["data:weight:airframe:mass"]
        mission_per_year = inputs["data:cost:operation:mission_per_year"]

        outputs["data:cost:operation:airframe_labor_cost"] = (
            labor_rate * ((0.655 + 0.01 * mass) * ft_year + 0.254 + 0.01 * mass) * mission_per_year
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ft_year = inputs["data:TLAR:flight_hours_per_year"]
        labor_rate = inputs["data:cost:operation:maintenance_labor_rate"]
        mass = inputs["data:weight:airframe:mass"]
        mission_per_year = inputs["data:cost:operation:mission_per_year"]

        partials[
            "data:cost:operation:airframe_labor_cost", "data:cost:operation:maintenance_labor_rate"
        ] = ((0.655 + 0.01 * mass) * ft_year + 0.254 + 0.01 * mass) * mission_per_year

        partials["data:cost:operation:airframe_labor_cost", "data:weight:airframe:mass"] = (
            labor_rate * (0.01 * ft_year + 0.01) * mission_per_year
        )

        partials["data:cost:operation:airframe_labor_cost", "data:TLAR:flight_hours_per_year"] = (
            labor_rate * (0.655 + 0.01 * mass) * mission_per_year
        )

        partials[
            "data:cost:operation:airframe_labor_cost", "data:cost:operation:mission_per_year"
        ] = labor_rate * ((0.655 + 0.01 * mass) * ft_year + 0.254 + 0.01 * mass)
