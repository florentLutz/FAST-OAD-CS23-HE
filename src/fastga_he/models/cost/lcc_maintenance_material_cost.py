# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCMaintenanceMaterialCost(om.ExplicitComponent):
    """
    Compute the annual material cost of the airframe maintenance, obtained from :cite:`salgas:2025`.
    The calculation is adjusted based on the cost rate from https://elixir-aircraft.com/.
    """

    def setup(self):
        self.add_input("data:weight:airframe:mass", units="t", val=np.nan)

        self.add_input(
            name="data:TLAR:flight_hours_per_year",
            val=np.nan,
            units="h",
            desc="Expected number of hours flown per year",
        )

        self.add_output(
            "data:cost:operation:airframe_material_cost",
            val=200.0,
            units="USD/yr",
            desc="Annual airframe maintenance material cost per aircraft",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ft_year = inputs["data:TLAR:flight_hours_per_year"]
        mass = inputs["data:weight:airframe:mass"]

        outputs["data:cost:operation:airframe_material_cost"] = (
            mass * (0.21 * ft_year + 13.7) + 57.5
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ft_year = inputs["data:TLAR:flight_hours_per_year"]
        mass = inputs["data:weight:airframe:mass"]

        partials["data:cost:operation:airframe_material_cost", "data:weight:airframe:mass"] = (
            0.21 * ft_year + 13.7
        )

        partials[
            "data:cost:operation:airframe_material_cost", "data:TLAR:flight_hours_per_year"
        ] = mass * 0.21
