# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCFlightTestCost(om.ExplicitComponent):
    """
    Computation of the airframe cost of flight test per aircraft during developmet obtained from
    :cite:`gudmundsson:2013`.
    """

    def setup(self):
        self.add_input("data:weight:airframe:mass", units="kg", val=np.nan)
        self.add_input("data:TLAR:v_cruise", units="kn", val=np.nan)
        self.add_input(
            "data:cost:prototype_number",
            val=np.nan,
            desc="number of prototypes",
        )
        self.add_input(
            "data:cost:num_aircraft_5years",
            val=np.nan,
            desc="Number of planned aircraft to be produced over a 5-year period or 60 months",
        )
        self.add_input(
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )

        self.add_output(
            "data:cost:flight_test_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Development flight test adjusted cost per aircraft",
        )
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:flight_test_cost_per_unit"] = (
            0.009646
            * inputs["data:weight:airframe:mass"] ** 1.16
            * inputs["data:TLAR:v_cruise"] ** 1.3718
            * inputs["data:cost:prototype_number"] ** 1.281
            * inputs["data:cost:cpi_2012"]
            / inputs["data:cost:num_aircraft_5years"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m_airframe = inputs["data:weight:airframe:mass"]
        v_cruise = inputs["data:TLAR:v_cruise"]
        cpi_2012 = inputs["data:cost:cpi_2012"]
        num_prototype = inputs["data:cost:prototype_number"]
        num_5years = inputs["data:cost:num_aircraft_5years"]

        partials["data:cost:flight_test_cost_per_unit", "data:weight:airframe:mass"] = (
            0.01118936
            * m_airframe**0.16
            * v_cruise**1.3718
            * num_prototype**1.281
            * cpi_2012
            / num_5years
        )

        partials["data:cost:flight_test_cost_per_unit", "data:TLAR:v_cruise"] = (
            0.0132323828
            * m_airframe**1.16
            * v_cruise**0.3718
            * num_prototype**1.281
            * cpi_2012
            / num_5years
        )

        partials[
            "data:cost:flight_test_cost_per_unit",
            "data:cost:prototype_number",
        ] = (
            0.012356526
            * m_airframe**1.16
            * v_cruise**1.3718
            * num_prototype**0.281
            * cpi_2012
            / num_5years
        )

        partials["data:cost:flight_test_cost_per_unit", "data:cost:cpi_2012"] = (
            0.009646 * m_airframe**1.16 * v_cruise**1.3718 * num_prototype**1.281 / num_5years
        )

        partials[
            "data:cost:flight_test_cost_per_unit",
            "data:cost:num_aircraft_5years",
        ] = (
            -0.009646
            * m_airframe**1.16
            * v_cruise**1.3718
            * num_prototype**1.281
            * cpi_2012
            / num_5years**2.0
        )
