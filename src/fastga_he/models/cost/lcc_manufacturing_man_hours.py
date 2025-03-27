# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCManufacturingManHours(om.ExplicitComponent):
    """
    Compute the number of man-hours required in manufacturing, obtained from
    :cite:`gudmundsson:2013`.
    """

    def setup(self):
        self.add_input("data:weight:airframe:mass", units="kg", val=np.nan)
        self.add_input("data:TLAR:v_cruise", units="kn", val=np.nan)
        self.add_input(
            "data:cost:airframe:num_aircraft_5years",
            val=np.nan,
            desc="Number of planned aircraft to be produced over a 5-year period or 60 months",
        )
        self.add_input(
            "data:cost:airframe:flap_factor",
            val=1.0,
            desc="Set to 1.01 for complex flap, 1.0 for simple flap",
        )

        self.add_input(
            "data:cost:airframe:composite_fraction",
            val=0.0,
            desc="Fraction of airframe made by composite, range from 0.0 to 1.0",
        )

        self.add_output(
            "data:cost:airframe:manufacturing_man_hours",
            val=2.0e5,
            units="h",
            desc="Number of engineering man-hours required per aircraft",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:airframe:manufacturing_man_hours"] = (
            9.6613
            * inputs["data:weight:airframe:mass"] ** 0.74
            * inputs["data:TLAR:v_cruise"] ** 0.543
            * inputs["data:cost:airframe:num_aircraft_5years"] ** -0.476
            * inputs["data:cost:airframe:flap_factor"]
            * (1.0 + 0.25 * inputs["data:cost:airframe:composite_fraction"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m_airframe = inputs["data:weight:airframe:mass"]
        v_cruise = inputs["data:TLAR:v_cruise"]
        num_5years = inputs["data:cost:airframe:num_aircraft_5years"]
        f_flap = inputs["data:cost:airframe:flap_factor"]
        f_composite = inputs["data:cost:airframe:composite_fraction"]

        partials[
            "data:cost:airframe:manufacturing_man_hours",
            "data:weight:airframe:mass",
        ] = (
            7.149362 * v_cruise**0.543 * num_5years**-0.476 * f_flap * (1.0 + 0.25 * f_composite)
        ) / m_airframe**0.26

        partials[
            "data:cost:airframe:manufacturing_man_hours",
            "data:TLAR:v_cruise",
        ] = (
            5.2460859
            * m_airframe**0.74
            * num_5years**-0.476
            * f_flap
            * (1.0 + 0.25 * f_composite)
            / v_cruise**0.457
        )

        partials[
            "data:cost:airframe:manufacturing_man_hours",
            "data:cost:airframe:num_aircraft_5years",
        ] = (
            -4.5987788 * m_airframe**0.74 * v_cruise**0.543 * f_flap * (1.0 + 0.25 * f_composite)
        ) / num_5years**1.476

        partials[
            "data:cost:airframe:manufacturing_man_hours",
            "data:cost:airframe:flap_factor",
        ] = (
            9.6613
            * m_airframe**0.74
            * v_cruise**0.543
            * num_5years**-0.476
            * (1.0 + 0.25 * f_composite)
        )

        partials[
            "data:cost:airframe:manufacturing_man_hours",
            "data:cost:airframe:composite_fraction",
        ] = 2.415325 * m_airframe**0.74 * v_cruise**0.543 * num_5years**-0.476 * f_flap
