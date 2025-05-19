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
        self.add_input("data:cost:v_cruise_design", units="kn", val=np.nan)
        self.add_input("data:geometry:flap_type", val=np.nan)
        self.add_input(
            "data:cost:production:number_aircraft_5_years",
            val=np.nan,
            desc="Number of planned aircraft to be produced over a 5-year period or 60 months",
        )
        self.add_input(
            "data:cost:production:composite_fraction",
            val=0.0,
            desc="Fraction of airframe made of composite, range from 0.0 to 1.0",
        )

        self.add_output(
            "data:cost:production:manufacturing_man_hours_5_years",
            val=2.0e5,
            units="h",
            desc="Number of engineering man-hours required per aircraft",
        )

        self.declare_partials(of="*", wrt="*", method="exact")
        self.declare_partials("*", "data:geometry:flap_type", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if inputs["data:geometry:flap_type"] != 0.0:
            f_flap = 1.01
        else:
            f_flap = 1.0

        outputs["data:cost:production:manufacturing_man_hours_5_years"] = (
            9.6613
            * inputs["data:weight:airframe:mass"] ** 0.74
            * inputs["data:cost:v_cruise_design"] ** 0.543
            * inputs["data:cost:production:number_aircraft_5_years"] ** -0.476
            * f_flap
            * (1.0 + 0.25 * inputs["data:cost:production:composite_fraction"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m_airframe = inputs["data:weight:airframe:mass"]
        v_cruise = inputs["data:cost:v_cruise_design"]
        num_5years = inputs["data:cost:production:number_aircraft_5_years"]
        f_composite = inputs["data:cost:production:composite_fraction"]

        if inputs["data:geometry:flap_type"] != 0.0:
            f_flap = 1.01
        else:
            f_flap = 1.0

        partials[
            "data:cost:production:manufacturing_man_hours_5_years",
            "data:weight:airframe:mass",
        ] = (
            7.149362 * v_cruise**0.543 * num_5years**-0.476 * f_flap * (1.0 + 0.25 * f_composite)
        ) / m_airframe**0.26

        partials[
            "data:cost:production:manufacturing_man_hours_5_years",
            "data:cost:v_cruise_design",
        ] = (
            5.2460859
            * m_airframe**0.74
            * num_5years**-0.476
            * f_flap
            * (1.0 + 0.25 * f_composite)
            / v_cruise**0.457
        )

        partials[
            "data:cost:production:manufacturing_man_hours_5_years",
            "data:cost:production:number_aircraft_5_years",
        ] = (
            -4.5987788 * m_airframe**0.74 * v_cruise**0.543 * f_flap * (1.0 + 0.25 * f_composite)
        ) / num_5years**1.476

        partials[
            "data:cost:production:manufacturing_man_hours_5_years",
            "data:cost:production:composite_fraction",
        ] = 2.415325 * m_airframe**0.74 * v_cruise**0.543 * num_5years**-0.476 * f_flap
