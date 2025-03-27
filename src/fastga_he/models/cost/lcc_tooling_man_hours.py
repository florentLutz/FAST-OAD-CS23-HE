# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCToolManHours(om.ExplicitComponent):
    """
    Compute the number of man-hours required in tooling labor, obtained from
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
            "data:cost:airframe:taper_factor",
            val=1.0,
            desc="Set to 0.95 for non-tapered wing , 1.0 for tapered wing",
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

        self.add_input(
            "data:cost:airframe:pressurization_factor",
            val=1.0,
            desc="Set to 1.03 for pressurized aircraft, 1.0 for unpressurized",
        )

        self.add_output(
            "data:cost:airframe:toolring_man_hours",
            val=2.0e5,
            units="h",
            desc="Number of tooling man-hours required per aircraft",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:airframe:tooling_man_hours"] = (
            0.76565
            * inputs["data:weight:airframe:mass"] ** 0.764
            * inputs["data:TLAR:v_cruise"] ** 0.899
            * inputs["data:cost:airframe:num_aircraft_5years"] ** -0.756
            * inputs["data:cost:airframe:taper_factor"]
            * inputs["data:cost:airframe:flap_factor"]
            * (1.0 + inputs["data:cost:airframe:composite_fraction"])
            * inputs["data:cost:airframe:pressurization_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m_airframe = inputs["data:weight:airframe:mass"]
        v_cruise = inputs["data:TLAR:v_cruise"]
        num_5years = inputs["data:cost:airframe:num_aircraft_5years"]
        f_taper = inputs["data:cost:airframe:taper_factor"]
        f_flap = inputs["data:cost:airframe:flap_factor"]
        f_composite = inputs["data:cost:airframe:composite_fraction"]
        f_pressurized = inputs["data:cost:airframe:pressurization_factor"]

        partials[
            "data:cost:airframe:tooling_man_hours",
            "data:weight:airframe:mass",
        ] = (
            0.5849566
            * v_cruise**0.899
            * num_5years**-0.756
            * f_taper
            * f_flap
            * (1.0 + f_composite)
            * f_pressurized
        ) / m_airframe**0.236

        partials[
            "data:cost:airframe:tooling_man_hours",
            "data:TLAR:v_cruise",
        ] = (
            0.68831935
            * m_airframe**0.764
            * num_5years**-0.756
            * f_taper
            * f_flap
            * (1.0 + f_composite)
            * f_pressurized
            / v_cruise**0.101
        )

        partials[
            "data:cost:airframe:tooling_man_hours",
            "data:cost:airframe:num_aircraft_5years",
        ] = (
            -0.5788314
            * m_airframe**0.764
            * v_cruise**0.899
            * f_taper
            * f_flap
            * (1.0 + f_composite)
            * f_pressurized
        ) / num_5years**1.756

        partials[
            "data:cost:airframe:tooling_man_hours",
            "data:cost:airframe:taper_factor",
        ] = (
            0.76565
            * m_airframe**0.764
            * v_cruise**0.899
            * num_5years**-0.756
            * f_flap
            * (1.0 + f_composite)
            * f_pressurized
        )

        partials[
            "data:cost:airframe:tooling_man_hours",
            "data:cost:airframe:flap_factor",
        ] = (
            0.76565
            * m_airframe**0.764
            * v_cruise**0.899
            * num_5years**-0.756
            * f_taper
            * (1.0 + f_composite)
            * f_pressurized
        )

        partials[
            "data:cost:airframe:tooling_man_hours",
            "data:cost:airframe:composite_fraction",
        ] = (
            0.76565
            * m_airframe**0.764
            * v_cruise**0.899
            * num_5years**-0.756
            * f_taper
            * f_flap
            * f_pressurized
        )

        partials[
            "data:cost:airframe:tooling_man_hours",
            "data:cost:airframe:pressurization_factor",
        ] = (
            0.76565
            * m_airframe**0.764
            * v_cruise**0.899
            * num_5years**-0.756
            * f_taper
            * f_flap
            * (1.0 + f_composite)
        )
