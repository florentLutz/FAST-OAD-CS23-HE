# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCEngineeringManHours(om.ExplicitComponent):
    """
    Compute the number of man-hours required in engineering labor, obtained from
    :cite:`gudmundsson:2013`.
    """

    def setup(self):
        self.add_input("data:weight:airframe:mass", units="kg", val=np.nan)
        self.add_input("data:cost:v_cruise_design", units="kn", val=np.nan)
        self.add_input("data:geometry:flap_type", val=np.nan)
        self.add_input(
            "data:geometry:cabin:pressurized",
            val=0.0,
            desc="Cabin pressurization; 0.0 for no pressurization, 1.0 for pressurization",
        )
        self.add_input(
            "data:cost:production:num_aircraft_5years",
            val=np.nan,
            desc="Number of planned aircraft to be produced over a 5-year period or 60 months",
        )

        self.add_input(
            "data:cost:production:composite_fraction",
            val=0.0,
            desc="Fraction of airframe made by composite, range from 0.0 to 1.0",
        )

        self.add_output(
            "data:cost:production:engineering_man_hours",
            val=2.0e5,
            units="h",
            desc="Number of engineering man-hours required per aircraft",
        )

        self.declare_partials(of="*", wrt="*", method="exact")
        self.declare_partials("*", "data:geometry:flap_type", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if inputs["data:geometry:flap_type"] != 0.0:
            f_flap = 1.03
        else:
            f_flap = 1.0

        outputs["data:cost:production:engineering_man_hours"] = (
            0.0396
            * inputs["data:weight:airframe:mass"] ** 0.791
            * inputs["data:cost:v_cruise_design"] ** 1.526
            * inputs["data:cost:production:num_aircraft_5years"] ** -0.817
            * f_flap
            * (1.0 + inputs["data:cost:production:composite_fraction"])
            * (1.0 + 0.03 * inputs["data:geometry:cabin:pressurized"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m_airframe = inputs["data:weight:airframe:mass"]
        v_cruise = inputs["data:cost:v_cruise_design"]
        num_5years = inputs["data:cost:production:num_aircraft_5years"]
        f_composite = inputs["data:cost:production:composite_fraction"]
        pressurized = inputs["data:geometry:cabin:pressurized"]

        if inputs["data:geometry:flap_type"] != 0.0:
            f_flap = 1.03
        else:
            f_flap = 1.0

        partials[
            "data:cost:production:engineering_man_hours",
            "data:weight:airframe:mass",
        ] = (
            0.0313236
            * v_cruise**1.526
            * num_5years**-0.817
            * f_flap
            * (1.0 + f_composite)
            * (1.0 + 0.03 * pressurized)
        ) / m_airframe**0.209

        partials[
            "data:cost:production:engineering_man_hours",
            "data:cost:v_cruise_design",
        ] = (
            0.0604296
            * m_airframe**0.791
            * v_cruise**0.526
            * num_5years**-0.817
            * f_flap
            * (1.0 + f_composite)
            * (1.0 + 0.03 * pressurized)
        )

        partials[
            "data:cost:production:engineering_man_hours",
            "data:cost:production:num_aircraft_5years",
        ] = (
            -0.0323532
            * m_airframe**0.791
            * v_cruise**1.526
            * f_flap
            * (1.0 + f_composite)
            * (1.0 + 0.03 * pressurized)
        ) / num_5years**1.817

        partials[
            "data:cost:production:engineering_man_hours",
            "data:cost:production:composite_fraction",
        ] = (
            0.0396
            * m_airframe**0.791
            * v_cruise**1.526
            * num_5years**-0.817
            * f_flap
            * (1.0 + 0.03 * pressurized)
        )

        partials[
            "data:cost:production:engineering_man_hours",
            "data:geometry:cabin:pressurized",
        ] = (
            0.001188
            * m_airframe**0.791
            * v_cruise**1.526
            * num_5years**-0.817
            * f_flap
            * (1.0 + f_composite)
        )
