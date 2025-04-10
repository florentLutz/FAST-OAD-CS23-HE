# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCMaintenanceLaborCost(om.ExplicitComponent):
    """
    Compute the annual labor cost of the airframe maintenance, obtained from :cite:``.
    """

    def setup(self):
        self.add_input("data:weight:airframe:mass", units="kg", val=np.nan)
        self.add_input("data:TLAR:v_cruise", units="kn", val=np.nan)
        self.add_input(
            "data:cost:num_aircraft_5years",
            val=np.nan,
            desc="Number of planned aircraft to be produced over a 5-year period or 60 months",
        )

        self.add_input(
            "data:cost:composite_fraction",
            val=0.0,
            desc="Fraction of airframe made by composite, range from 0.0 to 1.0",
        )

        self.add_output(
            "data:cost:tooling_man_hours",
            val=2.0e5,
            units="h",
            desc="Number of tooling man-hours required per aircraft",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options["complex_flap"]:
            f_flap = 1.02
        else:
            f_flap = 1.0

        if self.options["pressurized"]:
            f_pressurized = 1.01
        else:
            f_pressurized = 1.0

        if self.options["tapered_wing"]:
            f_tapered = 1.0
        else:
            f_tapered = 0.95

        outputs["data:cost:tooling_man_hours"] = (
            0.76565
            * inputs["data:weight:airframe:mass"] ** 0.764
            * inputs["data:TLAR:v_cruise"] ** 0.899
            * inputs["data:cost:num_aircraft_5years"] ** -0.756
            * f_tapered
            * f_flap
            * (1.0 + inputs["data:cost:composite_fraction"])
            * f_pressurized
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m_airframe = inputs["data:weight:airframe:mass"]
        v_cruise = inputs["data:TLAR:v_cruise"]
        num_5years = inputs["data:cost:num_aircraft_5years"]
        f_composite = inputs["data:cost:composite_fraction"]

        if self.options["complex_flap"]:
            f_flap = 1.02
        else:
            f_flap = 1.0

        if self.options["pressurized"]:
            f_pressurized = 1.01
        else:
            f_pressurized = 1.0

        if self.options["tapered_wing"]:
            f_tapered = 1.0
        else:
            f_tapered = 0.95

        partials[
            "data:cost:tooling_man_hours",
            "data:weight:airframe:mass",
        ] = (
            0.5849566
            * v_cruise**0.899
            * num_5years**-0.756
            * f_tapered
            * f_flap
            * (1.0 + f_composite)
            * f_pressurized
        ) / m_airframe**0.236

        partials[
            "data:cost:tooling_man_hours",
            "data:TLAR:v_cruise",
        ] = (
            0.68831935
            * m_airframe**0.764
            * num_5years**-0.756
            * f_tapered
            * f_flap
            * (1.0 + f_composite)
            * f_pressurized
            / v_cruise**0.101
        )

        partials[
            "data:cost:tooling_man_hours",
            "data:cost:num_aircraft_5years",
        ] = (
            -0.5788314
            * m_airframe**0.764
            * v_cruise**0.899
            * f_tapered
            * f_flap
            * (1.0 + f_composite)
            * f_pressurized
        ) / num_5years**1.756

        partials[
            "data:cost:tooling_man_hours",
            "data:cost:composite_fraction",
        ] = (
            0.76565
            * m_airframe**0.764
            * v_cruise**0.899
            * num_5years**-0.756
            * f_tapered
            * f_flap
            * f_pressurized
        )
