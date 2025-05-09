# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCToolingManHours(om.ExplicitComponent):
    """
    Compute the number of man-hours required in tooling labor, obtained from
    :cite:`gudmundsson:2013`.
    """

    def initialize(self):
        self.options.declare(
            name="tapered_wing",
            default=False,
            types=bool,
            desc="True if the aircraft has tapered wing",
        )

    def setup(self):
        self.add_input("data:weight:airframe:mass", units="kg", val=np.nan)
        self.add_input("data:cost:v_cruise_design", units="kn", val=np.nan)
        self.add_input("data:geometry:flap_type", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=1.0)
        self.add_input(
            "data:cost:production:number_aircraft_5_years",
            val=np.nan,
            desc="Number of planned aircraft to be produced over a 5-year period or 60 months",
        )
        self.add_input(
            "data:geometry:cabin:pressurized",
            val=0.0,
            desc="Cabin pressurization; 0.0 for no pressurization, 1.0 for pressurization",
        )
        self.add_input(
            "data:cost:production:composite_fraction",
            val=0.0,
            desc="Fraction of airframe made by composite, range from 0.0 to 1.0",
        )

        self.add_output(
            "data:cost:production:tooling_man_hours",
            val=2.0e5,
            units="h",
            desc="Number of tooling man-hours required per aircraft",
        )

        self.declare_partials(of="*", wrt="*", method="exact")
        self.declare_partials(
            "*", ["data:geometry:flap_type", "data:geometry:wing:taper_ratio"], method="fd"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if inputs["data:geometry:flap_type"] != 0.0:
            f_flap = 1.02
        else:
            f_flap = 1.0

        if inputs["data:geometry:wing:taper_ratio"] != 1.0:
            f_tapered = 1.0
        else:
            f_tapered = 0.95

        outputs["data:cost:production:tooling_man_hours"] = (
            0.76565
            * inputs["data:weight:airframe:mass"] ** 0.764
            * inputs["data:cost:v_cruise_design"] ** 0.899
            * inputs["data:cost:production:number_aircraft_5_years"] ** -0.756
            * f_tapered
            * f_flap
            * (1.0 + inputs["data:cost:production:composite_fraction"])
            * (1.0 + 0.01 * inputs["data:geometry:cabin:pressurized"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m_airframe = inputs["data:weight:airframe:mass"]
        v_cruise = inputs["data:cost:v_cruise_design"]
        num_5years = inputs["data:cost:production:number_aircraft_5_years"]
        f_composite = inputs["data:cost:production:composite_fraction"]
        pressurized = inputs["data:geometry:cabin:pressurized"]

        if inputs["data:geometry:flap_type"] != 0.0:
            f_flap = 1.02
        else:
            f_flap = 1.0

        if inputs["data:geometry:wing:taper_ratio"] != 1.0:
            f_tapered = 1.0
        else:
            f_tapered = 0.95

        partials[
            "data:cost:production:tooling_man_hours",
            "data:weight:airframe:mass",
        ] = (
            0.5849566
            * v_cruise**0.899
            * num_5years**-0.756
            * f_tapered
            * f_flap
            * (1.0 + f_composite)
            * (1.0 + 0.01 * pressurized)
        ) / m_airframe**0.236

        partials[
            "data:cost:production:tooling_man_hours",
            "data:cost:v_cruise_design",
        ] = (
            0.68831935
            * m_airframe**0.764
            * num_5years**-0.756
            * f_tapered
            * f_flap
            * (1.0 + f_composite)
            * (1.0 + 0.01 * pressurized)
            / v_cruise**0.101
        )

        partials[
            "data:cost:production:tooling_man_hours",
            "data:cost:production:number_aircraft_5_years",
        ] = (
            -0.5788314
            * m_airframe**0.764
            * v_cruise**0.899
            * f_tapered
            * f_flap
            * (1.0 + f_composite)
            * (1.0 + 0.01 * pressurized)
        ) / num_5years**1.756

        partials[
            "data:cost:production:tooling_man_hours",
            "data:cost:production:composite_fraction",
        ] = (
            0.76565
            * m_airframe**0.764
            * v_cruise**0.899
            * num_5years**-0.756
            * f_tapered
            * f_flap
            * (1.0 + 0.01 * pressurized)
        )

        partials[
            "data:cost:production:tooling_man_hours",
            "data:geometry:cabin:pressurized",
        ] = (
            0.0076565
            * m_airframe**0.764
            * v_cruise**0.899
            * num_5years**-0.756
            * f_tapered
            * f_flap
            * (1.0 + f_composite)
        )
