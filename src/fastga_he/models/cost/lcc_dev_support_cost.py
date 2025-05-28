# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDevSupportCost(om.ExplicitComponent):
    """
    Computation of the airframe development support cost as obtained from :cite:`gudmundsson:2013`.
    """

    def setup(self):
        self.add_input("data:weight:airframe:mass", units="kg", val=np.nan)
        self.add_input("data:TLAR:v_max_sl", units="kn", val=np.nan)
        self.add_input("data:geometry:flap_type", val=np.nan)
        self.add_input(
            "data:cost:prototype_number",
            val=np.nan,
            desc="number of prototypes",
        )
        self.add_input(
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )
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
            desc="Fraction of airframe made of composite, range from 0.0 to 1.0",
        )

        self.add_output(
            "data:cost:production:dev_support_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Development support adjusted cost per aircraft",
        )

        self.declare_partials("*", "*", method="exact")
        self.declare_partials("*", "data:geometry:flap_type", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if inputs["data:geometry:flap_type"] != 0.0:
            f_flap = 1.01
        else:
            f_flap = 1.0

        outputs["data:cost:production:dev_support_cost_per_unit"] = (
            0.06458
            * inputs["data:weight:airframe:mass"] ** 0.873
            * inputs["data:TLAR:v_max_sl"] ** 1.89
            * inputs["data:cost:prototype_number"] ** 0.346
            * inputs["data:cost:cpi_2012"]
            * f_flap
            * (1.0 + 0.5 * inputs["data:cost:production:composite_fraction"])
            * (1.0 + 0.03 * inputs["data:geometry:cabin:pressurized"])
            / inputs["data:cost:production:number_aircraft_5_years"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m_airframe = inputs["data:weight:airframe:mass"]
        v_cruise = inputs["data:TLAR:v_max_sl"]
        cpi_2012 = inputs["data:cost:cpi_2012"]
        num_prototype = inputs["data:cost:prototype_number"]
        num_5years = inputs["data:cost:production:number_aircraft_5_years"]
        f_composite = inputs["data:cost:production:composite_fraction"]
        pressurized = inputs["data:geometry:cabin:pressurized"]

        if inputs["data:geometry:flap_type"] != 0.0:
            f_flap = 1.01
        else:
            f_flap = 1.0

        partials["data:cost:production:dev_support_cost_per_unit", "data:weight:airframe:mass"] = (
            0.05637834
            * m_airframe**-0.127
            * v_cruise**1.89
            * num_prototype**0.346
            * cpi_2012
            * (1.0 + 0.01 * inputs["data:geometry:flap_type"])
            * (1.0 + 0.5 * f_composite)
            * (1.0 + 0.03 * pressurized)
            / num_5years
        )

        partials["data:cost:production:dev_support_cost_per_unit", "data:TLAR:v_max_sl"] = (
            0.1220562
            * m_airframe**0.873
            * v_cruise**0.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * (1.0 + 0.03 * pressurized)
            / num_5years
        )

        partials["data:cost:production:dev_support_cost_per_unit", "data:cost:prototype_number"] = (
            0.02234468
            * m_airframe**0.873
            * v_cruise**1.89
            * num_prototype**-0.654
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * (1.0 + 0.03 * pressurized)
            / num_5years
        )

        partials["data:cost:production:dev_support_cost_per_unit", "data:cost:cpi_2012"] = (
            0.06458
            * m_airframe**0.873
            * v_cruise**1.89
            * num_prototype**0.346
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * (1.0 + 0.03 * pressurized)
            / num_5years
        )

        partials[
            "data:cost:production:dev_support_cost_per_unit",
            "data:cost:production:composite_fraction",
        ] = (
            0.03229
            * m_airframe**0.873
            * v_cruise**1.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * (1.0 + 0.03 * pressurized)
            / num_5years
        )

        partials[
            "data:cost:production:dev_support_cost_per_unit",
            "data:cost:production:number_aircraft_5_years",
        ] = (
            -0.06458
            * m_airframe**0.873
            * v_cruise**1.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * (1.0 + 0.03 * pressurized)
            / num_5years**2.0
        )

        partials[
            "data:cost:production:dev_support_cost_per_unit",
            "data:geometry:cabin:pressurized",
        ] = (
            0.0019374
            * m_airframe**0.873
            * v_cruise**1.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
            / num_5years
        )
