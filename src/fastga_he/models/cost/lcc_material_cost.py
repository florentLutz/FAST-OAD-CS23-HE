# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCMaterialCost(om.ExplicitComponent):
    """
    Computation of the raw material cost per aircraft, obtained from :cite:`gudmundsson:2013`.
    """

    def setup(self):
        self.add_input("data:weight:airframe:mass", units="kg", val=np.nan)
        self.add_input("data:TLAR:v_max_sl", units="kn", val=np.nan)
        self.add_input("data:geometry:flap_type", val=np.nan)
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
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )

        self.add_output(
            "data:cost:production:material_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Development flight test adjusted cost per aircraft",
        )

        self.declare_partials("*", "*", method="exact")
        self.declare_partials("*", "data:geometry:flap_type", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if inputs["data:geometry:flap_type"] != 0.0:
            f_flap = 1.02
        else:
            f_flap = 1.0

        outputs["data:cost:production:material_cost_per_unit"] = (
            24.896
            * inputs["data:weight:airframe:mass"] ** 0.689
            * inputs["data:TLAR:v_max_sl"] ** 0.624
            * inputs["data:cost:production:number_aircraft_5_years"] ** -0.208
            * inputs["data:cost:cpi_2012"]
            * f_flap
            * (1.0 + 0.01 * inputs["data:geometry:cabin:pressurized"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m_airframe = inputs["data:weight:airframe:mass"]
        v_cruise = inputs["data:TLAR:v_max_sl"]
        cpi_2012 = inputs["data:cost:cpi_2012"]
        num_5years = inputs["data:cost:production:number_aircraft_5_years"]
        pressurized = inputs["data:geometry:cabin:pressurized"]

        if inputs["data:geometry:flap_type"] != 0.0:
            f_flap = 1.02
        else:
            f_flap = 1.0

        partials["data:cost:production:material_cost_per_unit", "data:weight:airframe:mass"] = (
            17.153344
            * v_cruise**0.624
            * num_5years**-0.208
            * cpi_2012
            * f_flap
            * (1.0 + 0.01 * pressurized)
            / m_airframe**0.311
        )

        partials["data:cost:production:material_cost_per_unit", "data:TLAR:v_max_sl"] = (
            15.535104
            * m_airframe**0.689
            * num_5years**-0.208
            * cpi_2012
            * f_flap
            * (1.0 + 0.01 * pressurized)
            / v_cruise**0.376
        )

        partials[
            "data:cost:production:material_cost_per_unit",
            "data:cost:production:number_aircraft_5_years",
        ] = (
            -5.178368
            * m_airframe**0.689
            * v_cruise**0.624
            * cpi_2012
            * f_flap
            * (1.0 + 0.01 * pressurized)
            / num_5years**1.208
        )

        partials["data:cost:production:material_cost_per_unit", "data:cost:cpi_2012"] = (
            24.896
            * m_airframe**0.689
            * v_cruise**0.624
            * num_5years**-0.208
            * f_flap
            * (1.0 + 0.01 * pressurized)
        )

        partials[
            "data:cost:production:material_cost_per_unit", "data:geometry:cabin:pressurized"
        ] = 0.24896 * m_airframe**0.689 * v_cruise**0.624 * num_5years**-0.208 * cpi_2012 * f_flap
