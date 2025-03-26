# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDevSupportCost(om.ExplicitComponent):
    """
    Computation of the airframe cost per aircraft of development support obtained from
    :cite:`gudmundsson:2013`.
    """

    def setup(self):
        self.add_input("data:weight:airframe:mass", units="kg", val=np.nan)
        self.add_input("data:TLAR:v_cruise", units="kn", val=np.nan)
        self.add_input(
            "data:cost:airframe:prototype_number",
            val=np.nan,
            desc="number of prototypes",
        )
        self.add_input(
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )

        self.add_input(
            "data:cost:airframe:flap_factor",
            val=1.0,
            desc="Set to 1.01 for complex flap, 1.0 for simple flap",
        )

        self.add_input(
            "data:cost:airframe:composite_fraction",
            val=np.nan,
            desc="Fraction of airframe made by composite, range from 0.0 to 1.0",
        )

        self.add_input(
            "data:cost:airframe:pressurization_factor",
            val=np.nan,
            desc="Set to 1.03 for pressurized aircraft, 1.0 for unpressurized",
        )

        self.add_output(
            "data:cost:airframe:dev_support_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Development support adjusted cost per aircraft",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:airframe:dev_support_cost_per_unit"] = (
            0.06458
            * inputs["data:weight:airframe:mass"] ** 0.873
            * inputs["data:TLAR:v_cruise"] ** 1.89
            * inputs["data:cost:airframe:prototype_number"] ** 0.346
            * inputs["data:cost:cpi_2012"]
            * inputs["data:cost:airframe:flap_factor"]
            * (1.0 + 0.5 * inputs["data:cost:airframe:composite_fraction"])
            * inputs["data:cost:airframe:pressurization_factor"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m_airframe = inputs["data:weight:airframe:mass"]
        v_cruise = inputs["data:TLAR:v_cruise"]
        cpi_2012 = inputs["data:cost:cpi_2012"]
        num_prototype = inputs["data:cost:airframe:prototype_number"]
        f_flap = inputs["data:cost:airframe:flap_factor"]
        f_composite = inputs["data:cost:airframe:composite_fraction"]
        f_pressurized = inputs["data:cost:airframe:pressurization_factor"]

        partials["data:cost:airframe:dev_support_cost_per_unit", "data:weight:airframe:mass"] = (
            0.05637834
            * v_cruise**1.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * f_pressurized
            / m_airframe**0.127
        )

        partials["data:cost:airframe:dev_support_cost_per_unit", "data:TLAR:v_cruise"] = (
            0.1220562
            * v_cruise**0.873
            * v_cruise**0.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * f_pressurized
        )

        partials[
            "data:cost:airframe:dev_support_cost_per_unit", "data:cost:airframe:prototype_number"
        ] = (
            0.02234468
            * v_cruise**0.873
            * v_cruise**1.89
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * f_pressurized
            / num_prototype**0.654
        )

        partials["data:cost:airframe:dev_support_cost_per_unit", "data:cost:cpi_2012"] = (
            0.06458
            * v_cruise**0.873
            * v_cruise**1.89
            * num_prototype**0.346
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * f_pressurized
        )

        partials[
            "data:cost:airframe:dev_support_cost_per_unit", "data:cost:airframe:flap_factor"
        ] = (
            0.06458
            * v_cruise**0.873
            * v_cruise**1.89
            * num_prototype**0.346
            * cpi_2012
            * (1.0 + 0.5 * f_composite)
            * f_pressurized
        )

        partials[
            "data:cost:airframe:dev_support_cost_per_unit", "data:cost:airframe:composite_fraction"
        ] = (
            0.03229
            * v_cruise**0.873
            * v_cruise**1.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * f_pressurized
        )

        partials[
            "data:cost:airframe:dev_support_cost_per_unit",
            "data:cost:airframe:pressurization_factor",
        ] = (
            0.06458
            * v_cruise**0.873
            * v_cruise**1.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
        )
