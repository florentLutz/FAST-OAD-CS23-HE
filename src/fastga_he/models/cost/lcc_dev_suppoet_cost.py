# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCDevSupportCost(om.ExplicitComponent):
    """
    Computation of the airframe cost of development support obtained from :cite:`gudmundsson:2013`.
    """

    def initialize(self):
        self.options.declare(
            name="complex_flap",
            default=False,
            types=bool,
            desc="True if complex flap system is selected in design",
        )
        self.options.declare(
            name="pressurized",
            default=False,
            types=bool,
            desc="True if the aircraft is pressurized",
        )

    def setup(self):
        self.add_input("data:weight:airframe:mass", units="kg", val=np.nan)
        self.add_input("data:TLAR:v_cruise", units="kn", val=np.nan)
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
            "data:cost:dev_support_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Development support adjusted cost per aircraft",
        )
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options["complex_flap"]:
            f_flap = 1.01
        else:
            f_flap = 1.0

        if self.options["pressurized"]:
            f_pressurized = 1.03
        else:
            f_pressurized = 1.0

        outputs["data:cost:dev_support_cost_per_unit"] = (
            0.06458
            * inputs["data:weight:airframe:mass"] ** 0.873
            * inputs["data:TLAR:v_cruise"] ** 1.89
            * inputs["data:cost:prototype_number"] ** 0.346
            * inputs["data:cost:cpi_2012"]
            * f_flap
            * (1.0 + 0.5 * inputs["data:cost:composite_fraction"])
            * f_pressurized
            / inputs["data:cost:num_aircraft_5years"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m_airframe = inputs["data:weight:airframe:mass"]
        v_cruise = inputs["data:TLAR:v_cruise"]
        cpi_2012 = inputs["data:cost:cpi_2012"]
        num_prototype = inputs["data:cost:prototype_number"]
        num_5years = inputs["data:cost:num_aircraft_5years"]
        f_composite = inputs["data:cost:composite_fraction"]

        if self.options["complex_flap"]:
            f_flap = 1.01
        else:
            f_flap = 1.0

        if self.options["pressurized"]:
            f_pressurized = 1.03
        else:
            f_pressurized = 1.0

        partials["data:cost:dev_support_cost_per_unit", "data:weight:airframe:mass"] = (
            0.05637834
            * m_airframe**-0.127
            * v_cruise**1.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * f_pressurized
            / num_5years
        )

        partials["data:cost:dev_support_cost_per_unit", "data:TLAR:v_cruise"] = (
            0.1220562
            * m_airframe**0.873
            * v_cruise**0.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * f_pressurized
            / num_5years
        )

        partials["data:cost:dev_support_cost_per_unit", "data:cost:prototype_number"] = (
            0.02234468
            * m_airframe**0.873
            * v_cruise**1.89
            * num_prototype**-0.654
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * f_pressurized
            / num_5years
        )

        partials["data:cost:dev_support_cost_per_unit", "data:cost:cpi_2012"] = (
            0.06458
            * m_airframe**0.873
            * v_cruise**1.89
            * num_prototype**0.346
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * f_pressurized
            / num_5years
        )

        partials["data:cost:dev_support_cost_per_unit", "data:cost:composite_fraction"] = (
            0.03229
            * m_airframe**0.873
            * v_cruise**1.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * f_pressurized
            / num_5years
        )

        partials[
            "data:cost:dev_support_cost_per_unit",
            "data:cost:num_aircraft_5years",
        ] = (
            -0.06458
            * m_airframe**0.873
            * v_cruise**1.89
            * num_prototype**0.346
            * cpi_2012
            * f_flap
            * (1.0 + 0.5 * f_composite)
            * f_pressurized
            / num_5years**2.0
        )
