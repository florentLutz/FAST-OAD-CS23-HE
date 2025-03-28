# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCQualityControlCost(om.ExplicitComponent):
    """
    Computation of the airframe cost per aircraft on quality control obtained from
    :cite:`gudmundsson:2013`.
    """

    def setup(self):
        self.add_input(
            "data:cost:airframe:manufacturing_cost_per_unit",
            val=np.nan,
            units="USD",
            desc="Manufacturing adjusted cost per aircraft",
        )
        self.add_input(
            "data:cost:airframe:composite_fraction",
            val=0.0,
            desc="Fraction of airframe made by composite, range from 0.0 to 1.0",
        )

        self.add_output(
            "data:cost:airframe:quality_control_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Quality control adjusted cost per aircraft",
        )
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:airframe:quality_control_cost_per_unit"] = (
            0.13
            * inputs["data:cost:airframe:manufacturing_cost_per_unit"]
            * (1.0 + 0.5 * inputs["data:cost:airframe:composite_fraction"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:cost:airframe:quality_control_cost_per_unit",
            "data:cost:airframe:manufacturing_man_hours",
        ] = 0.13 * (1.0 + 0.5 * inputs["data:cost:airframe:composite_fraction"])

        partials[
            "data:cost:airframe:quality_control_cost_per_unit",
            "data:cost:airframe:manufacturing_cost_per_hour",
        ] = 0.065 * inputs["data:cost:airframe:manufacturing_cost_per_unit"]
