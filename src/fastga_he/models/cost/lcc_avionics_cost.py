# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCAvionicsCost(om.ExplicitComponent):
    """
    Computation of the avionics cost and the default cost are obtained from
    :cite:`gudmundsson:2013`.
    """

    def setup(self):
        self.add_input(
            "data:cost:cpi_2012",
            val=np.nan,
            desc="Consumer price index relative to the year 2012",
        )
        self.add_input(
            "data:cost:production:avionics",
            val=15000.0,
            desc="Default avionics cost based on the USD of 2022",
        )

        self.add_output(
            "data:cost:production:avionics_cost_per_unit",
            val=2.0e5,
            units="USD",
            desc="Avionics adjusted cost per aircraft",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:cost:production:avionics_cost_per_unit"] = (
            inputs["data:cost:production:avionics"] * inputs["data:cost:cpi_2012"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:cost:production:avionics_cost_per_unit", "data:cost:production:avionics"] = (
            inputs["data:cost:cpi_2012"]
        )

        partials["data:cost:production:avionics_cost_per_unit", "data:cost:cpi_2012"] = inputs[
            "data:cost:production:avionics"
        ]
