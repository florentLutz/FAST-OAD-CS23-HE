# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import MAX_DEFAULT_POWER


class PerformancesPEMFCStackBOPWasteHeatFlow(om.ExplicitComponent):
    """
    Waste heat flow computation of the PEMFC stack, using for BOP and TMS sizing.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("power_out", units="kW", val=np.full(number_of_points, np.nan))
        self.add_input("efficiency", val=np.full(number_of_points, np.nan))

        self.add_output(
            "waste_heat_flow", units="kW", val=np.full(number_of_points, MAX_DEFAULT_POWER * 0.5)
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["waste_heat_flow"] = inputs["power_out"] / inputs["efficiency"] * 0.35

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["waste_heat_flow", "power_out"] = 1.0 / inputs["efficiency"] * 0.35

        partials["waste_heat_flow", "efficiency"] = (
            -inputs["power_out"] / inputs["efficiency"] ** 2.0 * 0.35
        )
