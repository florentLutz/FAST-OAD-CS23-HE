# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import MAX_DEFAULT_POWER


class PerformancesPEMFCStackWasteHeat(om.ExplicitComponent):
    """
    Waste heat computation of the PEMFC stack.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("power_out", units="kW", val=np.full(number_of_points, np.nan))

        self.add_input("efficiency", val=np.full(number_of_points, np.nan))

        self.add_output("waste_heat", units="kW", val=np.full(number_of_points, 10.0))

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
        outputs["waste_heat"] = 0.5 * (inputs["power_out"] * (1.0 - inputs["efficiency"]))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["waste_heat", "power_out"] = 0.5 * (1.0 - inputs["efficiency"])
        partials["waste_heat", "efficiency"] = -0.5 * inputs["power_out"]
