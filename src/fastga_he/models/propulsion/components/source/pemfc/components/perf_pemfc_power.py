# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import MAX_DEFAULT_POWER


class PerformancesPEMFCStackPower(om.ExplicitComponent):
    """
    Output power computation of the PEMFC stack, which is only considered in post-processing.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("voltage_out", units="V", val=np.full(number_of_points, np.nan))

        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))

        self.add_output("power_out", units="kW", val=np.full(number_of_points, MAX_DEFAULT_POWER))

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["power_out"] = inputs["voltage_out"] * inputs["dc_current_out"] / 1000.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["power_out", "voltage_out"] = inputs["dc_current_out"] / 1000.0
        partials["power_out", "dc_current_out"] = inputs["voltage_out"] / 1000.0
