# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import MAX_DEFAULT_POWER


class PerformancesPEMFCStackBOPThermalPower(om.ExplicitComponent):
    """
    Total power computation of the PEMFC stack, which the thermal waste is also considered.
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
            "thermal_power", units="kW", val=np.full(number_of_points, MAX_DEFAULT_POWER)
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
        outputs["thermal_power"] = inputs["power_out"] / inputs["efficiency"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["thermal_power", "power_out"] = 1.0 / inputs["efficiency"]
        partials["thermal_power", "efficiency"] = -inputs["power_out"] / inputs["efficiency"] ** 2.0
