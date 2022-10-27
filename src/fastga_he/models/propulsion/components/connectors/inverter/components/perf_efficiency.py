# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesEfficiency(om.ExplicitComponent):
    """Computation of the efficiency of the inverter."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "losses_inverter",
            units="W",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
        )
        self.add_input("current", units="A", val=np.full(number_of_points, np.nan))
        self.add_input("rms_voltage", units="V", val=np.full(number_of_points, np.nan))

        self.add_output("efficiency", val=np.full(number_of_points, 1.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        losses_inverter = inputs["losses_inverter"]

        # We recall here that the variable "current" contains the rms current of one arm of the
        # inverter (see losses definition), so to get the actual effective power on the AC side
        # we need to multiply it by 3
        current = inputs["current"]
        rms_voltage = inputs["rms_voltage"]

        useful_power = 3.0 * current * rms_voltage

        outputs["efficiency"] = useful_power / (useful_power + losses_inverter)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        losses_inverter = inputs["losses_inverter"]
        current = inputs["current"]
        rms_voltage = inputs["rms_voltage"]

        useful_power = 3.0 * current * rms_voltage

        partials["efficiency", "rms_voltage"] = np.diag(
            3.0 * current * losses_inverter / (useful_power + losses_inverter) ** 2.0
        )
        partials["efficiency", "current"] = np.diag(
            3.0 * rms_voltage * losses_inverter / (useful_power + losses_inverter) ** 2.0
        )
        partials["efficiency", "losses_inverter"] = np.diag(
            -useful_power / (useful_power + losses_inverter) ** 2.0
        )
