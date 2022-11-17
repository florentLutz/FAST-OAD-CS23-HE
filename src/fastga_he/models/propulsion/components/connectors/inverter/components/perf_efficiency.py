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
        self.add_input(
            "ac_current_rms_out_one_phase", units="A", val=np.full(number_of_points, np.nan)
        )
        self.add_input("ac_voltage_rms_out", units="V", val=np.full(number_of_points, np.nan))

        self.add_output("efficiency", val=np.full(number_of_points, 1.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        losses_inverter = inputs["losses_inverter"]

        # We recall here that the variable "ac_current_rms_out_one_phase" contains the rms current of one arm of the
        # inverter (see losses definition), so to get the actual effective power on the AC side
        # we need to multiply it by 3
        current = inputs["ac_current_rms_out_one_phase"]
        ac_voltage_rms_out = inputs["ac_voltage_rms_out"]

        useful_power = 3.0 * current * ac_voltage_rms_out

        outputs["efficiency"] = useful_power / (useful_power + losses_inverter)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        losses_inverter = inputs["losses_inverter"]
        current = inputs["ac_current_rms_out_one_phase"]
        ac_voltage_rms_out = inputs["ac_voltage_rms_out"]

        useful_power = 3.0 * current * ac_voltage_rms_out

        partials["efficiency", "ac_voltage_rms_out"] = np.diag(
            3.0 * current * losses_inverter / (useful_power + losses_inverter) ** 2.0
        )
        partials["efficiency", "ac_current_rms_out_one_phase"] = np.diag(
            3.0 * ac_voltage_rms_out * losses_inverter / (useful_power + losses_inverter) ** 2.0
        )
        partials["efficiency", "losses_inverter"] = np.diag(
            -useful_power / (useful_power + losses_inverter) ** 2.0
        )
