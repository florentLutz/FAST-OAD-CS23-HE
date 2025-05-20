# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesACPowerOut(om.ExplicitComponent):
    """
    Computation of the power of the inverter at the output side.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        # Needed even if not used because of the other submodel ...

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_current_rms_out_one_phase", units="A", val=np.full(number_of_points, np.nan)
        )
        self.add_input("ac_voltage_rms_out", units="kV", val=np.full(number_of_points, np.nan))

        self.add_output(
            "ac_power_out",
            units="kW",
            val=np.full(number_of_points, 250.0),
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # We recall here that the variable "ac_current_rms_out_one_phase" contains the rms
        # current of one arm of the inverter (see losses definition), so to get the actual
        # effective power on the AC side we need to multiply it by 3
        current = inputs["ac_current_rms_out_one_phase"]
        ac_voltage_rms_out = inputs["ac_voltage_rms_out"]

        outputs["ac_power_out"] = 3.0 * current * ac_voltage_rms_out

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        current = inputs["ac_current_rms_out_one_phase"]
        ac_voltage_rms_out = inputs["ac_voltage_rms_out"]

        partials["ac_power_out", "ac_voltage_rms_out"] = 3.0 * current
        partials["ac_power_out", "ac_current_rms_out_one_phase"] = 3.0 * ac_voltage_rms_out
