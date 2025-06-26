# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesVoltageRMS(om.ExplicitComponent):
    """
    Computation of the RMS of the voltage from the RMS of the current and apparent power.

    Formula can be seen in :cite:`wildi:2005`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("ac_current_rms_in", units="A", val=np.nan, shape=number_of_points)
        self.add_input("apparent_power", units="W", val=np.nan, shape=number_of_points)

        self.add_output(
            "ac_voltage_rms_in",
            units="V",
            val=np.full(number_of_points, 10.0),
            shape=number_of_points,
            desc="RMS voltage at the input of the motor",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Ones_like or Zeros_like ?
        outputs["ac_voltage_rms_in"] = np.where(
            inputs["ac_current_rms_in"] != 0.0,
            inputs["apparent_power"] / inputs["ac_current_rms_in"],
            np.ones_like(inputs["ac_current_rms_in"]),
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["ac_voltage_rms_in", "apparent_power"] = np.where(
            inputs["ac_current_rms_in"] != 0.0,
            1.0 / (inputs["ac_current_rms_in"]),
            np.full_like(inputs["ac_current_rms_in"], 1e-6),
        )
        partials["ac_voltage_rms_in", "ac_current_rms_in"] = -(
            np.where(
                inputs["ac_current_rms_in"] != 0.0,
                inputs["apparent_power"] / (inputs["ac_current_rms_in"] ** 2.0),
                np.full_like(inputs["ac_current_rms_in"], -1e-6),
            )
        )
