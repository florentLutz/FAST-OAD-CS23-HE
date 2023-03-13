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

        self.add_input("ac_current_rms_out", units="A", val=np.nan, shape=number_of_points)
        self.add_input("apparent_power", units="W", val=np.nan, shape=number_of_points)

        self.add_output(
            "ac_voltage_rms_out",
            units="V",
            val=np.full(number_of_points, 10.0),
            shape=number_of_points,
            desc="RMS voltage at the output of the generator",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["ac_voltage_rms_out"] = inputs["apparent_power"] / inputs["ac_current_rms_out"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["ac_voltage_rms_out", "apparent_power"] = np.diag(
            1.0 / (inputs["ac_current_rms_out"])
        )
        partials["ac_voltage_rms_out", "ac_current_rms_out"] = -np.diag(
            inputs["apparent_power"] / (inputs["ac_current_rms_out"] ** 2.0)
        )
