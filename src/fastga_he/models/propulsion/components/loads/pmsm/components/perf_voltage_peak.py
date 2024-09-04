# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesVoltagePeak(om.ExplicitComponent):
    """
    Computation of the peak line to neutral voltage from the RMS value of the voltage.

    Formula can be seen in :cite:`wildi:2005`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_voltage_rms_in",
            units="V",
            val=np.nan,
            shape=number_of_points,
            desc="RMS voltage at the input of the motor",
        )

        self.add_output(
            "ac_voltage_peak_in",
            units="V",
            val=np.full(number_of_points, 0.0),
            shape=number_of_points,
            desc="Peak line to neutral voltage at the input of the motor",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.full(number_of_points, np.sqrt(3.0 / 2.0)),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["ac_voltage_peak_in"] = inputs["ac_voltage_rms_in"] * np.sqrt(3.0 / 2.0)
