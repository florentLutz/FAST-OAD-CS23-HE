# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesVoltagePeakIn(om.ExplicitComponent):
    """
    Component which computes the value of the peak AC voltage at the input side of the rectifier.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_voltage_rms_in",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="RMS value of the voltage at the input side of the rectifier",
        )

        self.add_output(
            "ac_voltage_peak_in",
            units="V",
            val=np.full(number_of_points, 600.0),
            shape=number_of_points,
            desc="Peak line to neutral voltage at the input of the rectifier",
        )

        self.declare_partials(
            of="ac_voltage_peak_in",
            wrt="ac_voltage_rms_in",
            val=np.sqrt(3.0 / 2.0) * np.eye(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["ac_voltage_peak_in"] = np.sqrt(3.0 / 2.0) * inputs["ac_voltage_rms_in"]
