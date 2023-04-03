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
            "ac_voltage_rms",
            units="V",
            val=np.nan,
            shape=number_of_points,
            desc="RMS voltage of the AC bus",
        )

        self.add_output(
            "ac_voltage_peak",
            units="V",
            val=np.full(number_of_points, 0.0),
            shape=number_of_points,
            desc="Peak line to neutral voltage of the AC bus",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["ac_voltage_rms"] = inputs["ac_voltage_peak"] * np.sqrt(3.0 / 2.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        partials["ac_voltage_peak", "ac_voltage_rms"] = np.sqrt(3.0 / 2.0) * np.eye(
            number_of_points
        )
