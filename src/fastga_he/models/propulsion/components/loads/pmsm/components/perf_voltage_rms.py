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

        self.add_input("rms_current", units="A", val=np.nan, shape=number_of_points)
        self.add_input("apparent_power", units="W", val=np.nan, shape=number_of_points)

        self.add_output(
            "rms_voltage",
            units="V",
            val=np.full(number_of_points, 10.0),
            shape=number_of_points,
            desc="RMS voltage at the input of the motor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["rms_voltage"] = inputs["apparent_power"] / (3.0 * inputs["rms_current"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["rms_voltage", "apparent_power"] = np.diag(1.0 / (3.0 * inputs["rms_current"]))
        partials["rms_voltage", "rms_current"] = -np.diag(
            inputs["apparent_power"] / (3.0 * inputs["rms_current"] ** 2.0)
        )
