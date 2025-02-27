# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesPEMFCStackVoltageAdjustment(om.ExplicitComponent):
    """
    Computation of the single-layered PEMFC voltage correction factor bases on ambient pressure in
    different altitude, obtained from :cite:`juschus:2021`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="ambient_pressure",
            units="atm",
            val=np.full(number_of_points, np.nan),
        )

        self.add_output(
            name="ambient_pressure_voltage_correction",
            val=np.full(number_of_points, 1.0),
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        p = inputs["ambient_pressure"]
        outputs["ambient_pressure_voltage_correction"] = (
            -0.022830 * p**4 + 0.230982 * p**3 - 0.829603 * p**2 + 1.291515 * p + 0.329935
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        p = inputs["ambient_pressure"]
        partials["ambient_pressure_voltage_correction", "ambient_pressure"] = (
            -0.09132 * p**3 + 0.692946 * p**2 - 1.659206 * p + 1.291515
        )
