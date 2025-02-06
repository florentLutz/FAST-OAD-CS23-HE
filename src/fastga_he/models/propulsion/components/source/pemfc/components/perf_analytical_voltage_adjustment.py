# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesAnalyticalVoltageAdjustment(om.ExplicitComponent):
    """
    Computation of PEMFC voltage based on cathode pressure in different altitude
    from:`Preliminary Propulsion System Sizing Methods for PEM Fuel Cell Aircraft by D.Juschus:2021`
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="operation_pressure",
            units="atm",
            val=np.full(number_of_points, np.nan),
        )

        self.add_output(
            name="analytical_voltage_adjust_factor",
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
        p = inputs["operation_pressure"]
        outputs["analytical_voltage_adjust_factor"] = (
            -0.022830 * p**4 + 0.230982 * p**3 - 0.829603 * p**2 + 1.291515 * p + 0.329935
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        p = inputs["operation_pressure"]
        partials["analytical_voltage_adjust_factor", "operation_pressure"] = (
            -0.022830 * 4 * p**3 + 0.230982 * 3 * p**2 - 0.829603 * 2 * p + 1.291515
        )
