# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesDutyCycle(om.ExplicitComponent):
    """
    Computation of the duty cycle of the converter for further use in the efficiency computation.

    Based on the methodology from :cite:`hairik:2019`.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the input side of the converter",
        )
        self.add_input(
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage to output side",
        )

        self.add_output(
            "duty_cycle",
            val=np.full(number_of_points, 0.5),
            desc="Duty cycle of the converter",
            lower=np.full(number_of_points, 0.0),
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        outputs["duty_cycle"] = inputs["dc_voltage_out"] / (
            inputs["dc_voltage_out"] + inputs["dc_voltage_in"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        partials["duty_cycle", "dc_voltage_out"] = np.diag(
            inputs["dc_voltage_in"] / (inputs["dc_voltage_out"] + inputs["dc_voltage_in"]) ** 2.0
        )
        partials["duty_cycle", "dc_voltage_in"] = -np.diag(
            inputs["dc_voltage_out"] / (inputs["dc_voltage_out"] + inputs["dc_voltage_in"]) ** 2.0
        )
