# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDCSSPCPower(om.ExplicitComponent):
    """
    This module computes the power flowing through the SSPC at each point of the flight. Simple
    multiplication of the current by the greatest of input/output power. The result of this
    module will be used in the computation of the efficiency.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "dc_current_in",
            val=np.full(number_of_points, np.nan),
            units="A",
        )
        self.add_input(
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
        )
        self.add_input(
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
        )

        self.add_output(
            "power_flow",
            val=np.full(number_of_points, 500e3),
            units="W",
            desc="Power at the terminal of the SSPC with the greatest voltage",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        voltage = np.maximum(inputs["dc_voltage_in"], inputs["dc_voltage_out"])

        outputs["power_flow"] = voltage * inputs["dc_current_in"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        voltage_in = inputs["dc_voltage_in"]
        voltage_out = inputs["dc_voltage_out"]
        voltage = np.maximum(voltage_in, voltage_out)

        partials["power_flow", "dc_current_in"] = voltage
        partials["power_flow", "dc_voltage_in"] = (
            np.where(voltage == voltage_in, 1.0, 0.0) * inputs["dc_current_in"]
        )

        partials["power_flow", "dc_voltage_out"] = (
            np.where(voltage == voltage_out, 1.0, 0.0) * inputs["dc_current_in"]
        )
