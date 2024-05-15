# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesVoltageRMS(om.ImplicitComponent):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("voltage_out_target", units="V", val=np.nan, shape=number_of_points)

        self.add_output(
            "ac_voltage_rms_out",
            units="V",
            val=np.full(number_of_points, 500.0),
            shape=number_of_points,
            desc="RMS voltage at the output of the generator",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        residuals["ac_voltage_rms_out"] = (
            outputs["ac_voltage_rms_out"] - inputs["voltage_out_target"]
        )

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]

        jacobian["ac_voltage_rms_out", "ac_voltage_rms_out"] = np.eye(number_of_points)
        jacobian["ac_voltage_rms_out", "voltage_out_target"] = -np.eye(number_of_points)
