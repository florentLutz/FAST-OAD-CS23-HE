# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesModulationIndex(om.ImplicitComponent):
    """
    Computation of the modulation index, left as an implicit component to leave a degree of
    freedom to the code to converge.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_voltage_peak_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Peak voltage on the AC side of the inverter",
        )
        self.add_input(
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Bus voltage on the DC side of the inverter",
        )

        self.add_output(
            "modulation_index", val=np.full(number_of_points, 0.95), lower=0.0, upper=2.0
        )

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        residuals["modulation_index"] = (
            inputs["ac_voltage_peak_out"] - outputs["modulation_index"] * inputs["dc_voltage_in"]
        )

    def linearize(self, inputs, outputs, jacobian, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]

        jacobian["modulation_index", "ac_voltage_peak_out"] = np.ones(number_of_points)
        jacobian["modulation_index", "dc_voltage_in"] = -outputs["modulation_index"]
        jacobian["modulation_index", "modulation_index"] = -inputs["dc_voltage_in"]

    def guess_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        if any(
            np.clip(inputs["ac_voltage_peak_out"] / inputs["dc_voltage_in"], 0, 1.0)
            != inputs["ac_voltage_peak_out"] / inputs["dc_voltage_in"]
        ):

            outputs["modulation_index"] = np.clip(
                inputs["ac_voltage_peak_out"] / inputs["dc_voltage_in"], 0, 1.0
            )
