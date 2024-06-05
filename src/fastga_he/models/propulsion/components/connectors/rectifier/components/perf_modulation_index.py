# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesModulationIndex(om.ImplicitComponent):
    """
    Component which computes the value of the modulation index of the rectifier.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_voltage_peak_in",
            units="V",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            desc="Peak line to neutral voltage at the input of the rectifier",
        )
        self.add_input(
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the output side of the rectifier",
        )

        self.add_output(
            "modulation_index",
            val=np.full(number_of_points, 1.0),
            desc="Modulation index of the rectifier",
        )

        self.declare_partials(
            of="modulation_index",
            wrt="ac_voltage_peak_in",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=np.ones(number_of_points),
        )
        self.declare_partials(
            of="modulation_index",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def apply_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):

        residuals["modulation_index"] = (
            inputs["ac_voltage_peak_in"] - outputs["modulation_index"] * inputs["dc_voltage_out"]
        )

    def linearize(self, inputs, outputs, partials):

        partials["modulation_index", "dc_voltage_out"] = -outputs["modulation_index"]
        partials["modulation_index", "modulation_index"] = -inputs["dc_voltage_out"]

    def guess_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        if any(
            np.clip(inputs["ac_voltage_peak_in"] / inputs["dc_voltage_out"], 0, 1.0)
            != inputs["ac_voltage_peak_in"] / inputs["dc_voltage_out"]
        ):
            outputs["modulation_index"] = np.clip(
                inputs["ac_voltage_peak_in"] / inputs["dc_voltage_out"], 0, 1.0
            )
