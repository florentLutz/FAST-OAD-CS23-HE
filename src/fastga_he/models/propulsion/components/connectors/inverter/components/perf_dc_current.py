# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesDCCurrent(om.ImplicitComponent):
    """
    Computation of the current needed to be supplied by the DC side of the inverter, left as an
    implicit component to let more degree of freedom to the solver.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_voltage_rms_out",
            units="V",
            val=np.full(number_of_points, np.nan),
            desc="RMS voltage on the AC side of the inverter",
        )
        self.add_input(
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Bus voltage on the DC side of the inverter",
        )
        self.add_input(
            "ac_current_rms_out_one_phase",
            units="A",
            val=np.full(number_of_points, np.nan),
            desc="RMS current in one arm of the inverter",
        )
        self.add_input(
            "efficiency",
            val=np.full(number_of_points, np.nan),
            desc="Efficiency of the inverter",
        )

        self.add_output(
            "dc_current_in",
            val=np.full(number_of_points, 150.0),
            units="A",
            desc="Current coming in from the DC side of the inverter",
            lower=-1000.0,
            upper=1000.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(self, inputs, outputs, residuals):

        residuals["dc_current_in"] = (
            outputs["dc_current_in"] * inputs["dc_voltage_in"] * inputs["efficiency"]
            - 3.0 * inputs["ac_current_rms_out_one_phase"] * inputs["ac_voltage_rms_out"]
        )

    def linearize(self, inputs, outputs, partials):

        partials["dc_current_in", "dc_current_in"] = np.diag(
            inputs["dc_voltage_in"] * inputs["efficiency"]
        )
        partials["dc_current_in", "dc_voltage_in"] = np.diag(
            outputs["dc_current_in"] * inputs["efficiency"]
        )
        partials["dc_current_in", "efficiency"] = np.diag(
            outputs["dc_current_in"] * inputs["dc_voltage_in"]
        )
        partials["dc_current_in", "ac_current_rms_out_one_phase"] = np.diag(
            -3.0 * inputs["ac_voltage_rms_out"]
        )
        partials["dc_current_in", "ac_voltage_rms_out"] = np.diag(
            -3.0 * inputs["ac_current_rms_out_one_phase"]
        )
