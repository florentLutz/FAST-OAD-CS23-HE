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
            "rms_voltage",
            units="V",
            val=np.full(number_of_points, np.nan),
            desc="RMS voltage on the AC side of the inverter",
        )
        self.add_input(
            "dc_voltage",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Bus voltage on the DC side of the inverter",
        )
        self.add_input(
            "current",
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
            "dc_current",
            val=np.full(number_of_points, 150.0),
            units="A",
            desc="Current coming in from the DC side of the inverter",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def apply_nonlinear(self, inputs, outputs, residuals):

        residuals["dc_current"] = (
            outputs["dc_current"] * inputs["dc_voltage"] * inputs["efficiency"]
            - 3.0 * inputs["current"] * inputs["rms_voltage"]
        )

    def linearize(self, inputs, outputs, partials):

        partials["dc_current", "dc_current"] = np.diag(inputs["dc_voltage"] * inputs["efficiency"])
        partials["dc_current", "dc_voltage"] = np.diag(outputs["dc_current"] * inputs["efficiency"])
        partials["dc_current", "efficiency"] = np.diag(outputs["dc_current"] * inputs["dc_voltage"])
        partials["dc_current", "current"] = np.diag(-3.0 * inputs["rms_voltage"])
        partials["dc_current", "rms_voltage"] = np.diag(-3.0 * inputs["current"])
