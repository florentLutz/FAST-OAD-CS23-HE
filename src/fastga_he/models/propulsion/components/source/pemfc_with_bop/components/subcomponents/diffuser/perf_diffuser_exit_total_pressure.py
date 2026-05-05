# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDiffuserExitTotalPressure(om.ExplicitComponent):
    """
    Computation of the exit pressure of the diffuser.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "exit_air_speed",
            val=np.nan,
            units="m/s",
            shape=number_of_points,
        )
        self.add_input(
            "diffuser_air_density",
            val=np.nan,
            units="kg/m**3",
            shape=number_of_points,
        )
        self.add_input(
            "diffuser_exit_pressure",
            val=np.nan,
            units="Pa",
            shape=number_of_points,
        )

        self.add_output(
            "diffuser_exit_total_pressure",
            val=7000.0,
            units="Pa",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="diffuser_exit_pressure",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["diffuser_exit_total_pressure"] = (
            inputs["diffuser_exit_pressure"]
            + inputs["diffuser_air_density"] * inputs["exit_air_speed"] ** 2.0 / 2.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["diffuser_exit_total_pressure", "diffuser_air_density"] = (
            inputs["exit_air_speed"] ** 2.0 / 2.0
        )
        partials["diffuser_exit_total_pressure", "exit_air_speed"] = (
            inputs["diffuser_air_density"] * inputs["exit_air_speed"]
        )
