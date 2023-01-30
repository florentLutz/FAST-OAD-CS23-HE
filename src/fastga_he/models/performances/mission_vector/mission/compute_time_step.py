# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class ComputeTimeStep(om.ExplicitComponent):
    """Computes the time step size for the energy consumption later."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input(
            "time", val=np.full(number_of_points, np.nan), shape=number_of_points, units="s"
        )

        self.add_output("time_step", shape=number_of_points, units="s")

    def setup_partials(self):

        self.declare_partials(of="time_step", wrt="time", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        time = inputs["time"]

        time_step = time[1:] - time[:-1]
        time_step = np.append(time_step, time_step[-1])

        outputs["time_step"] = time_step

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        middle_diagonal = -np.eye(number_of_points)
        upper_diagonal = np.diagflat(np.full(number_of_points - 1, 1), 1)
        d_ts_dt = middle_diagonal + upper_diagonal
        d_ts_dt[-1, -1] = 1.0
        d_ts_dt[-1, -2] = -1.0
        partials["time_step", "time"] = d_ts_dt
