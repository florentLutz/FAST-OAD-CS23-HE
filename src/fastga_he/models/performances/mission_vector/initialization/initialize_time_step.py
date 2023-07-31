# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class InitializeTimeStep(om.ExplicitComponent):
    """Computes the time step size for the energy consumption later."""

    def initialize(self):

        self.options.declare(
            "number_of_points_climb", default=1, desc="number of equilibrium to be treated in climb"
        )
        self.options.declare(
            "number_of_points_cruise",
            default=1,
            desc="number of equilibrium to be treated in " "cruise",
        )
        self.options.declare(
            "number_of_points_descent",
            default=1,
            desc="number of equilibrium to be treated in " "descen",
        )
        self.options.declare(
            "number_of_points_reserve",
            default=1,
            desc="number of equilibrium to be treated in reserve",
        )

    def setup(self):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        self.add_input(
            "time", val=np.full(number_of_points, np.nan), shape=number_of_points, units="s"
        )

        self.add_output("time_step", shape=number_of_points, units="s")

        self.declare_partials(of="time_step", wrt="time", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]

        time = inputs["time"]

        time_step = time[1:] - time[:-1]
        time_step = np.append(time_step, time_step[-1])

        # Here we have to do an additional change. Since time step is computed for point i based
        # on time(i+1) - time(i) the last time step of climb will be computed with the first time
        # of cruise which means, since the cruise time step is very wide, that it will be very
        # wide and lead to an overestimation of climb fuel. For this reason we will replace the
        # last time step of climb with the precedent to get a good estimate. This will only serve
        # for the energy consumption calculation. Same remark holds for the end of descent and
        # start of reserve.

        time_step[number_of_points_climb - 1] = time_step[number_of_points_climb - 2]
        time_step[
            number_of_points_climb + number_of_points_cruise + number_of_points_descent - 1
        ] = time_step[
            number_of_points_climb + number_of_points_cruise + number_of_points_descent - 2
        ]

        outputs["time_step"] = time_step

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        middle_diagonal = -np.eye(number_of_points)
        upper_diagonal = np.diagflat(np.full(number_of_points - 1, 1), 1)
        d_ts_dt = middle_diagonal + upper_diagonal
        d_ts_dt[-1, -1] = 1.0
        d_ts_dt[-1, -2] = -1.0

        # Then we correct for the last point of climb and the last point of descent
        d_ts_dt[number_of_points_climb - 1, :] = d_ts_dt[number_of_points_climb - 2, :]
        d_ts_dt[
            number_of_points_climb + number_of_points_cruise + number_of_points_descent - 1, :
        ] = d_ts_dt[
            number_of_points_climb + number_of_points_cruise + number_of_points_descent - 2, :
        ]

        partials["time_step", "time"] = d_ts_dt
