# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesFuelOutput(om.ExplicitComponent):
    """
    Compute the fuel that the system has to output towards engine at each point of the flight,
    is simply the sum of the fuel consumed by each engine connected at the output.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, types=int, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="number_of_engines",
            default=1,
            types=int,
            desc="Number of connections at the output of the fuel system, should always be engine",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_output(
            name="fuel_flowing_t",
            units="kg",
            val=np.full(number_of_points, 5.0),
            shape=number_of_points,
            desc="Fuel flowing through the fuel system at each time step",
        )

        for i in range(self.options["number_of_engines"]):
            # Choice was made to start current numbering at 1 to be consistent with what is done
            # on electrical node (which coincidentally should irritate programmer)
            self.add_input(
                name="fuel_consumed_out_t_" + str(i + 1),
                units="kg",
                val=np.full(number_of_points, np.nan),
                shape=number_of_points,
                desc="Fuel consumed by the engine connected at the output number " + str(i + 1),
            )

            self.declare_partials(
                of="*", wrt="fuel_consumed_out_t_" + str(i + 1), val=np.eye(number_of_points)
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]

        fuel_output = np.zeros(number_of_points)

        for i in range(self.options["number_of_engines"]):
            fuel_output += inputs["fuel_consumed_out_t_" + str(i + 1)]

        outputs["fuel_flowing_t"] = fuel_output
