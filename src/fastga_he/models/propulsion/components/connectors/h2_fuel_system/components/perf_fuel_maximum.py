# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesH2FuelMaximum(om.ExplicitComponent):
    """
    Compute the maximum hydrogen mass flow rate of the flight that each source will draw from the
    tanks. This component could contain more maximum/minimum computation in the future.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, types=int, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="number_of_sources",
            default=1,
            types=int,
            desc="Number of connections at the output of the hydrogen fuel system, should always be "
            "power source",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        number_of_sources = self.options["number_of_sources"]

        for i in range(number_of_sources):
            self.add_input(
                name="fuel_consumption_out_" + str(i + 1),
                units="kg/h",
                val=np.full(number_of_points, np.nan),
                shape=number_of_points,
                desc="fuel flow rate required from the source connected at the output number "
                + str(i + 1),
            )
            self.add_output(
                name="fuel_mass_flow_rate_max_" + str(i + 1),
                units="kg/h",
                val=2.5,
                shape=number_of_points,
                desc="Maximum fuel flow rate required from the source connected at the output "
                "number " + str(i + 1),
            )
            self.declare_partials(
                of="fuel_consumption_out_" + str(i + 1),
                wrt="fuel_mass_flow_rate_max_" + str(i + 1),
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_sources = self.options["number_of_sources"]

        for i in range(number_of_sources):
            outputs["fuel_mass_flow_rate_max_" + str(i + 1)] = np.max(
                inputs["fuel_consumption_out_" + str(i + 1)]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_sources = self.options["number_of_sources"]

        for i in range(number_of_sources):
            partials[
                "fuel_mass_flow_rate_max_" + str(i + 1),
                "fuel_consumption_out_" + str(i + 1),
            ] = np.where(
                inputs["fuel_consumption_out_" + str(i + 1)]
                == np.max(inputs["fuel_consumption_out_" + str(i + 1)]),
                1.0,
                0.0,
            )
