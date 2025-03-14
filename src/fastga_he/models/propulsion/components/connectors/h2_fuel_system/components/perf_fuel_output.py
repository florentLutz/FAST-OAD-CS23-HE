# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesH2FuelSystemOutput(om.ExplicitComponent):
    """
    Compute the fuel that the system has to output towards power source at each point of the flight,
    is simply the sum of the fuel consumed by each power source connected at the output.
    """

    def initialize(self):
        self.options.declare(
            name="h2_fuel_system_id",
            default=None,
            desc="Identifier of the hydrogen fuel system",
            types=str,
            allow_none=False,
        )
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
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        self.add_output(
            name="fuel_flowing_t",
            units="kg",
            val=np.full(number_of_points, 5.0),
            shape=number_of_points,
            desc="Fuel flowing through the hydrogen fuel system at each time step",
        )

        self.add_input(name="time_step", units="h", val=np.full(number_of_points, np.nan))

        self.add_output(
            name="data:propulsion:he_power_train:fuel_system:"
            + h2_fuel_system_id
            + ":number_source",
            val=self.options["number_of_sources"],
            desc="Number of power source connected to this hydrogen fuel system",
        )

        for i in range(self.options["number_of_sources"]):
            # Choice was made to start current numbering at 1 to be consistent with what is done
            # on electrical node (which coincidentally should irritate programmer)
            self.add_input(
                name="fuel_consumed_out_t_" + str(i + 1),
                units="kg",
                val=np.full(number_of_points, np.nan),
                shape=number_of_points,
                desc="Fuel consumed by the power source connected at the output number "
                + str(i + 1),
            )

            self.declare_partials(
                of="fuel_flowing_t",
                wrt="fuel_consumed_out_t_" + str(i + 1),
                val=np.ones(number_of_points),
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]

        fuel_output = np.zeros(number_of_points)

        for i in range(self.options["number_of_sources"]):
            fuel_output += inputs["fuel_consumed_out_t_" + str(i + 1)]

        outputs["fuel_flowing_t"] = fuel_output
