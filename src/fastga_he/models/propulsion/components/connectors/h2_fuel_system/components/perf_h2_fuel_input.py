# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesH2FuelSystemInput(om.ExplicitComponent):
    """
    Computation of the amount of hydrogen that the system will draw from the tanks at each point
    of the flight. The hydrogen required by the connected source is drawn from the connected
    tanks based on a user inout. The default value set considers that it is equally distributed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.hydrogen_distribution = None

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
            name="number_of_tanks",
            default=1,
            types=int,
            desc="Number of connections at the input of the hydrogen fuel system, should always be tanks",
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        number_of_tanks = self.options["number_of_tanks"]

        self.add_input(
            name="fuel_flowing_t",
            units="kg",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            desc="Fuel flowing through the hydrogen fuel system at each time step",
        )

        self.add_input(
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":fuel_distribution",
            val=np.ones(number_of_tanks),
        )

        for i in range(number_of_tanks):
            self.add_output(
                name="fuel_consumed_in_t_" + str(i + 1),
                units="kg",
                val=np.full(number_of_points, 2.5),
                shape=number_of_points,
                desc="Fuel drawn from the tank connected at the input number " + str(i + 1),
            )

            self.declare_partials(
                of="fuel_consumed_in_t_" + str(i + 1),
                wrt="fuel_flowing_t",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
            )
            self.declare_partials(
                of="fuel_consumed_in_t_" + str(i + 1),
                wrt="data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":fuel_distribution",
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        number_of_tanks = self.options["number_of_tanks"]

        #  First we rescale the distribution so that at all point it is between 0 and 1
        self.hydrogen_distribution = inputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":fuel_distribution"
        ] / np.sum(
            inputs[
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":fuel_distribution"
            ]
        )

        fuel_flow = inputs["fuel_flowing_t"]

        for i in range(number_of_tanks):
            outputs["fuel_consumed_in_t_" + str(i + 1)] = fuel_flow * self.hydrogen_distribution[i]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]
        number_of_tanks = self.options["number_of_tanks"]
        number_of_points = self.options["number_of_points"]

        fuel_flow = inputs["fuel_flowing_t"]
        scale_factor = sum(
            inputs[
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":fuel_distribution"
            ]
        )

        for i in range(number_of_tanks):
            partials["fuel_consumed_in_t_" + str(i + 1), "fuel_flowing_t"] = (
                self.hydrogen_distribution[i] * np.ones(number_of_points)
            )

            base_partials = (
                (-np.tile(fuel_flow, (number_of_tanks, 1)))
                * self.hydrogen_distribution[i]
                / scale_factor
            )
            base_partials[i, :] = fuel_flow * (1.0 - self.hydrogen_distribution[i]) / scale_factor
            partials[
                "fuel_consumed_in_t_" + str(i + 1),
                "data:propulsion:he_power_train:H2_fuel_system:"
                + h2_fuel_system_id
                + ":fuel_distribution",
            ] = np.transpose(base_partials)
