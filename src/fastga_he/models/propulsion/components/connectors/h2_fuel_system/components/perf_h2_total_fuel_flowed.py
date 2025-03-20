# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesTotalH2FuelFlowed(om.ExplicitComponent):
    """
    Computation of the total amount of fuel which has flown through the hydrogen fuel system.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="h2_fuel_system_id",
            default=None,
            desc="Identifier of the hydrogen fuel system",
            types=str,
            allow_none=False,
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        self.add_input(
            "fuel_flowing_t",
            units="kg",
            val=np.full(number_of_points, np.nan),
            desc="Fuel flowing through the hydrogen fuel system at each time step",
        )

        self.add_output(
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":total_fuel_flowed",
            units="kg",
            val=50.0,
            desc="Total amount of fuel that flowed through the system",
        )

        self.declare_partials(
            of="*",
            wrt="*",
            val=np.ones(number_of_points),
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h2_fuel_system_id = self.options["h2_fuel_system_id"]

        outputs[
            "data:propulsion:he_power_train:H2_fuel_system:"
            + h2_fuel_system_id
            + ":total_fuel_flowed"
        ] = sum(inputs["fuel_flowing_t"])
