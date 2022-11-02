# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesMaximumCurrent(om.ExplicitComponent):
    """
    Class that identifies the maximum current flowing through the bus bar in order to size it.
    Based on a simple addition of the current going in.
    """

    def initialize(self):

        self.options.declare(
            name="dc_bus_id",
            default=None,
            desc="Identifier of the DC bus",
            types=str,
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, types=int, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="number_of_inputs",
            default=1,
            types=int,
            desc="Number of connections at the input of the bus",
            allow_none=False,
        )

    def setup(self):

        dc_bus_id = self.options["dc_bus_id"]
        number_of_points = self.options["number_of_points"]

        for i in range(self.options["number_of_inputs"]):
            self.add_input(
                name="current_in_" + str(i + 1),
                units="A",
                val=np.full(number_of_points, np.nan),
                shape=number_of_points,
                desc="Current going into the bus at input number " + str(i + 1),
            )

        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max",
            units="A",
            val=500.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_bus_id = self.options["dc_bus_id"]
        number_of_points = self.options["number_of_points"]

        total_current = np.zeros(number_of_points)
        for i in range(self.options["number_of_inputs"]):
            total_current += inputs["current_in_" + str(i + 1)]

        outputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max"] = np.max(
            total_current
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_bus_id = self.options["dc_bus_id"]
        number_of_points = self.options["number_of_points"]

        total_current = np.zeros(number_of_points)
        for i in range(self.options["number_of_inputs"]):
            total_current += inputs["current_in_" + str(i + 1)]

        idx_max_current = np.argmax(total_current)
        partials_flat = np.zeros(number_of_points)
        partials_flat[idx_max_current] = 1.0

        for i in range(self.options["number_of_inputs"]):
            partials[
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max",
                "current_in_" + str(i + 1),
            ] = partials_flat
