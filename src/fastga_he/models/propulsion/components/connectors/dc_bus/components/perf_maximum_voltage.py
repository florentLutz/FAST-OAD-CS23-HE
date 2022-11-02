# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesMaximumVoltage(om.ExplicitComponent):
    """
    Class that identifies the maximum voltage of the bus bar in order to size it.
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

    def setup(self):

        dc_bus_id = self.options["dc_bus_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="voltage",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage of the " "bus",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_max",
            units="V",
            val=800.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_bus_id = self.options["dc_bus_id"]

        outputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_max"] = np.max(
            inputs["voltage"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_bus_id = self.options["dc_bus_id"]
        number_of_points = self.options["number_of_points"]

        idx_max_voltage = np.argmax(inputs["voltage"])
        partials_flat = np.zeros(number_of_points)
        partials_flat[idx_max_voltage] = 1.0

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_max", "voltage"
        ] = partials_flat
