# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class that identifies the maximum voltage and current of the bus bar in order to size it.
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

        self.add_input(
            name="dc_voltage",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage of the bus",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_max",
            units="V",
            val=800.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_max",
            wrt="dc_voltage",
            method="exact",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max",
            units="A",
            val=500.0,
        )

        # For once, input are going to be after outputs, just to ensure the declaration of
        # partials goes well
        for i in range(self.options["number_of_inputs"]):
            self.add_input(
                name="dc_current_in_" + str(i + 1),
                units="A",
                val=np.full(number_of_points, np.nan),
                shape=number_of_points,
                desc="Current going into the bus at input number " + str(i + 1),
            )

            self.declare_partials(
                of="data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max",
                wrt="dc_current_in_" + str(i + 1),
                method="exact",
                rows=np.zeros(number_of_points),
                cols=np.arange(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_bus_id = self.options["dc_bus_id"]
        number_of_points = self.options["number_of_points"]

        outputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_max"] = np.max(
            inputs["dc_voltage"]
        )

        total_current = np.zeros(number_of_points)
        for i in range(self.options["number_of_inputs"]):
            total_current += inputs["dc_current_in_" + str(i + 1)]

        outputs["data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max"] = np.max(
            total_current
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_bus_id = self.options["dc_bus_id"]
        number_of_points = self.options["number_of_points"]

        dc_voltage = inputs["dc_voltage"]

        partials[
            "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":voltage_max", "dc_voltage"
        ] = np.where(dc_voltage == np.amax(dc_voltage), 1.0, 0.0)

        total_current = np.zeros(number_of_points)
        for i in range(self.options["number_of_inputs"]):
            total_current += inputs["dc_current_in_" + str(i + 1)]

        for j in range(self.options["number_of_inputs"]):
            partials[
                "data:propulsion:he_power_train:DC_bus:" + dc_bus_id + ":current_max",
                "dc_current_in_" + str(j + 1),
            ] = np.where(total_current == np.amax(total_current), 1.0, 0.0)
