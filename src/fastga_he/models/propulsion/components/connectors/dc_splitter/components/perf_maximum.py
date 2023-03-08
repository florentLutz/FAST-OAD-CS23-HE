# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class that identifies the maximum voltage and current of the splitter in order to size it.
    """

    def initialize(self):

        self.options.declare(
            name="dc_splitter_id",
            default=None,
            desc="Identifier of the DC splitter",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        dc_splitter_id = self.options["dc_splitter_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="dc_voltage",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage of the bus",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":voltage_max",
            units="V",
            val=800.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":voltage_max",
            wrt="dc_voltage",
            method="exact",
        )

        self.add_input(
            name="dc_current_out",
            units="A",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            desc="Current going out of the bus at the output",
        )
        self.add_input(
            name="dc_current_in_1",
            units="A",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            desc="Current going in the bus at the primary input (number 1)",
        )
        self.add_input(
            name="dc_current_in_2",
            units="A",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            desc="Current going in the bus at the primary input (number 2)",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_max",
            units="A",
            val=500.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_max",
            wrt=["dc_current_in_1", "dc_current_in_2", "dc_current_out"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_splitter_id = self.options["dc_splitter_id"]
        number_of_points = self.options["number_of_points"]

        outputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":voltage_max"
        ] = np.max(inputs["dc_voltage"])

        element_wise_max_current = np.maximum(
            inputs["dc_current_in_1"],
            inputs["dc_current_in_2"],
            inputs["dc_current_out"],
        )

        outputs[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_max"
        ] = np.max(element_wise_max_current)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_splitter_id = self.options["dc_splitter_id"]
        number_of_points = self.options["number_of_points"]

        dc_voltage = inputs["dc_voltage"]

        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":voltage_max",
            "dc_voltage",
        ] = np.where(dc_voltage == np.amax(dc_voltage), 1.0, 0.0)

        element_wise_max_current = np.maximum(
            inputs["dc_current_in_1"],
            inputs["dc_current_in_2"],
            inputs["dc_current_out"],
        )
        max_current = np.max(element_wise_max_current)
        index_max_current = np.argmax(element_wise_max_current)

        partials_wrt_dc_current_in_1 = np.zeros(number_of_points)
        partials_wrt_dc_current_in_2 = np.zeros(number_of_points)
        partials_wrt_dc_current_out = np.zeros(number_of_points)

        if np.max(inputs["dc_current_in_1"]) == max_current:
            partials_wrt_dc_current_in_1[index_max_current] = 1.0
        elif np.max(inputs["dc_current_in_2"]) == max_current:
            partials_wrt_dc_current_in_2[index_max_current] = 1.0
        else:
            partials_wrt_dc_current_out[index_max_current] = 1.0

        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_max",
            "dc_current_in_1",
        ] = partials_wrt_dc_current_in_1
        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_max",
            "dc_current_in_2",
        ] = partials_wrt_dc_current_in_2
        partials[
            "data:propulsion:he_power_train:DC_splitter:" + dc_splitter_id + ":current_max",
            "dc_current_out",
        ] = partials_wrt_dc_current_out
