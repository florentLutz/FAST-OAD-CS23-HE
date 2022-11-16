# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="harness_id",
            default=None,
            desc="Identifier of the cable harness",
            allow_none=False,
        )

    def setup(self):

        harness_id = self.options["harness_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the expected output (current flows from source/input to load/output)",
            shape=number_of_points,
        )
        self.add_input(
            "voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the expected input (current flows from source/input to load/output)",
            shape=number_of_points,
        )
        self.add_input(
            "total_current",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="current of the harness",
            shape=number_of_points,
        )
        self.add_input(
            "cable_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
            val=1000.0,
            units="A",
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
            val=800.0,
            units="V",
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:"
            + harness_id
            + ":temperature_max",
            val=288.15,
            units="degK",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
            wrt="total_current",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
            wrt=["voltage_out", "voltage_in"],
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":temperature_max",
            wrt=["cable_temperature"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max"
        ] = np.amax(inputs["total_current"])

        voltage_max = np.maximum(inputs["voltage_out"], inputs["voltage_in"])

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max"
        ] = np.amax(voltage_max)

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":temperature_max"
        ] = np.amax(inputs["cable_temperature"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        current = inputs["total_current"]
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
            "total_current",
        ] = np.where(current == np.amax(current), 1.0, 0.0)

        voltage_out = inputs["voltage_out"]
        voltage_in = inputs["voltage_in"]
        if np.amax(voltage_in) > np.amax(voltage_out):
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
                "voltage_out",
            ] = np.zeros_like(voltage_out)
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
                "voltage_in",
            ] = np.where(voltage_in == np.amax(voltage_in), 1.0, 0.0)
        else:
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
                "voltage_in",
            ] = np.zeros_like(voltage_out)
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
                "voltage_out",
            ] = np.where(voltage_out == np.amax(voltage_out), 1.0, 0.0)

        cable_temperature = inputs["cable_temperature"]
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":temperature_max",
            "cable_temperature",
        ] = np.where(cable_temperature == np.amax(cable_temperature), 1.0, 0.0)
