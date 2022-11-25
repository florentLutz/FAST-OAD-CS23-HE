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
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the expected output (current flows from source/input to load/output)",
            shape=number_of_points,
        )
        self.add_input(
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
            desc="Voltage at the expected input (current flows from source/input to load/output)",
            shape=number_of_points,
        )
        self.add_input(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
            val=1.0,
        )
        self.add_input(
            "dc_current_one_cable",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="current of one cable of the harness",
            shape=number_of_points,
        )
        self.add_input(
            "cable_temperature",
            val=np.full(number_of_points, np.nan),
            units="degK",
            desc="temperature inside of the cable",
            shape=number_of_points,
        )
        self.add_input(
            "conduction_losses",
            units="W",
            desc="Joule losses in one cable of the harness",
            val=np.full(number_of_points, np.nan),
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
            val=500.0,
            units="A",
            desc="Maximum current flowing through the harness, all cables included",
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
        self.add_output(
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":losses_max",
            val=200.0,
            units="W",
            desc="Maximum losses in one cable of the harness",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
            wrt=[
                "dc_current_one_cable",
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
            ],
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
            wrt=["dc_voltage_out", "dc_voltage_in"],
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":temperature_max",
            wrt=["cable_temperature"],
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":losses_max",
            wrt=["conduction_losses"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        harness_id = self.options["harness_id"]

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max"
        ] = (
            np.amax(inputs["dc_current_one_cable"])
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
        )

        voltage_max = np.maximum(inputs["dc_voltage_out"], inputs["dc_voltage_in"])

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max"
        ] = np.amax(voltage_max)

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":temperature_max"
        ] = np.amax(inputs["cable_temperature"])

        outputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":losses_max"
        ] = np.amax(inputs["conduction_losses"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        current = inputs["dc_current_one_cable"]
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
            "dc_current_one_cable",
        ] = (
            np.where(current == np.amax(current), 1.0, 0.0)
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
        )
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":current_max",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
        ] = np.amax(inputs["dc_current_one_cable"])

        voltage_out = inputs["dc_voltage_out"]
        voltage_in = inputs["dc_voltage_in"]
        if np.amax(voltage_in) > np.amax(voltage_out):
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
                "dc_voltage_out",
            ] = np.zeros_like(voltage_out)
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
                "dc_voltage_in",
            ] = np.where(voltage_in == np.amax(voltage_in), 1.0, 0.0)
        else:
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
                "dc_voltage_in",
            ] = np.zeros_like(voltage_out)
            partials[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":voltage_max",
                "dc_voltage_out",
            ] = np.where(voltage_out == np.amax(voltage_out), 1.0, 0.0)

        cable_temperature = inputs["cable_temperature"]
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":temperature_max",
            "cable_temperature",
        ] = np.where(cable_temperature == np.amax(cable_temperature), 1.0, 0.0)

        losses = inputs["conduction_losses"]
        partials[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":losses_max",
            "conduction_losses",
        ] = np.where(losses == np.amax(losses), 1.0, 0.0)
