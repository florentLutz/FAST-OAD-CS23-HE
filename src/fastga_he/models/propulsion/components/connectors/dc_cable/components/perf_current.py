# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCurrent(om.ExplicitComponent):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be " "treated"
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
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
            val=1.0,
        )
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
            "resistance_per_cable",
            val=np.full(number_of_points, np.nan),
            units="ohm",
            desc="resistance of line",
            shape=number_of_points,
        )

        self.add_output(
            "current",
            units="A",
            desc="current of line",
            shape=number_of_points,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        number_cables = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
        ]
        voltage_out = inputs["voltage_out"]
        voltage_in = inputs["voltage_in"]
        resistance_per_cable = inputs["resistance_per_cable"]

        equivalent_resistance = resistance_per_cable / number_cables
        total_current = (voltage_in - voltage_out) / equivalent_resistance
        current = total_current / number_cables

        # Equivalent to :
        # current = (voltage_in - voltage_out) / resistance_per_cable ?

        outputs["current"] = current

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        voltage_out = inputs["voltage_out"]
        voltage_in = inputs["voltage_in"]
        resistance_per_cable = inputs["resistance_per_cable"]

        partials["current", "voltage_in"] = np.diag(1.0 / resistance_per_cable)
        partials["current", "voltage_out"] = np.diag(-1.0 / resistance_per_cable)
        partials["current", "resistance_per_cable"] = np.diag(
            -(voltage_in - voltage_out) / resistance_per_cable ** 2.0
        )
        partials[
            "current",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
        ] = 0.0


class PerformancesHarnessCurrent(om.ExplicitComponent):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be " "treated"
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
            name="data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
            val=1.0,
        )
        self.add_input(
            "current",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="current of line",
            shape=number_of_points,
        )

        self.add_output(
            "total_current",
            units="A",
            desc="current of the harness",
            shape=number_of_points,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        outputs["total_current"] = (
            inputs["current"]
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        partials["total_current", "current"] = np.diag(
            np.full_like(
                inputs["current"],
                inputs[
                    "data:propulsion:he_power_train:DC_cable_harness:"
                    + harness_id
                    + ":number_cables"
                ],
            )
        )

        partials[
            "total_current",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
        ] = inputs["current"]
