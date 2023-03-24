# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

# Should be an option ?
CUTOFF_CABLE_VOLTAGE = 0.5  # in V


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
            "resistance_per_cable",
            val=np.full(number_of_points, np.nan),
            units="ohm",
            desc="resistance of line",
            shape=number_of_points,
        )

        self.add_output(
            "dc_current_one_cable",
            units="A",
            desc="current of line",
            shape=number_of_points,
            val=250.0,
            lower=-1000.0,
            upper=1000.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        number_cables = inputs[
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
        ]
        voltage_out = inputs["dc_voltage_out"]
        voltage_in = inputs["dc_voltage_in"]
        resistance_per_cable = inputs["resistance_per_cable"]

        equivalent_resistance = resistance_per_cable / number_cables
        total_current = (voltage_in - voltage_out) / equivalent_resistance

        # On some tests, it happened that while the voltage were very low (indicating an open
        # SSPC in this case), current was still flowing through the cable. We will put a simple
        # failsafe checking that voltage is greater than 0.5V which is big enough to filter out
        # the cases of an open SSPC and hopefully small enough to not cause problem with low
        # voltage loads (12V bus for instance)
        low_voltage = np.logical_and(
            np.less_equal(voltage_out, np.full_like(voltage_out, CUTOFF_CABLE_VOLTAGE)),
            np.less_equal(voltage_in, np.full_like(voltage_in, CUTOFF_CABLE_VOLTAGE)),
            dtype=bool,
        )
        current = np.where(low_voltage, 0.0, total_current / number_cables)

        # Equivalent to :
        # current = (voltage_in - voltage_out) / resistance_per_cable ?

        outputs["dc_current_one_cable"] = current

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        harness_id = self.options["harness_id"]

        voltage_out = inputs["dc_voltage_out"]
        voltage_in = inputs["dc_voltage_in"]
        resistance_per_cable = inputs["resistance_per_cable"]

        partials["dc_current_one_cable", "dc_voltage_in"] = np.diag(1.0 / resistance_per_cable)
        partials["dc_current_one_cable", "dc_voltage_out"] = np.diag(-1.0 / resistance_per_cable)
        partials["dc_current_one_cable", "resistance_per_cable"] = np.diag(
            -(voltage_in - voltage_out) / resistance_per_cable ** 2.0
        )
        partials[
            "dc_current_one_cable",
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
            "dc_current_one_cable",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="current of line",
            shape=number_of_points,
        )

        self.add_output(
            "dc_current",
            units="A",
            desc="current of the harness",
            shape=number_of_points,
            val=400.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        harness_id = self.options["harness_id"]

        outputs["dc_current"] = (
            inputs["dc_current_one_cable"]
            * inputs[
                "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        harness_id = self.options["harness_id"]

        partials["dc_current", "dc_current_one_cable"] = np.diag(
            np.full_like(
                inputs["dc_current_one_cable"],
                inputs[
                    "data:propulsion:he_power_train:DC_cable_harness:"
                    + harness_id
                    + ":number_cables"
                ],
            )
        )

        partials[
            "dc_current",
            "data:propulsion:he_power_train:DC_cable_harness:" + harness_id + ":number_cables",
        ] = inputs["dc_current_one_cable"]
