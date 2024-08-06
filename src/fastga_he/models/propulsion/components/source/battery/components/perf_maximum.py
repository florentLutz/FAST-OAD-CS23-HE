# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum torque, rpm, tip mach and advance ratio of the propeller.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("c_rate", units="h**-1", val=np.full(number_of_points, np.nan))
        self.add_input("terminal_voltage", units="V", val=np.full(number_of_points, np.nan))
        self.add_input("state_of_charge", units="percent", val=np.full(number_of_points, np.nan))
        self.add_input("losses_cell", units="W", val=np.full(number_of_points, np.nan))

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:voltage_min",
            units="V",
            val=3.2,
            desc="Minimum voltage provided by the cells during the mission",
        )
        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:voltage_max",
            units="V",
            val=3.2,
            desc="Maximum voltage needed by the cells during the mission",
        )
        self.declare_partials(
            of=[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell:voltage_min",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":cell:voltage_max",
            ],
            wrt="terminal_voltage",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min",
            val=20.0,
            units="percent",
            desc="Minimum state-of-charge of the battery during the mission",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min",
            wrt="state_of_charge",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max",
            val=2.0,
            units="h**-1",
            desc="Maximum C-rate of the battery modules during the mission",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max",
            wrt="c_rate",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:losses_max",
            val=2.0,
            units="W",
            desc="Minimum state-of-charge of the battery during the mission",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":cell:losses_max",
            wrt="losses_cell",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:voltage_max"
        ] = np.max(inputs["terminal_voltage"])
        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:voltage_min"
        ] = np.min(inputs["terminal_voltage"])
        outputs["data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min"] = (
            np.min(inputs["state_of_charge"])
        )
        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max"
        ] = np.max(inputs["c_rate"])
        outputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:losses_max"
        ] = np.max(inputs["losses_cell"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:voltage_max",
            "terminal_voltage",
        ] = np.where(inputs["terminal_voltage"] == np.max(inputs["terminal_voltage"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:voltage_min",
            "terminal_voltage",
        ] = np.where(inputs["terminal_voltage"] == np.min(inputs["terminal_voltage"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":SOC_min",
            "state_of_charge",
        ] = np.where(inputs["state_of_charge"] == np.min(inputs["state_of_charge"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":c_rate_max",
            "c_rate",
        ] = np.where(inputs["c_rate"] == np.max(inputs["c_rate"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:losses_max",
            "losses_cell",
        ] = np.where(inputs["losses_cell"] == np.max(inputs["losses_cell"]), 1.0, 0.0)
