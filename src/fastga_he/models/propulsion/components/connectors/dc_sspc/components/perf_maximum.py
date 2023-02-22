# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDCSSPCMaximum(om.ExplicitComponent):
    """
    This identifies the maximum seen by the components during the mission.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        dc_sspc_id = self.options["dc_sspc_id"]

        self.add_input(
            "dc_current_in",
            val=np.full(number_of_points, np.nan),
            units="A",
        )
        self.add_input(
            "dc_voltage_in",
            val=np.full(number_of_points, np.nan),
            units="V",
        )
        self.add_input(
            "dc_voltage_out",
            val=np.full(number_of_points, np.nan),
            units="V",
        )

        self.add_output(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max",
            val=1000.0,
            units="A",
            desc="Maximum current flowing through the SSPC",
        )
        self.add_output(
            name="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max",
            val=800.0,
            units="V",
            desc="Maximum voltage at the terminals SSPC",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max",
            wrt="dc_current_in",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max",
            wrt=["dc_voltage_in", "dc_voltage_out"],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_sspc_id = self.options["dc_sspc_id"]

        outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max"] = np.amax(
            inputs["dc_current_in"]
        )

        outputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max"] = np.amax(
            np.maximum(inputs["dc_voltage_in"], inputs["dc_voltage_out"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_sspc_id = self.options["dc_sspc_id"]

        current = inputs["dc_current_in"]
        partials[
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":current_max", "dc_current_in"
        ] = np.where(current == np.amax(current), 1.0, 0.0)

        voltage_out = inputs["dc_voltage_out"]
        voltage_in = inputs["dc_voltage_in"]
        if np.amax(voltage_in) > np.amax(voltage_out):
            partials[
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max",
                "dc_voltage_out",
            ] = np.zeros_like(voltage_out)
            partials[
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max",
                "dc_voltage_in",
            ] = np.where(voltage_in == np.amax(voltage_in), 1.0, 0.0)
        else:
            partials[
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max",
                "dc_voltage_in",
            ] = np.zeros_like(voltage_out)
            partials[
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":voltage_max",
                "dc_voltage_out",
            ] = np.where(voltage_out == np.amax(voltage_out), 1.0, 0.0)
