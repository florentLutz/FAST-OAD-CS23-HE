# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingMotorPhaseResistanceScaling(om.ExplicitComponent):
    """
    Computation of scaling factor for the phase resistance of cylindrical PMSM.

    The exponent for the voltage is taken from :cite:`budinger:2012`, the exponents for diameter
    and length were computed by doing a regression on the EMRAX family. Regression can be seen in
    ..methodology.resistance_scaling.
    """

    def initialize(self):
        # Reference motor : EMRAX 268, HV

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "max_voltage_ref",
            default=800.0,
            desc="Max voltage of the reference motor in [V]",
        )

    def setup(self):

        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":voltage:peak",
            val=np.nan,
            units="V",
            desc="Max voltage of the motor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:phase_resistance",
            val=1.0,
            desc="Scaling factor for the phase resistance of the motor",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:phase_resistance",
            wrt=[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":voltage:peak",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        max_voltage_ref = self.options["max_voltage_ref"]

        max_voltage = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":voltage:peak"]
        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]
        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]

        voltage_scaling = max_voltage / max_voltage_ref

        resistance_scaling = voltage_scaling ** 2.0 * l_scaling ** -2.62 * d_scaling ** -1.12

        outputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:phase_resistance"
        ] = resistance_scaling

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        max_voltage_ref = self.options["max_voltage_ref"]

        max_voltage = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":voltage:peak"]
        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]
        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]

        voltage_scaling = max_voltage / max_voltage_ref

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:phase_resistance",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":voltage:peak",
        ] = (
            2.0 * max_voltage / max_voltage_ref ** 2.0 * l_scaling ** -2.62 * d_scaling ** -1.12
        )
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:phase_resistance",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
        ] = (
            -1.12 * voltage_scaling ** 2.0 * l_scaling ** -2.62 * d_scaling ** -2.12
        )
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:phase_resistance",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
        ] = (
            -2.62 * voltage_scaling ** 2.0 * l_scaling ** -3.62 * d_scaling ** -1.12
        )
