# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCurrentRMS(om.ExplicitComponent):
    """Computation of the rms current in all phases based on the torque and torque constant."""

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("torque_out", units="N*m", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_constant",
            val=np.nan,
            units="N*m/A",
        )

        self.add_output(
            "ac_current_rms_in",
            units="A",
            val=np.full(number_of_points, 10.0),
            shape=number_of_points,
            desc="RMS current in all the phases of the motor",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="torque_out",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_constant",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["ac_current_rms_in"] = (
            inputs["torque_out"]
            / inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_constant"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        torque = inputs["torque_out"]
        k_t = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_constant"]

        partials["ac_current_rms_in", "torque_out"] = np.ones(number_of_points) / k_t
        partials[
            "ac_current_rms_in",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_constant",
        ] = -torque / k_t**2.0
