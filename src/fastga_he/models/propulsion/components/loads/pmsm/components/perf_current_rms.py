# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

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
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_constant",
            val=1.0,
            units="N*m/A",
        )

        self.add_output(
            "ac_current_rms_in",
            units="A",
            val=np.full(number_of_points, 10.0),
            shape=number_of_points,
            desc="RMS current in all the phases of the motor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        torque = inputs["torque_out"]
        k_t = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_constant"]

        rms_current = torque / k_t

        outputs["ac_current_rms_in"] = rms_current

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        torque = inputs["torque_out"]
        k_t = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_constant"]

        partials["ac_current_rms_in", "torque_out"] = np.eye(number_of_points) / k_t
        partials[
            "ac_current_rms_in",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":torque_constant",
        ] = (
            -torque / k_t ** 2.0
        )
