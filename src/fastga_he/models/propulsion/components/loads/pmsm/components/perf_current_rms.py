# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesCurrentRMS(om.ExplicitComponent):
    """Computation of the rms current in one phase from the torque and torque constant."""

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

        self.add_input("torque", units="N*m", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:torque_constant",
            val=1.0,
            units="N*m/A",
        )

        self.add_output(
            "rms_current",
            units="A",
            val=np.full(number_of_points, 10.0),
            shape=number_of_points,
            desc="RMS current in one of the phase of the motor",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        torque = inputs["torque"]
        k_t = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:torque_constant"]

        rms_current = torque / k_t

        outputs["rms_current"] = rms_current

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        torque = inputs["torque"]
        k_t = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:torque_constant"]

        partials["rms_current", "torque"] = np.eye(number_of_points) / k_t
        partials[
            "rms_current",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:torque_constant",
        ] = (
            -torque / k_t ** 2.0
        )