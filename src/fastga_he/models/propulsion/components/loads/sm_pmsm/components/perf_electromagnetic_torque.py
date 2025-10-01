# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesElectromagneticTorque(om.ExplicitComponent):
    """
    Computation of the electromagnetic torque of the SM PMSM.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        motor_id = self.options["motor_id"]

        self.add_input("torque_out", units="N*m", val=np.nan, shape=number_of_points)
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate",
            val=0.95,
            desc="The ratio of the electromagnetic torque converted to output torque",
        )

        self.add_output(
            "electromagnetic_torque",
            units="N*m",
            val=200.0,
            shape=number_of_points,
            desc="Total electromechanical torque from the motor",
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="electromagnetic_torque",
            wrt="torque_out",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="electromagnetic_torque",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["electromagnetic_torque"] = (
            inputs["torque_out"]
            / inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials["electromagnetic_torque", "torque_out"] = (
            1.0
            / inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate"
            ]
        )

        partials[
            "electromagnetic_torque",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate",
        ] = (
            -inputs["torque_out"]
            / inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate"
            ]
            ** 2.0
        )
