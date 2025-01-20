# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingMotorLengthScaling(om.ExplicitComponent):
    """
    Computation of scaling factor for the length of a cylindrical PMSM.

    Formula taken from :cite:`thauvin:2018`
    """

    def initialize(self):
        # Reference motor : EMRAX 268

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "torque_cont_ref",
            default=200.0,
            desc="Max continuous torque of the reference motor in [N*m]",
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":torque_rating",
            val=np.nan,
            units="N*m",
            desc="Max continuous torque of the motor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:diameter",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:length",
            val=1.0,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:length",
            wrt=[
                "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":torque_rating",
                "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:diameter",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        torque_cont_ref = self.options["torque_cont_ref"]

        torque_cont = inputs[
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":torque_rating"
        ]
        d_scaling = inputs[
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:diameter"
        ]

        torque_scaling = torque_cont / torque_cont_ref

        l_scaling = torque_scaling * d_scaling**-2.5

        outputs["data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:length"] = (
            l_scaling
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        torque_cont_ref = self.options["torque_cont_ref"]

        torque_cont = inputs[
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":torque_rating"
        ]
        d_scaling = inputs[
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:diameter"
        ]

        torque_scaling = torque_cont / torque_cont_ref

        partials[
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:length",
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":torque_rating",
        ] = 1.0 / torque_cont_ref * d_scaling**-2.5
        partials[
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:length",
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":scaling:diameter",
        ] = -2.5 * torque_scaling * d_scaling**-3.5
