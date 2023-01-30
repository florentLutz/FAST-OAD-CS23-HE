# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingMotorLength(om.ExplicitComponent):
    """Computation of the length of a cylindrical PMSM."""

    def initialize(self):
        # Reference motor : EMRAX 268

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "length_ref", default=0.091, desc="Length of the reference motor in [m]"
        )

    def setup(self):

        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":length",
            val=self.options["length_ref"],
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":length",
            wrt="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]

        l_ref = self.options["length_ref"]

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":length"] = l_ref * l_scaling

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        l_ref = self.options["length_ref"]

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":length",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
        ] = l_ref
