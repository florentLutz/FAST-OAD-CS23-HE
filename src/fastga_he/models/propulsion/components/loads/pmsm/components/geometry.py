# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class MotorGeometry(om.ExplicitComponent):
    def initialize(self):
        # Reference motor : POWERPHASE HD 250

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            name="diameter_ref", default=0.390, desc="Diameter of the reference motor in [m]"
        )
        self.options.declare(
            "length_ref", default=0.226, desc="Length of the reference motor in [m]"
        )

    def setup(self):

        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":diameter",
            val=self.options["diameter_ref"],
            units="m",
        )
        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":length",
            val=self.options["length_ref"],
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":diameter",
            wrt="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":length",
            wrt="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]
        l_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length"]

        d_ref = self.options["diameter_ref"]
        l_ref = self.options["length_ref"]

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":diameter"] = d_ref * d_scaling
        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":length"] = l_ref * l_scaling

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        d_ref = self.options["diameter_ref"]
        l_ref = self.options["length_ref"]

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":diameter",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
        ] = d_ref

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":length",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:length",
        ] = l_ref
