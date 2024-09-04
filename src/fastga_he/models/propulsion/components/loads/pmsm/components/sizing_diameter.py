# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingMotorDiameter(om.ExplicitComponent):
    """Computation of the diameter of a cylindrical PMSM."""

    def initialize(self):
        # Reference motor : EMRAX 268

        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "diameter_ref",
            default=0.268,
            desc="Diameter of the reference motor in [m]",
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":diameter",
            val=self.options["diameter_ref"],
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":diameter",
            wrt="data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]
        d_ref = self.options["diameter_ref"]

        d_scaling = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter"]

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":diameter"] = d_ref * d_scaling

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        d_ref = self.options["diameter_ref"]

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":diameter",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":scaling:diameter",
        ] = d_ref
