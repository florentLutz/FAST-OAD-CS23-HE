# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class OptimalTorque(om.ExplicitComponent):
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

        self.add_input("power", units="W", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":alpha",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":beta",
            val=np.nan,
        )

        self.add_output("torque", units="N*m", val=0.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        power = inputs["power"]

        alpha = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":alpha"]
        beta = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":beta"]

        torque = (
            (1.5 / 2.0) ** (1.0 / 3.5)
            * beta ** (1.0 / 3.5)
            * alpha ** (-1.0 / 3.5)
            * power ** (1.5 / 3.5)
        )

        outputs["torque"] = torque

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        power = inputs["power"]

        alpha = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":alpha"]
        beta = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":beta"]

        partials["torque", "power"] = np.diag(
            (1.5 / 3.5)
            * (1.5 / 2.0) ** (1.0 / 3.5)
            * beta ** (1.0 / 3.5)
            * alpha ** (-1.0 / 3.5)
            * power ** (-2.0 / 3.5)
        )
        partials["torque", "data:propulsion:he_power_train:PMSM:" + motor_id + ":alpha"] = (
            (-1.0 / 3.5)
            * (1.5 / 2.0) ** (1.0 / 3.5)
            * beta ** (1.0 / 3.5)
            * alpha ** (-4.5 / 3.5)
            * power ** (1.5 / 3.5)
        )
        partials["torque", "data:propulsion:he_power_train:PMSM:" + motor_id + ":beta"] = (
            (1.0 / 3.5)
            * (1.5 / 2.0) ** (1.0 / 3.5)
            * beta ** (-2.5 / 3.5)
            * alpha ** (-1.0 / 3.5)
            * power ** (1.5 / 3.5)
        )
