# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesActivePower(om.ExplicitComponent):
    """Computation of the electric active power required to run the motor."""

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        motor_id = self.options["motor_id"]

        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":efficiency",
            val=np.nan,
        )

        self.add_output(
            "active_power", units="W", val=np.full(number_of_points, 50e3), shape=number_of_points
        )

        self.declare_partials(
            of="*",
            wrt="shaft_power_out",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":efficiency",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["active_power"] = (
            inputs["shaft_power_out"]
            / inputs["data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":efficiency"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        partials["active_power", "shaft_power_out"] = (
            np.ones(number_of_points)
            / inputs["data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":efficiency"]
        )
        partials[
            "active_power", "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":efficiency"
        ] = (
            -inputs["shaft_power_out"]
            / inputs["data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":efficiency"]
            ** 2.0
        )
