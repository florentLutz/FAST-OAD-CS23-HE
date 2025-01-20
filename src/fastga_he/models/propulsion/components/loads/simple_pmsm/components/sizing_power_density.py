# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingPowerDensity(om.ExplicitComponent):
    """Computation of the electric active power required to run the motor."""

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):

        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":shaft_power_rating",
            units="kW",
            val=np.nan,
            desc="Max continuous shaft power of the motor",
        )

        self.add_input(
            name="data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":mass",
            val=np.nan,
            units="kg",
        )

        self.add_output(
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":power_density",
            units="kW/kg",
            val=10.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":power_density"] = (
            inputs["data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":shaft_power_rating"]
            / inputs["data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":mass"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        partials[
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":power_density",
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":shaft_power_rating",
        ] = (
            1 / inputs["data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":mass"]
        )
        partials[
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":power_density",
            "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":mass",
        ] = (
            -inputs[
                "data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":shaft_power_rating"
            ]
            / inputs["data:propulsion:he_power_train:simple_PMSM:" + motor_id + ":mass"] ** 2.0
        )
