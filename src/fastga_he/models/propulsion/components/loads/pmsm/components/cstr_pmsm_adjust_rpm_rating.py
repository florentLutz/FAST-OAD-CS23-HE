#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class ConstraintsPMSMAdjustRPMRating(om.ExplicitComponent):
    """
    The low robustness of the electric motor model and the strong dependency of the coherence of the
    module on the correlation between rating shaft power and rpm rating has already been
    established. This module proposes to auto adjust the rpm rating to improve stability and
    coherence of results when doing some design space exploration. It will of course not be enabled
    by default.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_rating",
            units="kW",
            val=np.nan,
            desc="Value of the maximum power the PMSM can provide, used for sizing",
        )

        self.add_output(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating",
            units="min**-1",
            val=6000.0,
            desc="Max continuous rpm of the motor",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        shaft_power_rating = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_rating"
        ]
        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating"] = np.clip(
            0.0864 * shaft_power_rating**2 - 43.22 * shaft_power_rating + 7686.8, 3250.0, 8000.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        shaft_power_rating = inputs[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_rating"
        ]
        expected_rpm = 0.0864 * shaft_power_rating**2 - 43.22 * shaft_power_rating + 7686.8

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":rpm_rating",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":shaft_power_rating",
        ] = np.where(
            expected_rpm == np.clip(expected_rpm, 3250.0, 8000.0),
            2.0 * 0.0864 * shaft_power_rating - 43.22,
            1e-6,
        )
