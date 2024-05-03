# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesEfficiency(om.ExplicitComponent):
    """Computation of the efficiency from shaft power and power losses."""

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

        self.add_input("shaft_power_out", units="W", val=np.nan, shape=number_of_points)
        self.add_input("power_losses", units="W", val=np.nan, shape=number_of_points)
        self.add_input(
            "settings:propulsion:he_power_train:PMSM:" + motor_id + ":k_efficiency",
            val=1.0,
            desc="K factor for the PMSM efficiency",
        )

        self.add_output(
            "efficiency",
            val=np.full(number_of_points, 0.95),
            shape=number_of_points,
            lower=0.0,
            upper=1.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        motor_id = self.options["motor_id"]

        outputs["efficiency"] = np.where(
            inputs["shaft_power_out"] != 0.0,
            inputs["settings:propulsion:he_power_train:PMSM:" + motor_id + ":k_efficiency"]
            * inputs["shaft_power_out"]
            / (inputs["shaft_power_out"] + inputs["power_losses"]),
            np.ones_like(inputs["shaft_power_out"]),
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        motor_id = self.options["motor_id"]

        partials["efficiency", "shaft_power_out"] = np.diag(
            inputs["settings:propulsion:he_power_train:PMSM:" + motor_id + ":k_efficiency"]
            * inputs["power_losses"]
            / (inputs["shaft_power_out"] + inputs["power_losses"]) ** 2.0
        )
        partials["efficiency", "power_losses"] = -np.diag(
            inputs["settings:propulsion:he_power_train:PMSM:" + motor_id + ":k_efficiency"]
            * inputs["shaft_power_out"]
            / (inputs["shaft_power_out"] + inputs["power_losses"]) ** 2.0
        )
        partials[
            "efficiency", "settings:propulsion:he_power_train:PMSM:" + motor_id + ":k_efficiency"
        ] = inputs["shaft_power_out"] / (inputs["shaft_power_out"] + inputs["power_losses"])
