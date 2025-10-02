# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

COPPER_RESISTIVITY = 1.68e-8  # Copper resistivity at 293.15K [Ohm·m]
COPPER_TEMPERATURE_COEFF = 0.00393  # Temperature coefficient for copper [1/K]


class PerformancesWindingResistivityFixed(om.ExplicitComponent):
    """
    Computation of the copper electrical resistivity of a fixed temperature. The formula is
    obtained from equation (II-64) in :cite:`touhami:2020`.
    """

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

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":winding_temperature",
            val=np.nan,
            units="degK",
            desc="The temperature of the winding conductor cable",
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistivity",
            units="ohm*m",
            desc="Copper electrical resistivity",
            val=3.0e-8,
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            val=COPPER_RESISTIVITY * COPPER_TEMPERATURE_COEFF,
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":resistivity"] = (
            COPPER_RESISTIVITY
            * np.ones(number_of_points)
            * (
                1.0
                + COPPER_TEMPERATURE_COEFF
                * (
                    inputs[
                        "data:propulsion:he_power_train:SM_PMSM:"
                        + motor_id
                        + ":winding_temperature"
                    ]
                    - 293.15
                )
            )
        )
