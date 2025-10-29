# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


COPPER_TEMPERATURE_COEFF = 0.00393  # Temperature coefficient for copper [1/K]


class PerformancesResistance(om.ExplicitComponent):
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
            name="winding_temperature",
            val=np.nan,
            units="degK",
            shape=number_of_points,
            desc="The temperature of the winding conductor cable",
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":reference_conductor_resistance",
            units="ohm",
            val=np.nan,
            desc="The conductor's reference electric resistance at 293.15K",
        )

        self.add_output(
            "resistance",
            units="ohm",
            val=1.0e-4,
            shape=number_of_points,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="resistance",
            wrt="winding_temperature",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="resistance",
            wrt="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":reference_conductor_resistance",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["resistance"] = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":reference_conductor_resistance"
        ] * (1.0 + COPPER_TEMPERATURE_COEFF * (inputs["winding_temperature"] - 293.15))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials[
            "resistance",
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":reference_conductor_resistance",
        ] = 1.0 + COPPER_TEMPERATURE_COEFF * (inputs["winding_temperature"] - 293.15)

        partials["resistance", "winding_temperature"] = (
            COPPER_TEMPERATURE_COEFF
            * inputs[
                "data:propulsion:he_power_train:SM_PMSM:"
                + motor_id
                + ":reference_conductor_resistance"
            ]
        )
