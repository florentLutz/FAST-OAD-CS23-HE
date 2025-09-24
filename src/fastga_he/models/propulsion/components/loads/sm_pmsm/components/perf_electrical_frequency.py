# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesElectricalFrequency(om.ExplicitComponent):
    """
    Computation of the electrical frequency which is used to determine the speed at which the magnetic field
    rotates inside the motor.
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

        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
            val=np.nan,
            desc="Number of the north and south pole pairs in the PMSM",
        )

        self.add_output(
            "electrical_frequency",
            units="Hz",
            val=0.0,
            shape=number_of_points,
            desc="The oscillation frequency of the PMSM AC current",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="electrical_frequency",
            wrt="rpm",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="electrical_frequency",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["electrical_frequency"] = (
            inputs["rpm"]
            * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"]
            / 60.0
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        partials[
            "electrical_frequency",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
        ] = inputs["rpm"] / 60.0

        partials["electrical_frequency", "rpm"] = np.full(
            number_of_points,
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"]
            / 60.0,
        )
