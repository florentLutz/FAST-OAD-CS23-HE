# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingConductorsNumber(om.ExplicitComponent):
    """
    Computation of the number of the conductor slots. The formula is obtained from
    equation (III-58) in :cite:`touhami:2020`.
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

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
            val=np.nan,
            desc="Number of the north and south pairs in the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slots_per_poles_phases",
            val=np.nan,
            desc="The number of conductor slots per poles and per phases",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductors_number",
            desc="Number of conductor slots",
            val=24,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductors_number",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slots_per_poles_phases",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductors_number"] = (
            6.0
            * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"]
            * inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slots_per_poles_phases"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductors_number",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
        ] = (
            6.0
            * inputs[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slots_per_poles_phases"
            ]
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductors_number",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slots_per_poles_phases",
        ] = (
            6.0
            * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"]
        )
