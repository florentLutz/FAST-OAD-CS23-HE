# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingFrameDimension(om.ExplicitComponent):
    """
    Computation of the frame diameter and length of the SM PMSM. The formula is obtained from
    equation (II-53) and figure (II.10.a) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            val=np.nan,
            units="m",
            desc="The length of electromagnetism active part of SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
            val=np.nan,
            desc="The factor to account extra length from end winding",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
            val=np.nan,
            units="m",
            desc="The outer stator diameter of the SM PMSM",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_diameter",
            units="m",
            val=0.252,
            desc="The motor casing diameter",
        )
        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_length",
            units="m",
            val=0.293,
            desc="The motor casing length",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_length",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        stator_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter"
        ]
        stator_radius = stator_diameter / 2.0
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        end_winding_coeff = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff"
        ]

        tau_frame = (
            0.7371 * stator_radius**2.0 - 0.580 * stator_radius + 1.1599
            if (stator_radius <= 0.4)
            else 1.04
        )

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_diameter"] = (
            tau_frame * stator_diameter
        )
        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_length"] = (
            tau_frame * active_length * end_winding_coeff
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        stator_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter"
        ]
        stator_radius = stator_diameter / 2.0
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        end_winding_coeff = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff"
        ]

        tau_frame = (
            0.7371 * stator_radius**2.0 - 0.580 * stator_radius + 1.1599
            if (stator_radius <= 0.4)
            else 1.04
        )

        d_tau_d_stator_diameter = 0.7371 * stator_radius - 0.29 if stator_radius <= 0.4 else 0.0
        d_frame_diameter_d_stator_diameter = tau_frame + d_tau_d_stator_diameter * stator_diameter

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
        ] = d_frame_diameter_d_stator_diameter

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
        ] = d_tau_d_stator_diameter * active_length * end_winding_coeff

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
        ] = tau_frame * end_winding_coeff

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
        ] = tau_frame * active_length
