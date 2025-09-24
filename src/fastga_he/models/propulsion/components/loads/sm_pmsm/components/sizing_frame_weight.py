# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingFrameGeometry(om.ExplicitComponent):
    """
    Computation of the frame diameter and weight of the PMSM. The formula is obtained from
    equation (II-53) and (II-59) respectively in :cite:`touhami:2020`.
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
            desc="The stator length of PMSM",
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
            desc="The outer stator diameter of the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_density",
            val=2.7,
            units="kg/m**3",
            desc="Default value set as the density of 6063 aluminum alloy",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_diameter",
            units="m",
            val=0.23,
        )
        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            units="kg",
            val=6.0,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_density",
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
        rho_frame = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_density"]

        tau_frame = (
            0.7371 * stator_radius**2.0 - 0.580 * stator_radius + 1.1599
            if (stator_radius <= 0.4)
            else 1.04
        )
        frame_radius = tau_frame * stator_radius

        # Frame weight
        frame_mass = rho_frame * (
            np.pi * active_length * end_winding_coeff * (frame_radius**2.0 - stator_radius**2.0)
            + 2.0 * np.pi * (tau_frame - 1.0) * stator_radius * frame_radius**2.0
        )

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_diameter"] = (
            tau_frame * stator_diameter
        )
        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass"] = frame_mass

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
        rho_frame = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_density"]

        tau_frame = (
            0.7371 * stator_radius**2 - 0.580 * stator_radius + 1.1599
            if stator_radius <= 0.4
            else 1.04
        )
        d_tau_d_stator_diameter = 0.7371 * stator_radius - 0.29 if stator_radius <= 0.4 else 0.0
        frame_radius = tau_frame * stator_radius
        d_frame_diameter_d_stator_diameter = tau_frame + d_tau_d_stator_diameter * stator_diameter

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
        ] = d_frame_diameter_d_stator_diameter

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
        ] = rho_frame * (
            np.pi
            * active_length
            * end_winding_coeff
            * (frame_radius * d_frame_diameter_d_stator_diameter - stator_radius)
            + 2.0 * np.pi * d_tau_d_stator_diameter * stator_radius * frame_radius**2.0
            + 2.0 * np.pi * (tau_frame - 1.0) * 0.5 * frame_radius**2.0
            + 2.0
            * np.pi
            * (tau_frame - 1.0)
            * stator_radius
            * frame_radius
            * d_frame_diameter_d_stator_diameter
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
        ] = rho_frame * np.pi * end_winding_coeff * (frame_radius**2.0 - stator_radius**2.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
        ] = rho_frame * np.pi * active_length * (frame_radius**2.0 - stator_radius**2.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_density",
        ] = (
            np.pi * active_length * end_winding_coeff * (frame_radius**2.0 - stator_radius**2.0)
            + 2.0 * np.pi * (tau_frame - 1.0) * stator_radius * frame_radius**2.0
        )
