# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingFrameWeight(om.ExplicitComponent):
    """
    Computation of the frame diameter and weight of the SM PMSM. The formula is obtained from
    equation (II-59) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_length",
            units="m",
            val=np.nan,
            desc="The motor casing length",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_diameter",
            units="m",
            val=0.03,
            desc="The motor bearing bore diameter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
            val=np.nan,
            units="m",
            desc="The outer stator diameter of the SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_diameter",
            units="m",
            val=np.nan,
            desc="The motor casing diameter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_density",
            val=2.7e3,
            units="kg/m**3",
            desc="Default value set as the density of 6063 aluminum alloy",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            units="kg",
            val=25.8,
            desc="The weight of the motor casing",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        stator_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter"
        ]
        frame_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_diameter"
        ]
        bearing_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_diameter"
        ]
        frame_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_length"
        ]
        rho_frame = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_density"]
        thickness = 0.5 * (frame_diameter - stator_diameter)

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass"] = (
            rho_frame
            * (
                0.25 * np.pi * (frame_diameter**2.0 - bearing_diameter**2.0) * thickness
                + 0.25
                * np.pi
                * (frame_diameter**2.0 - stator_diameter**2.0)
                * (frame_length - 2.0 * thickness)
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        stator_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter"
        ]
        frame_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_diameter"
        ]
        bearing_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_diameter"
        ]
        frame_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_length"
        ]
        rho_frame = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_density"]
        thickness = 0.5 * (frame_diameter - stator_diameter)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_diameter",
        ] = rho_frame * np.pi * (stator_diameter - frame_diameter) * bearing_diameter / 4.0

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_diameter",
        ] = (
            -rho_frame
            * np.pi
            * (
                6.0 * stator_diameter**2.0
                + 4.0 * (frame_length - frame_diameter) * stator_diameter
                - frame_diameter**2.0
                - bearing_diameter**2.0
            )
            / 8.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_diameter",
        ] = (
            -rho_frame
            * np.pi
            * (
                3.0 * frame_diameter**2.0
                - (2.0 * stator_diameter + 4.0 * frame_length) * frame_diameter
                - 2.0 * stator_diameter**2.0
                + bearing_diameter**2.0
            )
            / 8.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_length",
        ] = rho_frame * np.pi * (frame_diameter**2.0 - stator_diameter**2.0) / 4.0

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_density",
        ] = 0.25 * np.pi * (
            frame_diameter**2.0 - bearing_diameter**2.0
        ) * thickness + 0.25 * np.pi * (frame_diameter**2.0 - stator_diameter**2.0) * (
            frame_length - 2.0 * thickness
        )
