# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRadiusRatio(om.ExplicitComponent):
    """
    Computation of the radius ratio and air gap thickness of a cylindrical PMSM. The formulas
    are obtained from equation (II-24) and (II-51) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            val=np.nan,
            units="m",
            desc="Stator bore diameter of the SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio",
            val=0.97,
            desc="the radius ratio of the rotor radius and the stator bore radius",
        )
        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness",
            units="m",
            desc="The distance between the rotor and the stator bore",
            val=0.0017,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio",
            wrt="*",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            val=0.5,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            val=-0.5,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"]
            / inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"]
        )
        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"]
            - inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"]
        ) / 2.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = 1.0 / inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
        ] = (
            -inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"]
            / inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"] ** 2.0
        )
