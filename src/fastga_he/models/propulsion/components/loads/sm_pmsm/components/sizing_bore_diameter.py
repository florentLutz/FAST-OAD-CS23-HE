# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorBoreDiameter(om.ExplicitComponent):
    """
    Computation of the stator bore diameter of the SM PMSM. The formula is obtained from equation (
    II-43) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient",
            val=np.nan,
            desc="The fraction of stator bore diameter and active length",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":tangential_stress_caliber",
            val=np.nan,
            units="Pa",
            desc="The maximum surface tangential stress applied on rotor by electromagnetism",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":electromagnetic_torque_max",
            val=np.nan,
            units="N*m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            units="m",
            desc="Stator bore diameter of the SM PMSM",
            val=0.115,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        form_coefficient = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient"
        ]
        sigma = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber"
        ]
        electromagnetic_torque_max = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_max"
        ]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"] = 2.0 * (
            ((form_coefficient / (4.0 * np.pi * sigma)) * electromagnetic_torque_max) ** (1.0 / 3.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        form_coefficient = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient"
        ]
        sigma = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber"
        ]
        electromagnetic_torque_max = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_max"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient",
        ] = (
            2.0
            * (electromagnetic_torque_max ** (1.0 / 3.0) * (np.pi * sigma) ** (2.0 / 3.0))
            / (3.0 * (2.0 ** (2.0 / 3.0)) * np.pi * sigma * form_coefficient ** (2.0 / 3.0))
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber",
        ] = (
            -2.0
            * (
                form_coefficient ** (1.0 / 3.0)
                * electromagnetic_torque_max ** (1.0 / 3.0)
                * (np.pi * sigma) ** (2.0 / 3.0)
            )
            / (3.0 * (2.0 ** (2.0 / 3.0)) * np.pi * sigma**2.0)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_max",
        ] = (
            2.0
            * (form_coefficient ** (1.0 / 3.0) * (np.pi * sigma) ** (2.0 / 3.0))
            / (
                3.0
                * (2.0 ** (2.0 / 3.0))
                * np.pi
                * sigma
                * electromagnetic_torque_max ** (2.0 / 3.0)
            )
        )
