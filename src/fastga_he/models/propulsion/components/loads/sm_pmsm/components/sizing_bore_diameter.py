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
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_rating",
            val=np.nan,
            units="N*m",
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate",
            val=0.95,
            desc="The ratio of the electromagnetic torque converted to output torque",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            units="m",
            desc="Stator bore diameter of the SM PMSM",
            val=0.114,
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
        torque_rating = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_rating"
        ]
        conversion_rate = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate"
        ]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"] = (
            2.0
            * np.cbrt((form_coefficient * torque_rating / (4.0 * np.pi * sigma * conversion_rate)))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        form_coefficient = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient"
        ]
        sigma = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber"
        ]
        torque_rating = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_rating"
        ]
        conversion_rate = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate",
        ] = (
            -2.0
            * np.cbrt(
                (form_coefficient * torque_rating / (4.0 * np.pi * sigma * conversion_rate**4.0))
            )
            / 3.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient",
        ] = (
            2.0
            * np.cbrt(
                (torque_rating / (4.0 * np.pi * sigma * conversion_rate * form_coefficient**2.0))
            )
            / 3.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber",
        ] = (
            -2.0
            * np.cbrt(
                (form_coefficient * torque_rating / (4.0 * np.pi * sigma**4.0 * conversion_rate))
            )
            / 3.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_rating",
        ] = (
            2.0
            * np.cbrt(
                (form_coefficient / (4.0 * np.pi * sigma * conversion_rate * torque_rating**2.0))
            )
            / 3.0
        )
