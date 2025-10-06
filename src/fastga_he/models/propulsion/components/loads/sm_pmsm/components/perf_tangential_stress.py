# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesTangentialStress(om.ExplicitComponent):
    """
    Computation of the rotor surface tangential stress due to electromagnetism. The formula is
    obtained from equation (II-4) in :cite:`touhami:2020`.
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

        self.add_input("torque_out", units="N*m", val=np.nan, shape=number_of_points)
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate",
            val=0.95,
            desc="The ratio of the electromagnetic torque converted to output torque",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            units="m",
            desc="The length of electromagnetism active part of SM PMSM",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="tangential_stress",
            units="Pa",
            desc="The surface tangential stress applied by electromagnetism",
            shape=number_of_points,
            val=50.0e3,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt=["torque_out"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="*",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        rotor_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"
        ]
        torque_out = inputs["torque_out"]
        conversion_rate = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate"
        ]

        outputs["tangential_stress"] = (
            2.0 * torque_out / (conversion_rate * np.pi * rotor_diameter**2.0 * active_length)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        rotor_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"
        ]
        torque_out = inputs["torque_out"]
        conversion_rate = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate"
        ]

        partials[
            "tangential_stress",
            "torque_out",
        ] = np.full(number_of_points, 2.0) / (
            conversion_rate * np.pi * rotor_diameter**2.0 * active_length
        )

        partials[
            "tangential_stress",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":torque_conversion_rate",
        ] = -2.0 * torque_out / (conversion_rate**2.0 * np.pi * rotor_diameter**2.0 * active_length)

        partials[
            "tangential_stress",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
        ] = -2.0 * torque_out / (conversion_rate * np.pi * rotor_diameter**2.0 * active_length**2.0)

        partials[
            "tangential_stress",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = -4.0 * torque_out / (conversion_rate * np.pi * rotor_diameter**3.0 * active_length)
