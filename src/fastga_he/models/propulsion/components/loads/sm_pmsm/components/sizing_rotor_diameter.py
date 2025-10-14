# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om


class SizingRotorDiameter(om.ExplicitComponent):
    """
    Computation of the rotor diameterof a cylindrical PMSM. The formulas are obtained from
    equation (II-85) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio",
            val=0.97,
            desc="the radius ratio of the rotor radius and the stator bore radius",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness",
            units="m",
            desc="The distance between the rotor and the stator bore",
            val=0.00075,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_factor",
            desc="Remain at 1.0 if the maximum mechanical stress is within the range of yield "
            "stress",
            val=1.0,
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            units="m",
            val=0.11,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        mechanical_stress_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_factor"
        ]
        air_gap_thickness = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness"
        ]
        radius_ratio = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio"
        ]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] = (
            2.0 * air_gap_thickness * radius_ratio * mechanical_stress_factor / (1.0 - radius_ratio)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        mechanical_stress_factor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_factor"
        ]
        air_gap_thickness = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness"
        ]
        radius_ratio = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio",
        ] = 2.0 * air_gap_thickness * mechanical_stress_factor / (1.0 - radius_ratio) ** 2.0

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness",
        ] = 2.0 * radius_ratio * mechanical_stress_factor / (1.0 - radius_ratio)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_factor",
        ] = 2.0 * air_gap_thickness * radius_ratio / (1.0 - radius_ratio)
