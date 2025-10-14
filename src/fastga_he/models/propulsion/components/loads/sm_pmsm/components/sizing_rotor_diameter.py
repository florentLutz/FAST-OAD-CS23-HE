# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
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
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            units="m",
            desc="The length of electromagnetism active part of SM PMSM",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_poissons_ratio",
            val=0.29,
            desc="The rotor material Poisson's ratio, 4340 steel alloy is set to default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_max",
            units="Pa",
            val=np.nan,
            desc="The maximum rotor mechanical stress",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_youngs_modules",
            units="Pa",
            val=19.0e10,
            desc="Young's modules of the rotor material, 4340 steel alloy is set to default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":safety_factor",
            val=1.5,
            desc="Safety factor for rotor diameter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
            val=np.nan,
            desc="The factor to account extra length from end winding",
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

        end_winding_coeff = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff"
        ]
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        sigma_mec = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_max"
        ]
        e_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_youngs_modules"
        ]
        poissons_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_poissons_ratio"
        ]
        sf_rotor = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":safety_factor"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] = (
            active_length
            * end_winding_coeff
            * (30.0 * sigma_mec * sf_rotor / (e_rotor * (3.0 + poissons_rotor))) ** 0.25
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        end_winding_coeff = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff"
        ]
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        sigma_mec = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_max"
        ]
        e_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_youngs_modules"
        ]
        poissons_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_poissons_ratio"
        ]
        sf_rotor = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":safety_factor"]

        common_denominator = (e_rotor * (3.0 + poissons_rotor)) ** 0.25 * (
            30.0 * sigma_mec * sf_rotor
        ) ** 0.75

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
        ] = (
            active_length
            * (30.0 * sigma_mec * sf_rotor / (e_rotor * (3.0 + poissons_rotor))) ** 0.25
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
        ] = (
            end_winding_coeff
            * (30.0 * sigma_mec * sf_rotor / (e_rotor * (3.0 + poissons_rotor))) ** 0.25
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_max",
        ] = active_length * end_winding_coeff * 7.5 * sf_rotor / common_denominator

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":safety_factor",
        ] = active_length * end_winding_coeff * 7.5 * sigma_mec / common_denominator

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_youngs_modules",
        ] = (
            -active_length
            * end_winding_coeff
            * 7.5
            * sigma_mec
            * sf_rotor
            / (e_rotor * common_denominator)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_poissons_ratio",
        ] = (
            -active_length
            * end_winding_coeff
            * 7.5
            * sigma_mec
            * sf_rotor
            / ((3.0 + poissons_rotor) * common_denominator)
        )
