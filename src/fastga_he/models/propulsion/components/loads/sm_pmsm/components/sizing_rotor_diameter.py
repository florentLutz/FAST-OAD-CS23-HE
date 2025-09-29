# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRotorDiameter(om.ExplicitComponent):
    """
    Computation of the rotor diameterof a cylindrical PMSM. The formulas are obtained from
    equation (II-81) and (II-89) in :cite:`touhami:2020`.
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
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
            val=np.nan,
            desc="The factor to account extra length from end winding",
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
            units="min**-1",
            val=np.nan,
            desc="Max continuous rpm of the motor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density",
            units="kg/m**3",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":youngs_modules",
            units="MPa",
            val=np.nan,
            desc="Young's modules of the rotor material",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            units="m",
            val=0.05,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        end_winding_coeff = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff"
        ]
        bore_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"
        ]
        rpm_rating = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating"]
        form_coefficient = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":form_coefficient"
        ]
        rho_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density"
        ]
        e_rotor = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":youngs_modules"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] = (
            (bore_diameter * end_winding_coeff / form_coefficient) ** 2.0
            * (np.pi * rpm_rating / 20.0)
            * np.sqrt(5.0 * rho_rotor / (6.0 * e_rotor))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio",
        ] = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
        ] = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio",
        ] = (-inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"]) / 2.0

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
        ] = (
            1.0 - inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":radius_ratio"]
        ) / 2.0
