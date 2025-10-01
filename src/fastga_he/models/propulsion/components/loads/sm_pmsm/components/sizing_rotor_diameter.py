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
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            units="m",
            desc="The length of electromagnetism active part of SM PMSM",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density",
            units="kg/m**3",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":youngs_modules",
            units="Pa",
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
        rpm_rating = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating"]
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        rho_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density"
        ]
        e_rotor = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":youngs_modules"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] = (
            0.25
            * rpm_rating
            * np.pi
            * active_length**2.0
            * end_winding_coeff**2.0
            / np.sqrt(30.0 * e_rotor / rho_rotor)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        end_winding_coeff = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff"
        ]
        rpm_rating = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating"]
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        rho_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density"
        ]
        e_rotor = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":youngs_modules"]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
        ] = (
            0.5
            * rpm_rating
            * np.pi
            * active_length**2.0
            * end_winding_coeff
            / np.sqrt(30.0 * e_rotor / rho_rotor)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
        ] = (
            0.5
            * rpm_rating
            * np.pi
            * active_length
            * end_winding_coeff**2.0
            / np.sqrt(30.0 * e_rotor / rho_rotor)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
        ] = (
            0.25
            * np.pi
            * active_length**2.0
            * end_winding_coeff**2.0
            / np.sqrt(30.0 * e_rotor / rho_rotor)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":youngs_modules",
        ] = (
            -0.25
            * np.pi
            * active_length**2.0
            * end_winding_coeff**2.0
            / (2.0 * rho_rotor * np.sqrt(30.0) * np.sqrt(e_rotor / rho_rotor) ** 3.0)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density",
        ] = (
            0.25
            * rpm_rating
            * np.pi
            * active_length**2.0
            * end_winding_coeff**2.0
            / (2.0 * np.sqrt(30.0 * e_rotor * rho_rotor))
        )
