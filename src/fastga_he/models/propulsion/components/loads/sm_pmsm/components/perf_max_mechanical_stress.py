# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaxMechanicalStress(om.ExplicitComponent):
    """
    Computation of the rotor maximum mechanical stress and compare it with the material yield
    stress. The formula is obtained from equation (II-81) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_yield_stress",
            units="Pa",
            val=74.0e7,
            desc="The rotor material yield stress, 4340 steel alloy is set to default",
        )
        self.add_input(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max",
            units="min**-1",
            val=np.nan,
            desc="Max continuous rpm of the motor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_poissons_ratio",
            val=0.29,
            desc="The rotor material Poisson's ratio, 4340 steel alloy is set to default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density",
            units="kg/m**3",
            val=7850.0,
            desc="The rotor material density, 4340 steel alloy is set to default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_max",
            units="Pa",
            desc="The maximum rotor mechanical stress",
            val=29.3e3,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        rho_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density"
        ]
        rotor_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"
        ]
        sigma_yield = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_yield_stress"
        ]
        poissons_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_poissons_ratio"
        ]
        rpm_max = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max"]

        sigma_mec = (
            (3.0 + poissons_rotor) * rho_rotor * rotor_diameter**2.0 * np.pi * rpm_max**2.0 / 480.0
        )

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_max"] = (
            np.array(sigma_mec if sigma_mec < sigma_yield else sigma_yield)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        rho_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density"
        ]
        rotor_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"
        ]
        sigma_yield = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_yield_stress"
        ]
        poissons_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_poissons_ratio"
        ]
        rpm_max = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max"]

        sigma_mec = (
            (3.0 + poissons_rotor) * rho_rotor * rotor_diameter**2.0 * np.pi * rpm_max**2.0 / 480.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_max",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_poissons_ratio",
        ] = np.array(
            rho_rotor * rotor_diameter**2.0 * np.pi * rpm_max**2.0 / 480.0
            if (sigma_mec < sigma_yield)
            else 0.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_max",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density",
        ] = np.array(
            (3.0 + poissons_rotor) * rotor_diameter**2.0 * np.pi * rpm_max**2.0 / 480.0
            if (sigma_mec < sigma_yield)
            else 0.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_max",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = np.array(
            (3.0 + poissons_rotor) * rho_rotor * rotor_diameter * np.pi * rpm_max**2.0 / 240.0
            if (sigma_mec < sigma_yield)
            else 0.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_max",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_max",
        ] = np.array(
            (3.0 + poissons_rotor) * rho_rotor * rotor_diameter**2.0 * np.pi * rpm_max / 240.0
            if (sigma_mec < sigma_yield)
            else 0.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mechanical_stress_max",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_yield_stress",
        ] = np.array(0.0 if (sigma_mec < sigma_yield) else 1.0)
