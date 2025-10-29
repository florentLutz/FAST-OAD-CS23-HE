# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingShaftDiameter(om.ExplicitComponent):
    """
    Computation of the shaft diameter of a SM PMSM based on its mechanical limit. The formula is
    obtained from equation (II-81) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
            val=np.nan,
            units="min**-1",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_poissons_ratio",
            val=0.29,
            desc="The shaft material Poisson's ratio, 4340 steel alloy is set to default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_material_density",
            units="kg/m**3",
            val=7850.0,
            desc="The shaft material density, 4340 steel alloy is set to default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_yield_stress",
            units="Pa",
            desc="The yield stress of the shaft material, 4340 steel alloy is set to default",
            val=0.74e9,
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_diameter",
            units="m",
            val=0.02,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        rotor_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"
        ]
        rpm = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating"]
        poissons_ratio = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_poissons_ratio"
        ]
        rho_shaft = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_material_density"
        ]
        sigma_yield = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_yield_stress"
        ]
        mechanical_limit_diameter = 120.0 * np.sqrt(
            2.0 * sigma_yield / (rho_shaft * rpm**2.0 * np.pi**2.0 * (3.0 + poissons_ratio))
        )

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_diameter"] = (
            rotor_diameter / 3.0
            if rotor_diameter / 3.0 <= mechanical_limit_diameter
            else mechanical_limit_diameter
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        rotor_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"
        ]
        rpm = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating"]
        poissons_ratio = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_poissons_ratio"
        ]
        rho_shaft = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_material_density"
        ]
        sigma_yield = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_yield_stress"
        ]
        mechanical_limit_diameter = 120.0 * np.sqrt(
            2.0 * sigma_yield / (rho_shaft * rpm**2.0 * np.pi**2.0 * (3.0 + poissons_ratio))
        )

        if rotor_diameter / 3.0 <= mechanical_limit_diameter:
            partials[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_diameter",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            ] = 1.0 / 3.0

        else:
            partials[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_diameter",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_yield_stress",
            ] = 120.0 / np.sqrt(
                2.0 * sigma_yield * (3.0 + poissons_ratio) * rho_shaft * rpm**2.0 * np.pi**2.0
            )

            partials[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_diameter",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_material_density",
            ] = (
                -120.0
                * sigma_yield
                / np.sqrt(
                    2.0
                    * sigma_yield
                    * (3.0 + poissons_ratio)
                    * rho_shaft**3.0
                    * rpm**2.0
                    * np.pi**2.0
                )
            )

            partials[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_diameter",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
            ] = -mechanical_limit_diameter / rpm

            partials[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_diameter",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":shaft_poissons_ratio",
            ] = (
                -120.0
                * sigma_yield
                / np.sqrt(
                    2.0
                    * sigma_yield
                    * (3.0 + poissons_ratio) ** 3.0
                    * rho_shaft
                    * rpm**2.0
                    * np.pi**2.0
                )
            )
