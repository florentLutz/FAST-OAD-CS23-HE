# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingActiveLength(om.ExplicitComponent):
    """
    Computation of the length in the SM PMSM that is electromagnetically active. The formula is
    obtained from equation (II-44) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
            val=np.nan,
            desc="The factor to account extra length from end winding",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density",
            units="kg/m**3",
            val=7850.0,
            desc="The rotor material density, 4340 steel alloy is set to default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_youngs_modules",
            units="Pa",
            val=19.0e10,
            desc="Young's modules of the rotor material, 4340 steel alloy is set to default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
            val=np.nan,
            units="min**-1",
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
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_rating",
            units="N*m",
            val=np.nan,
            desc="Max electromagnetic torque of the motor",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            units="m",
            desc="The length of electromagnetism active part of SM PMSM",
            val=0.1897,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        e_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_youngs_modules"
        ]
        rho_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density"
        ]
        rpm = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating"]
        end_winding_coeff = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff"
        ]
        rotor_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"
        ]
        sigma_t = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber"
        ]
        torque_em = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_rating"
        ]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"] = (
            64.0
            / np.pi
            * (15.0 * e_rotor / (rho_rotor * rpm**2.0 * end_winding_coeff**4.0)) ** (6.0 / 7.0)
            * (rotor_diameter * torque_em / (4.0 * sigma_t)) ** (1.0 / 7.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        e_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_youngs_modules"
        ]
        rho_rotor = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density"
        ]
        rpm = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating"]
        end_winding_coeff = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff"
        ]
        rotor_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"
        ]
        sigma_t = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber"
        ]
        torque_em = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_rating"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = (
            64.0
            / (np.pi * 7.0)
            * (15.0 * e_rotor / (rotor_diameter * rho_rotor * rpm**2.0 * end_winding_coeff**4.0))
            ** (6.0 / 7.0)
            * (torque_em / (4.0 * sigma_t)) ** (1.0 / 7.0)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":electromagnetic_torque_rating",
        ] = (
            64.0
            / (np.pi * 7.0)
            * (15.0 * e_rotor / (torque_em * rho_rotor * rpm**2.0 * end_winding_coeff**4.0))
            ** (6.0 / 7.0)
            * (rotor_diameter / (4.0 * sigma_t)) ** (1.0 / 7.0)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":tangential_stress_caliber",
        ] = (
            -64.0
            / (np.pi * 7.0)
            * (15.0 * e_rotor / (rho_rotor * rpm**2.0 * end_winding_coeff**4.0)) ** (6.0 / 7.0)
            * (rotor_diameter * torque_em / (4.0 * sigma_t**8.0)) ** (1.0 / 7.0)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_youngs_modules",
        ] = (
            384.0
            / (np.pi * 7.0)
            * (15.0 / (rho_rotor * rpm**2.0 * end_winding_coeff**4.0)) ** (6.0 / 7.0)
            * (rotor_diameter * torque_em / (4.0 * e_rotor * sigma_t**8.0)) ** (1.0 / 7.0)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_material_density",
        ] = (
            -384.0
            / (np.pi * 7.0)
            * (15.0 * e_rotor / (rpm**2.0 * end_winding_coeff**4.0)) ** (6.0 / 7.0)
            * (rotor_diameter * torque_em / (4.0 * sigma_t * rho_rotor**13.0)) ** (1.0 / 7.0)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rpm_rating",
        ] = (
            -768.0
            / (np.pi * 7.0)
            * (15.0 * e_rotor / (rho_rotor * end_winding_coeff**4.0)) ** (6.0 / 7.0)
            * (rotor_diameter * torque_em / (4.0 * sigma_t * rpm**19.0)) ** (1.0 / 7.0)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
        ] = (
            1536.0
            / (np.pi * 7.0)
            * (15.0 * e_rotor / (rho_rotor * rpm**2.0)) ** (6.0 / 7.0)
            * (rotor_diameter * torque_em / (4.0 * sigma_t * end_winding_coeff**31.0))
            ** (1.0 / 7.0)
        )
