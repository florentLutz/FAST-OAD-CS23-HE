# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRotorWeight(om.ExplicitComponent):
    """
    Computation of the rotor weight of the PMSM. The formula and the conditions are obtained
    from part II.2.6b in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
            val=np.nan,
            desc="Number of the north and south pairs in the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            val=np.nan,
            units="m",
            desc="The stator length of PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            val=np.nan,
            units="m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
            units="kg",
            val=9.3,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        lm = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"]
        p = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"]
        r_r = (
            (inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"]) / 2.0
        )

        if p <= 10.0:
            rho_rot = -431.67 * p + 7932.0
        elif 10.0 < p <= 50.0:
            rho_rot = 1.09 * p**2.0 - 117.45 * p + 4681.0
        else:
            rho_rot = 1600.0

        # Rotor weight
        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass"] = (
            np.pi * r_r**2.0 * lm * rho_rot
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        lm = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"]
        p = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"]
        r_r = (
            (inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"]) / 2.0
        )

        if p <= 10.0:
            rho_rot = -431.67 * p + 7932.0
            drho_dp = -431.67
        elif 10.0 < p <= 50.0:
            rho_rot = 1.09 * p**2.0 - 117.45 * p + 4681.0
            drho_dp = 2.0 * 1.09 * p - 117.45
        else:
            rho_rot = 1600.0
            drho_dp = 0.0

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
        ] = np.pi * r_r**2.0 * rho_rot

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = np.pi * r_r * lm * rho_rot

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
        ] = np.pi * r_r**2.0 * lm * drho_dp
