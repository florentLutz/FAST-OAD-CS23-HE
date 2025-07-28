# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRotorWeight(om.ExplicitComponent):
    """Computation of the rotor weight of the PMSM."""

    def initialize(self):
        # Reference motor : HASTECS project, Sarah Touhami

        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        # self.options.declare(
        # "diameter_ref",
        # default=0.268,
        # desc="Diameter of the reference motor in [m]",
        # )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter",
            val=np.nan,
            units="m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight",
            units="kg",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
        p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
        R_r = (inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter"]) / 2

        if p <= 10:
            rho_rot = -431.67 * p + 7932
        elif 10 < p <= 50:
            rho_rot = 1.09 * p**2 - 117.45 * p + 4681
        else:
            rho_rot = 1600

        # Rotor weight
        W_rot = np.pi * R_r**2 * Lm * rho_rot

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight"] = W_rot

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     pmsm_id = self.options["pmsm_id"]
    #     Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
    #     p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
    #     R_r = (inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter"]) / 2
    #
    #     if p <= 10:
    #         rho_rot = -431.67 * p + 7932
    #         drho_dp = -431.67
    #     elif 10 < p <= 50:
    #         rho_rot = 1.09 * p**2 - 117.45 * p + 4681
    #         drho_dp = 2 * 1.09 * p - 117.45
    #     else:
    #         rho_rot = 1600
    #         drho_dp = 0
    #
    #     # Rotor weight
    #     dW_dDr = np.pi * R_r * Lm * rho_rot
    #     dW_dLm = np.pi * R_r**2 * rho_rot
    #     dW_dp = np.pi * R_r**2 * Lm * drho_dp
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
    #     ] = dW_dLm
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter",
    #     ] = dW_dDr
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
    #     ] = dW_dp
