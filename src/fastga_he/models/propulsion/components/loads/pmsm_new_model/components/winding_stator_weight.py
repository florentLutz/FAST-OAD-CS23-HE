# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorWindingWeight(om.ExplicitComponent):
    """Computation of the stator core weight of the PMSM."""

    def initialize(self):
        # Reference motor : EMRAX 268

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
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_twisting_coeff",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_mat_density",
            val=np.nan,
            units="kg/m**3",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":insul_mat_density",
            val=np.nan,
            units="kg/m**3",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            units="kg",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_mat_density",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":insul_mat_density",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_twisting_coeff",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        q = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases"]
        m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases"]
        p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
        Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
        hs = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height"]
        ls = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width"]
        rho_c = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_mat_density"]
        rho_insl = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":insul_mat_density"]
        k_fill = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor"]
        k_tb = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff"]
        k_lc = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_twisting_coeff"]

        Ns = 2 * p * q * m
        vol_wind = k_tb * k_lc * hs * Lm * Ns * ls
        mat_mix_density = k_fill * rho_c + (1 - k_fill) * rho_insl
        W_stat_wind = vol_wind * mat_mix_density

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight"] = (
            W_stat_wind
        )

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     pmsm_id = self.options["pmsm_id"]
    #     q = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases"]
    #     m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases"]
    #     p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
    #     Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
    #     hs = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height"]
    #     ls = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width"]
    #     rho_c = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_mat_density"]
    #     rho_insl = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":insul_mat_density"]
    #     k_fill = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor"]
    #     k_tb = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff"]
    #     k_lc = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_twisting_coeff"]
    #
    #     Ns = 2 * p * q * m
    #     vol_wind = k_tb * k_lc * hs * Lm * Ns * ls
    #     mat_mix_density = k_fill * rho_c + (1 - k_fill) * rho_insl
    #
    #     dW_dktb = k_lc * hs * Lm * Ns * ls * mat_mix_density
    #     dW_dklc = k_tb * hs * Lm * Ns * ls * mat_mix_density
    #     dW_dhs = k_tb * k_lc * Lm * Ns * ls * mat_mix_density
    #     dW_dLm = k_tb * k_lc * hs * Ns * ls * mat_mix_density
    #     dW_dls = k_tb * k_lc * hs * Lm * Ns * mat_mix_density
    #     dW_dp = k_tb * k_lc * hs * Lm * q * m * ls * mat_mix_density
    #     dW_dq = k_tb * k_lc * hs * Lm * p * m * ls * mat_mix_density
    #     dW_dm = k_tb * k_lc * hs * Lm * p * q * ls * mat_mix_density
    #
    #     dW_dkfill = vol_wind * (rho_c - rho_insl)
    #     dW_drhoc = vol_wind * (k_fill)
    #     dW_drhoinsl = vol_wind * (1 - k_fill)
    #
    #     # Equation II-46: Slot height hs
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff",
    #     ] = dW_dktb
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_twisting_coeff",
    #     ] = dW_dklc
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height",
    #     ] = dW_dhs
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
    #     ] = dW_dls
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
    #     ] = dW_dLm
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
    #     ] = dW_dp
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases",
    #     ] = dW_dq
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases",
    #     ] = dW_dm
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor",
    #     ] = dW_dkfill
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_mat_density",
    #     ] = dW_drhoc
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":insul_mat_density",
    #     ] = dW_drhoinsl
