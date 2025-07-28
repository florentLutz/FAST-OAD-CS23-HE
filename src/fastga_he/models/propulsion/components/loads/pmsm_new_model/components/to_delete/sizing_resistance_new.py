# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingResistanceNew(om.ExplicitComponent):
    """
    Computation of the Resistance (all phases).

    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

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
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
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
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_conductor_factor",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":winding_temperature",
            val=np.nan,
            units="degC",
        )

        self.add_output(
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
            units="ohm",
            val=0.0,
            shape=number_of_points,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
            wrt=[
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_twisting_coeff",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_conductor_factor",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":winding_temperature",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        q = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases"]
        m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases"]
        p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
        Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
        k_lc = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_twisting_coeff"]
        k_tb = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff"]
        hs = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height"]
        ls = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width"]
        k_fill = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor"]
        k_sc = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_conductor_factor"]
        T_win = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":winding_temperature"]
        rho_cu_20 = 1.68e-8  # Copper resistivity at 20°C [Ohm·m]
        alpha_th = 0.00393  # Temperature coefficient for copper [1/°C]

        rho_cu_Twin = rho_cu_20 * (1 + alpha_th * (T_win - 20))
        S_slot = hs * ls
        S_cond = S_slot * k_sc * k_fill
        l_c = Lm * k_lc * k_tb
        N_c = 2 * p * q * m
        R_s = N_c * rho_cu_Twin * l_c / S_cond

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance"] = R_s

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     pmsm_id = self.options["pmsm_id"]
    #     q = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases"]
    #     m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases"]
    #     p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
    #     Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
    #     k_lc = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_twisting_coeff"]
    #     k_tb = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff"]
    #     hs = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height"]
    #     ls = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width"]
    #     k_fill = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor"]
    #     k_sc = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_conductor_factor"]
    #     T_win = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":winding_temperature"]
    #     rho_cu_20 = 1.68e-8  # Copper resistivity at 20°C [Ohm·m]
    #     alpha_th = 0.00393  # Temperature coefficient for copper [1/°C]
    #
    #     rho_cu_Twin = rho_cu_20 * (1 + alpha_th * (T_win - 20))
    #     S_slot = hs * ls
    #     S_cond = S_slot * k_sc * k_fill
    #     l_c = Lm * k_lc * k_tb
    #     N_c = 2 * p * q * m
    #     R_s = N_c * rho_cu_Twin * l_c / S_cond
    #
    #     dR_dp = 2 * q * m * rho_cu_Twin * l_c / S_cond
    #     dR_dq = 2 * p * m * rho_cu_Twin * l_c / S_cond
    #     dR_dm = 2 * p * q * rho_cu_Twin * l_c / S_cond
    #     dR_dLm = N_c * rho_cu_Twin * k_lc * k_tb / S_cond
    #     dR_dklc = N_c * rho_cu_Twin * Lm * k_tb / S_cond
    #     dR_dktb = N_c * rho_cu_Twin * Lm * k_lc / S_cond
    #     dR_dTwin = N_c * rho_cu_20 * alpha_th * l_c / S_cond
    #     dR_dhs = -(N_c * rho_cu_Twin * l_c) / (hs**2 * ls * k_sc * k_fill)
    #     dR_dls = -(N_c * rho_cu_Twin * l_c) / (hs * ls**2 * k_sc * k_fill)
    #     dR_dksc = -(N_c * rho_cu_Twin * l_c) / (hs * ls * k_sc**2 * k_fill)
    #     dR_dkfill = -(N_c * rho_cu_Twin * l_c) / (hs * ls * k_sc * k_fill**2)
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
    #     ] = dR_dp
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":number_of_phases",
    #     ] = dR_dq
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slots_per_poles_phases",
    #     ] = dR_dm
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
    #     ] = dR_dLm
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":cond_twisting_coeff",
    #     ] = dR_dklc
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff",
    #     ] = dR_dktb
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":winding_temperature",
    #     ] = dR_dTwin
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_height",
    #     ] = dR_dhs
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
    #     ] = dR_dls
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_fill_factor",
    #     ] = dR_dkfill
    #
    #     partials[
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":resistance",
    #         "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_conductor_factor",
    #     ] = dR_dksc
