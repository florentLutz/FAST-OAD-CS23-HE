# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDensityCurrent(om.ExplicitComponent):
    """
    Computation of the Density Current.

    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":pole_pairs_number",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":number_of_phases",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":slots_per_poles_phases",
            val=np.nan,
        )

        self.add_input(
            "ac_current_rms_in_one_phase",
            units="A",
            val=np.full(number_of_points, np.nan),
        )

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":active_length",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":cond_twisting_coeff",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":end_winding_coeff",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_height",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_width",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_fill_factor",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_conductor_factor",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:PMSM:" + motor_id + ":winding_temperature",
            val=np.nan,
            units="degC",
        )

        self.add_output(
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            units="W",
            val=0.0,
            shape=number_of_points,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            wrt=["ac_current_rms_in_one_phase"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            wrt=[
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":pole_pairs_number",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":number_of_phases",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":slots_per_poles_phases",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":active_length",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":cond_twisting_coeff",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":end_winding_coeff",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_height",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_width",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_fill_factor",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_conductor_factor",
                "data:propulsion:he_power_train:PMSM:" + motor_id + ":winding_temperature",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]
        I_rms = inputs["ac_current_rms_in_one_phase"]
        hs = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_height"]
        ls = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_width"]
        k_fill = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_fill_factor"]
        k_sc = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_conductor_factor"]

        S_slot = hs * ls
        S_cond = S_slot * k_sc * k_fill
        j_rms = I_rms / S_cond

        outputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses"] = P_j

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]
        I_rms = inputs["ac_current_rms_in_one_phase"]
        q = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":number_of_phases"]
        m = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":slots_per_poles_phases"]
        p = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":pole_pairs_number"]
        Lm = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":active_length"]
        k_lc = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":cond_twisting_coeff"]
        k_tb = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":end_winding_coeff"]
        hs = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_height"]
        ls = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_width"]
        k_fill = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_fill_factor"]
        k_sc = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_conductor_factor"]
        T_win = inputs["data:propulsion:he_power_train:PMSM:" + motor_id + ":winding_temperature"]
        rho_cu_20 = 1.68e-8  # Copper resistivity at 20°C [Ohm·m]
        alpha_th = 0.00393  # Temperature coefficient for copper [1/°C]

        rho_cu_Twin = rho_cu_20 * (1 + alpha_th * (T_win - 20))
        S_slot = hs * ls
        S_cond = S_slot * k_sc * k_fill
        l_c = Lm * k_lc * k_tb
        N_c = 2 * p * q * m
        R_s = N_c * rho_cu_Twin * l_c / S_cond
        P_j = R_s * I_rms**2

        dP_dp = 2 * q * m * rho_cu_Twin * l_c * I_rms**2 / S_cond
        dP_dq = 2 * p * m * rho_cu_Twin * l_c * I_rms**2 / S_cond
        dP_dm = 2 * p * q * rho_cu_Twin * l_c * I_rms**2 / S_cond
        dP_dLm = N_c * rho_cu_Twin * k_lc * k_tb * I_rms**2 / S_cond
        dP_dklc = N_c * rho_cu_Twin * Lm * k_tb * I_rms**2 / S_cond
        dP_dktb = N_c * rho_cu_Twin * Lm * k_lc * I_rms**2 / S_cond
        dP_dTwin = N_c * rho_cu_20 * alpha_th * l_c * I_rms**2 / S_cond
        dP_dhs = -(N_c * rho_cu_Twin * l_c * I_rms**2) / (hs**2 * ls * k_sc * k_fill)
        dP_dls = -(N_c * rho_cu_Twin * l_c * I_rms**2) / (hs * ls**2 * k_sc * k_fill)
        dP_dksc = -(N_c * rho_cu_Twin * l_c * I_rms**2) / (hs * ls * k_sc**2 * k_fill)
        dP_dkfill = -(N_c * rho_cu_Twin * l_c * I_rms**2) / (hs * ls * k_sc * k_fill**2)
        dP_dIrms = 2 * R_s * I_rms

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":pole_pairs_number",
        ] = dP_dp

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":number_of_phases",
        ] = dP_dq

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":slots_per_poles_phases",
        ] = dP_dm

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":active_length",
        ] = dP_dLm

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":cond_twisting_coeff",
        ] = dP_dklc
        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":end_winding_coeff",
        ] = dP_dktb

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":winding_temperature",
        ] = dP_dTwin

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_height",
        ] = dP_dhs

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_width",
        ] = dP_dls

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_fill_factor",
        ] = dP_dkfill

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":slot_conductor_factor",
        ] = dP_dksc

        partials[
            "data:propulsion:he_power_train:PMSM:" + motor_id + ":Joule_power_losses",
            "ac_current_rms_in_one_phase",
        ] = dP_dIrms
