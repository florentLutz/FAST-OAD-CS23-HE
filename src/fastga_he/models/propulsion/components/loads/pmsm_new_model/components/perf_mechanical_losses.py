# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMechanicalLosses(om.ExplicitComponent):
    """
    Computation of the Mechanical losses.

    """

    def initialize(self):
        # Reference motor : HASTECS project, Sarah Touhami

        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Airgap_thickness",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight",
            val=np.nan,
            units="kg",
        )

        self.add_output(
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses",
            units="kW",
            val=0.0,
            shape=number_of_points,
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses",
            wrt=["rpm"],
            method="fd",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses",
            wrt=[
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Airgap_thickness",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff",
                "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight",
            ],
            method="fd",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        RPM = inputs["rpm"]
        Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
        k_tb = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff"]
        R_r = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter"] / 2
        e_g = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Airgap_thickness"]
        W_rot = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight"]
        Cf_bearing = 0.0015  # Bearing friction coefficient
        d_bearing = 0.03  # Bearing bore diameter [m]
        g = 9.81  # Gravity [m/s²]
        T = 300  # Air temperature [K]
        k1 = 1  # Smoothness factor
        pr = 1  # Air pressure [atm]
        Omega = 2 * np.pi * RPM / 60  # Mechanical angular speed [rad/s]
        R_sh = R_r / 3  # shaft radius
        L = Lm * k_tb

        # Air properties
        mu_air = 8.88e-15 * T**3 - 3.23e-11 * T**2 + 6.26e-8 * T + 2.35e-6
        rho_air = 1.293 * (273.15 / T) * pr

        Re_a = (rho_air * R_r * e_g * Omega) / mu_air
        Cf_a = np.zeros_like(Re_a)

        # Conditions
        mask_1 = (Re_a > 500) & (Re_a < 1e4)
        mask_2 = Re_a >= 1e4
        mask_3 = Re_a <= 500

        Cf_a[mask_1] = 0.515 * (e_g / R_r) ** 0.3 * (Re_a[mask_1] ** -0.5)
        Cf_a[mask_2] = 0.0325 * (e_g / R_r) ** 0.3 * (Re_a[mask_2] ** -0.2)
        Cf_a[mask_3] = 0.0

        # Eventuale gestione di valori troppo bassi
        # if np.any(Re_a < 500):
        #     raise ValueError("Re_a is too low (Re_a < 500)")

        P_windage_airgap = k1 * Cf_a * np.pi * rho_air * R_r**4 * Omega**3 * L

        # Rotor windage
        Re_rot = (rho_air * R_r**2 * Omega) / mu_air
        C_fr = np.where(Re_rot < 3e5, 3.87 / Re_rot**0.5, 0.146 / Re_rot**0.2)

        P_windage_rotor = 0.5 * C_fr * np.pi * rho_air * Omega**3 * (R_r**5 - R_sh**5)

        # Bearing friction losses
        P_eq = W_rot * g
        P_friction = 0.5 * Cf_bearing * P_eq * d_bearing * Omega

        # Total mechanical losses
        P_mec_loss = (P_windage_airgap + 2 * P_windage_rotor) + (2 * P_friction)

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mechanical_power_losses"] = (
            P_mec_loss
        ) /1000.0

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     pmsm_id = self.options["pmsm_id"]
    #     RPM = inputs["rpm"]
    #     Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
    #     k_tb = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff"]
    #     R_r = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rot_diameter"] / 2
    #     e_g = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Airgap_thickness"]
    #     W_rot = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight"]
    #     Cf_bearing = 0.0015  # Bearing friction coefficient
    #     d_bearing = 0.03  # Bearing bore diameter [m]
    #     g = 9.81  # Gravity [m/s²]
    #     T = 300  # Air temperature [K]
    #     k1 = 1  # Smoothness factor
    #     pr = 1  # Air pressure [atm]
    #     Omega = 2 * np.pi * RPM / 60  # Mechanical angular speed [rad/s]
    #     R_sh = R_r / 3  # shaft radius
    #     L = Lm * k_tb
    #
    #     # Air properties
    #     mu_air = 8.88e-15 * T**3 - 3.23e-11 * T**2 + 6.26e-8 * T + 2.35e-6
    #     rho_air = 1.293 * (273.15 / T) * pr
    #
    #     Re_a = (rho_air * R_r * e_g * Omega) / mu_air
    #
    #     Cf_a = np.zeros_like(Re_a)
    #
    #     # Conditions
    #     mask_1 = (Re_a > 500) & (Re_a < 1e4)
    #     mask_2 = Re_a >= 1e4
    #
    #     Cf_a[mask_1] = 0.515 * (e_g / R_r) ** 0.3 * (Re_a[mask_1] ** -0.5)
    #     Cf_a[mask_2] = 0.0325 * (e_g / R_r) ** 0.3 * (Re_a[mask_2] ** -0.2)
    #
    #     # Eventuale gestione di valori troppo bassi
    #     if np.any(Re_a < 500):
    #         raise ValueError("Re_a is too low (Re_a < 500)")
    #
    #     P_windage_airgap = k1 * Cf_a * np.pi * rho_air * R_r**4 * Omega**3 * L
    #
    #     # Rotor windage
    #     Re_rot = (rho_air * R_r**2 * Omega) / mu_air
    #     C_fr = np.where(Re_rot < 3e5, 3.87 / Re_rot**0.5, 0.146 / Re_rot**0.2)
    #
    #     P_windage_rotor = 0.5 * C_fr * np.pi * rho_air * Omega**3 * (R_r**5 - R_sh**5)
    #
    #     # Bearing friction losses
    #     P_eq = W_rot * g
    #     P_friction = 0.5 * Cf_bearing * P_eq * d_bearing * Omega
    #
    #     # Total mechanical losses
    #     P_mec_loss = (P_windage_airgap + 2 * P_windage_rotor) + (2 * P_friction)
    #
    #     # partials[
    #     #    "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":Joule_power_losses",
    #     #    "ac_current_rms_in_one_phase",
    #     # ] = dP_dIrms
