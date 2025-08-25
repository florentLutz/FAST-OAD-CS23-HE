# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMechanicalLosses(om.ExplicitComponent):
    """
    Computation of the mechanical losses result from frictions between gas and rotating solid or
    between rotating solid and stationary solid. This is obtained from part II.3.3 in
    :cite:`touhami:2020`.
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

        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input(
            "airgap_friction_coeff",
            val=np.nan,
            shape=number_of_points,
            desc="Air friction coefficient of airgap",
        )
        self.add_input(
            "rotor_end_friction_coeff",
            val=np.nan,
            shape=number_of_points,
            desc="Air friction coefficient at the two rotor ends",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
            desc="The stator length of PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff",
            val=np.nan,
            desc="The factor to account extra length from end winding",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight",
            val=np.nan,
            units="kg",
        )

        self.add_output(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mechanical_power_losses",
            units="W",
            val=0.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mechanical_power_losses",
            wrt=["rpm", "airgap_friction_coeff", "rotor_end_friction_coeff"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mechanical_power_losses",
            wrt=[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_diameter",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        rpm = inputs["rpm"]
        cf_a = inputs["airgap_friction_coeff"]
        cf_r = inputs["rotor_end_friction_coeff"]
        active_length = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"
        ]
        k_tb = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff"]
        r_rot = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_diameter"] / 2.0
        )
        w_rot = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight"]
        cf_bearing = 0.0015  # Bearing friction coefficient
        d_bearing = 0.03  # Bearing bore diameter [m]
        g = 9.81  # Gravity [m/s²]
        temp = 300.0  # Air temperature [K]
        k1 = 1.0  # Smoothness factor
        pr = 1.0  # Air pressure [atm]
        omega = 2.0 * np.pi * rpm / 60.0  # Mechanical angular speed [rad/s]
        r_sh = r_rot / 3.0  # shaft radius
        length_k = active_length * k_tb

        # Air properties
        rho_air = 1.293 * (273.15 / temp) * pr

        # Airgap windage
        p_windage_airgap = k1 * cf_a * np.pi * rho_air * r_rot**4.0 * omega**3.0 * length_k

        # Rotor windage
        p_windage_rotor = 0.5 * cf_r * np.pi * rho_air * omega**3.0 * (r_rot**5.0 - r_sh**5.0)

        # Bearing friction losses
        p_friction = 0.5 * cf_bearing * w_rot * g * d_bearing * omega

        # Total mechanical losses
        P_mec_loss = (p_windage_airgap + 2.0 * p_windage_rotor) + (2.0 * p_friction)

        outputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mechanical_power_losses"
        ] = P_mec_loss

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        rpm = inputs["rpm"]
        cf_a = inputs["airgap_friction_coeff"]
        cf_r = inputs["rotor_end_friction_coeff"]
        active_length = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"
        ]
        k_tb = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff"]
        r_rot = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_diameter"] / 2.0
        )
        w_rot = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight"]
        cf_bearing = 0.0015  # Bearing friction coefficient
        d_bearing = 0.03  # Bearing bore diameter [m]
        g = 9.81  # Gravity [m/s²]
        temp = 300.0  # Air temperature [K]
        k1 = 1.0  # Smoothness factor
        pr = 1.0  # Air pressure [atm]
        omega = 2.0 * np.pi * rpm / 60.0  # Mechanical angular speed [rad/s]
        r_sh = r_rot / 3.0  # shaft radius
        length_k = active_length * k_tb
        rho_air = 1.293 * (273.15 / temp) * pr

        # Air properties
        p_windage_airgap = k1 * cf_a * np.pi * rho_air * r_rot**4.0 * omega**3.0 * length_k

        # Rotor windage
        p_windage_rotor = 0.5 * cf_r * np.pi * rho_air * omega**3.0 * (r_rot**5.0 - r_sh**5.0)

        # Bearing friction losses
        p_friction = 0.5 * cf_bearing * w_rot * g * d_bearing * omega

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mechanical_power_losses",
            "airgap_friction_coeff",
        ] = k1 * np.pi * rho_air * r_rot**4.0 * omega**3.0 * length_k

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mechanical_power_losses",
            "rotor_end_friction_coeff",
        ] = np.pi * rho_air * omega**3.0 * (r_rot**5.0 - r_sh**5.0)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mechanical_power_losses",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
        ] = k1 * cf_a * np.pi * rho_air * r_rot**4.0 * omega**3.0 * k_tb

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mechanical_power_losses",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff",
        ] = k1 * cf_a * np.pi * rho_air * r_rot**4.0 * omega**3.0 * active_length

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mechanical_power_losses",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_diameter",
        ] = (
            2.0 * p_windage_airgap / r_rot
            + cf_r * np.pi * rho_air * omega**3.0 * 605.0 * r_rot**4.0 / 243.0
        )

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mechanical_power_losses",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_weight",
        ] = 2.0 * p_friction / w_rot

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":mechanical_power_losses", "rpm"
        ] = (
            0.1 * p_windage_airgap * np.pi / omega + 0.2 * p_windage_rotor * np.pi / omega
        ) + p_friction * np.pi / (15.0 * omega)
