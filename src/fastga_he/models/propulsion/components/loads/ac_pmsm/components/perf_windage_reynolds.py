# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesWindageReynolds(om.ExplicitComponent):
    """
    Computation of the reynold's numbers for the friction coefficients of motor windage. The
    formulas are given by equation (II-72) and (II-77) in :cite:`touhami:2020`.
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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_thickness",
            val=np.nan,
            units="m",
            desc="The distance between the rotor and the stator bore",
        )

        self.add_output(
            "airgap_reynolds_number",
            val=np.zeros(number_of_points),
            desc="The reynolds number of the airgap between the rotor and stator.",
        )
        self.add_output(
            "rotor_end_reynolds_number",
            val=np.zeros(number_of_points),
            desc="The reynolds number of the gap two rotor ends and the casing",
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="rpm",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_diameter",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="airgap_reynolds_number",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_thickness",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        rpm = inputs["rpm"]
        r_rot = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_diameter"] / 2.0
        )
        e_g = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_thickness"]
        temp = 300.0  # Air temperature [K]
        pr = 1.0  # Air pressure [atm]
        omega = 2.0 * np.pi * rpm / 60.0  # Mechanical angular speed [rad/s]

        # Air properties
        mu_air = 8.88e-15 * temp**3.0 - 3.23e-11 * temp**2.0 + 6.26e-8 * temp + 2.35e-6
        rho_air = 1.293 * (273.15 / temp) * pr

        outputs["rotor_end_reynolds_number"] = (rho_air * r_rot**2.0 * omega) / mu_air
        outputs["airgap_reynolds_number"] = (rho_air * r_rot * e_g * omega) / mu_air

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        rpm = inputs["rpm"]
        r_rot = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_diameter"] / 2.0
        )
        e_g = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_thickness"]
        temp = 300.0  # Air temperature [K]
        pr = 1.0  # Air pressure [atm]
        omega = 2.0 * np.pi * rpm / 60.0  # Mechanical angular speed [rad/s]

        # Air properties
        mu_air = 8.88e-15 * temp**3.0 - 3.23e-11 * temp**2.0 + 6.26e-8 * temp + 2.35e-6
        rho_air = 1.293 * (273.15 / temp) * pr

        partials[
            "airgap_reynolds_number",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_diameter",
        ] = (0.5 * rho_air * e_g * omega) / mu_air

        partials[
            "airgap_reynolds_number",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_thickness",
        ] = (rho_air * r_rot * omega) / mu_air

        partials["airgap_reynolds_number", "rpm"] = (
            np.ones_like(omega) * (rho_air * r_rot * e_g * np.pi) / (30.0 * mu_air)
        )

        partials[
            "rotor_end_reynolds_number",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":rotor_diameter",
        ] = (rho_air * r_rot * omega) / mu_air

        partials["rotor_end_reynolds_number", "rpm"] = (
            np.ones_like(omega) * (rho_air * r_rot**2.0 * np.pi) / (30.0 * mu_air)
        )
