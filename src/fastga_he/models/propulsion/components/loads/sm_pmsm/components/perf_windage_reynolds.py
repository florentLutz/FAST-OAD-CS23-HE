# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import DEFAULT_DENSITY


class PerformancesWindageReynolds(om.ExplicitComponent):
    """
    Computation of the Reynold's numbers for the friction coefficients of motor windage. The
    formulas are given by equation (II-72) and (II-77) in :cite:`touhami:2020`.
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

        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input(
            "density",
            shape=number_of_points,
            val=DEFAULT_DENSITY,
            units="kg/m**3",
        )
        self.add_input(
            "dynamic_viscosity",
            shape=number_of_points,
            val=np.nan,
            units="kg/m/s",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":airgap_thickness",
            val=np.nan,
            units="m",
            desc="The distance between the rotor and the stator bore",
        )

        self.add_output(
            "airgap_reynolds_number",
            val=np.full(number_of_points, 3e4),
            desc="The reynolds number of the airgap between the rotor and stator.",
        )
        self.add_output(
            "rotor_end_reynolds_number",
            val=np.full(number_of_points, 9e5),
            desc="The reynolds number of the gap two rotor ends and the casing",
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt=["rpm", "density", "dynamic_viscosity"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="airgap_reynolds_number",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":airgap_thickness",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        rpm = inputs["rpm"]
        rho_air = inputs["density"]
        mu_air = inputs["dynamic_viscosity"]
        rotor_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] / 2.0
        )
        airgap_thickness = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":airgap_thickness"
        ]
        omega = 2.0 * np.pi * rpm / 60.0  # Mechanical angular speed [rad/s]

        outputs["rotor_end_reynolds_number"] = (rho_air * rotor_radius**2.0 * omega) / mu_air
        outputs["airgap_reynolds_number"] = (
            rho_air * rotor_radius * airgap_thickness * omega
        ) / mu_air

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        rpm = inputs["rpm"]
        rho_air = inputs["density"]
        mu_air = inputs["dynamic_viscosity"]
        rotor_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] / 2.0
        )
        airgap_thickness = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":airgap_thickness"
        ]
        omega = 2.0 * np.pi * rpm / 60.0  # Mechanical angular speed [rad/s]

        partials[
            "airgap_reynolds_number",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = (0.5 * rho_air * airgap_thickness * omega) / mu_air

        partials[
            "airgap_reynolds_number",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":airgap_thickness",
        ] = (rho_air * rotor_radius * omega) / mu_air

        partials["airgap_reynolds_number", "rpm"] = (
            np.ones_like(omega)
            * (rho_air * rotor_radius * airgap_thickness * np.pi)
            / (30.0 * mu_air)
        )

        partials["airgap_reynolds_number", "density"] = (
            rotor_radius * airgap_thickness * omega
        ) / mu_air

        partials["airgap_reynolds_number", "dynamic_viscosity"] = (
            -(rho_air * rotor_radius * airgap_thickness * omega) / mu_air**2.0
        )

        partials[
            "rotor_end_reynolds_number",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = (rho_air * rotor_radius * omega) / mu_air

        partials["rotor_end_reynolds_number", "rpm"] = (
            np.ones_like(omega) * (rho_air * rotor_radius**2.0 * np.pi) / (30.0 * mu_air)
        )

        partials["rotor_end_reynolds_number", "density"] = (rotor_radius**2.0 * omega) / mu_air

        partials["rotor_end_reynolds_number", "dynamic_viscosity"] = (
            -(rho_air * rotor_radius**2.0 * omega) / mu_air**2.0
        )
