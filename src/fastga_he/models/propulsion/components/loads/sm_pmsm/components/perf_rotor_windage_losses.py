# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesRotorWindageLoss(om.ExplicitComponent):
    """
    Computation of the rotor windage losses result from frictions between air at the gap of both
    ends and the rotor. This is obtained from equation (II.75) in :cite:`touhami:2020`.
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

        self.add_input("angular_speed", units="rad/s", val=np.nan, shape=number_of_points)
        self.add_input(
            "density",
            shape=number_of_points,
            val=np.nan,
            units="kg/m**3",
        )
        self.add_input(
            "rotor_end_friction_coeff",
            val=np.nan,
            shape=number_of_points,
            desc="Air friction coefficient at the two rotor ends",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            val=np.nan,
            units="m",
        )

        self.add_output(
            "rotor_windage_losses",
            units="W",
            val=0.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="rotor_windage_losses",
            wrt=["angular_speed", "density", "rotor_end_friction_coeff"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="rotor_windage_losses",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        omega = inputs["angular_speed"]
        rho_air = inputs["density"]
        rotor_end_friction_coeff = inputs["rotor_end_friction_coeff"]
        rotor_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] / 2.0
        )
        shaft_radius = rotor_radius / 3.0  # shaft radius

        outputs["rotor_windage_losses"] = (
            0.5
            * rotor_end_friction_coeff
            * np.pi
            * rho_air
            * omega**3.0
            * (rotor_radius**5.0 - shaft_radius**5.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        omega = inputs["angular_speed"]
        rho_air = inputs["density"]
        rotor_end_friction_coeff = inputs["rotor_end_friction_coeff"]
        rotor_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] / 2.0
        )
        shaft_radius = rotor_radius / 3.0  # shaft radius

        partials[
            "rotor_windage_losses",
            "rotor_end_friction_coeff",
        ] = 0.5 * np.pi * rho_air * omega**3.0 * (rotor_radius**5.0 - shaft_radius**5.0)

        partials[
            "rotor_windage_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = (
            1.25
            * rotor_end_friction_coeff
            * np.pi
            * rho_air
            * omega**3.0
            * rotor_radius**4.0
            * (1.0 - 3.0**-5.0)
        )

        partials["rotor_windage_losses", "angular_speed"] = (
            1.5
            * rotor_end_friction_coeff
            * np.pi
            * rho_air
            * omega**2.0
            * (rotor_radius**5.0 - shaft_radius**5.0)
        )

        partials["rotor_windage_losses", "density"] = (
            0.5
            * rotor_end_friction_coeff
            * np.pi
            * omega**3.0
            * (rotor_radius**5.0 - shaft_radius**5.0)
        )
