# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import scipy.constants as sc


class PerformancesBearingLosses(om.ExplicitComponent):
    """
    Computation of the bearing friction losses result from friction at the contacting surface
    between the bearing and the motor shaft. This is obtained from equation (II.78) in
    :cite:`touhami:2020`. The bearing friction coefficient is given by :cite:`skf:2016`.
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
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_diameter",
            units="m",
            val=0.03,
            desc="The motor bearing bore diameter",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":bearing_friction_coefficient",
            val=0.0015,
            desc="friction coefficient of the motor that house the shaft, the ball bearing is set "
            "to default",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
            val=np.nan,
            units="kg",
        )

        self.add_output(
            "bearing_friction_losses",
            units="W",
            val=0.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="bearing_friction_losses",
            wrt="angular_speed",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="bearing_friction_losses",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_diameter",
                "data:propulsion:he_power_train:SM_PMSM:"
                + motor_id
                + ":bearing_friction_coefficient",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        omega = inputs["angular_speed"]
        rotor_mass = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass"]
        cf_bearing = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_friction_coefficient"
        ]
        bearing_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_diameter"
        ]

        outputs["bearing_friction_losses"] = (
            0.5 * cf_bearing * rotor_mass * sc.g * bearing_diameter * omega
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        omega = inputs["angular_speed"]
        rotor_mass = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass"]
        cf_bearing = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_friction_coefficient"
        ]
        bearing_diameter = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_diameter"
        ]

        partials[
            "bearing_friction_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
        ] = 0.5 * cf_bearing * sc.g * bearing_diameter * omega

        partials["bearing_friction_losses", "angular_speed"] = (
            0.5 * cf_bearing * rotor_mass * sc.g * bearing_diameter
        )

        partials[
            "bearing_friction_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_diameter",
        ] = 0.5 * cf_bearing * rotor_mass * sc.g * omega

        partials[
            "bearing_friction_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bearing_friction_coefficient",
        ] = 0.5 * rotor_mass * sc.g * bearing_diameter * omega
