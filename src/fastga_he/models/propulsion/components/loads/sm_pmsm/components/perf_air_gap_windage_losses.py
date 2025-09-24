# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from ..constants import DEFAULT_DENSITY


class PerformancesAirGapWindageLosses(om.ExplicitComponent):
    """
    Computation of the air gap windage losses result from frictions of the air gap between the rotor
    and the stator bore. This is obtained from equation II.70 in :cite:`touhami:2020`.
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
            "air_gap_friction_coeff",
            val=np.nan,
            shape=number_of_points,
            desc="Air friction coefficient of air gap",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            val=np.nan,
            units="m",
            desc="The stator length of PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
            val=np.nan,
            desc="The factor to account extra length from end winding",
        )

        self.add_output(
            "air_gap_windage_losses",
            units="W",
            val=0.0,
            shape=number_of_points,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="air_gap_windage_losses",
            wrt=["rpm", "air_gap_friction_coeff", "density"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="air_gap_windage_losses",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        rpm = inputs["rpm"]
        rho_air = inputs["density"]
        cf_air_gap = inputs["air_gap_friction_coeff"]
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        end_winding_coeff = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff"
        ]
        rotor_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] / 2.0
        )
        omega = 2.0 * np.pi * rpm / 60.0  # Mechanical angular speed [rad/s]
        air_gap_length = active_length * end_winding_coeff

        outputs["air_gap_windage_losses"] = (
            cf_air_gap * np.pi * rho_air * rotor_radius**4.0 * omega**3.0 * air_gap_length
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        rpm = inputs["rpm"]
        rho_air = inputs["density"]
        cf_air_gap = inputs["air_gap_friction_coeff"]
        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        end_winding_coeff = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff"
        ]
        rotor_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] / 2.0
        )
        omega = 2.0 * np.pi * rpm / 60.0  # Mechanical angular speed [rad/s]
        air_gap_length = active_length * end_winding_coeff

        partials[
            "air_gap_windage_losses",
            "air_gap_friction_coeff",
        ] = np.pi * rho_air * rotor_radius**4.0 * omega**3.0 * air_gap_length

        partials[
            "air_gap_windage_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
        ] = cf_air_gap * np.pi * rho_air * rotor_radius**4.0 * omega**3.0 * end_winding_coeff

        partials[
            "air_gap_windage_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":end_winding_coeff",
        ] = cf_air_gap * np.pi * rho_air * rotor_radius**4.0 * omega**3.0 * active_length

        partials[
            "air_gap_windage_losses",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = 2.0 * cf_air_gap * np.pi * rho_air * rotor_radius**3.0 * omega**3.0 * air_gap_length

        partials["air_gap_windage_losses", "rpm"] = (
            0.1
            * cf_air_gap
            * np.pi**2.0
            * rho_air
            * rotor_radius**4.0
            * omega**2.0
            * air_gap_length
        )

        partials["air_gap_windage_losses", "density"] = (
            cf_air_gap * np.pi * rotor_radius**4.0 * omega**3.0 * air_gap_length
        )
