# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRotorWeight(om.ExplicitComponent):
    """
    Computation of the rotor weight of the SM PMSM. The formula and the conditions are obtained
    from part II.2.6b in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
            val=np.nan,
            desc="Number of the north and south pairs in the SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
            val=np.nan,
            units="m",
            desc="The length of electromagnetism active part of SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            val=np.nan,
            units="m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
            units="kg",
            val=9.3,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        active_length = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"
        ]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]
        rotor_radius = (
            (inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"]) / 2.0
        )

        conditions = [num_pole_pairs <= 10.0, 10.0 < num_pole_pairs <= 50.0]

        rho_rotor = [
            -431.67 * num_pole_pairs + 7932.0,
            1.09 * num_pole_pairs**2.0 - 117.45 * num_pole_pairs + 4681.0,
        ]

        # Rotor weight
        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass"] = (
            np.pi
            * rotor_radius**2.0
            * active_length
            * np.select(conditions, rho_rotor, default=1600.0)
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        lm = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length"]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]
        rotor_radius = (
            (inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"]) / 2.0
        )

        conditions = [num_pole_pairs <= 10.0, 10.0 < num_pole_pairs <= 50.0]

        rho_rotor = [
            -431.67 * num_pole_pairs + 7932.0,
            1.09 * num_pole_pairs**2.0 - 117.45 * num_pole_pairs + 4681.0,
        ]
        d_rho_d_p = [-431.67, 2.0 * 1.09 * num_pole_pairs - 117.45]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":active_length",
        ] = np.pi * rotor_radius**2.0 * np.select(conditions, rho_rotor, default=1600.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
        ] = np.pi * rotor_radius * lm * np.select(conditions, rho_rotor, default=1600.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
        ] = np.pi * rotor_radius**2.0 * lm * np.select(conditions, d_rho_d_p, default=0.0)
