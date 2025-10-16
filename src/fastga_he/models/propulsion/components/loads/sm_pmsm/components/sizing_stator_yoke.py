# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorYokeHeight(om.ExplicitComponent):
    """
    Computation of the stator yoke thickness of the SM PMSM. The formula is obtained from
    equation (II-45) in :cite:`touhami:2020`.
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
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            val=np.nan,
            units="m",
            desc="Stator bore diameter of the SM PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_yoke_flux_density",
            val=np.nan,
            units="T",
            desc="The design magnetic flux density in the stator yoke layer",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":design_surface_current_density",
            val=np.nan,
            units="kA/m",
            desc="The design surface current density of the motor",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":design_air_gap_flux_density",
            val=np.nan,
            units="T",
            desc="The design air gap magnetic flux density",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            units="m",
            desc="Stator yoke thickness of the SM PMSM",
            val=0.02,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        bore_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"] / 2.0
        )
        air_gap_flux_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_air_gap_flux_density"
        ]
        yoke_flux_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_yoke_flux_density"
        ]
        surface_current_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_surface_current_density"
        ]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]
        x2p_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio"]
        mu_0 = 4.0 * np.pi * 1e-7  # Magnetic permeability [H/m]
        total_flux_density = np.sqrt(
            (mu_0 * surface_current_density * x2p_ratio) ** 2.0 + air_gap_flux_density**2.0
        )

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height"] = (
            bore_radius * total_flux_density / (num_pole_pairs * np.abs(yoke_flux_density))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        bore_radius = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"] / 2.0
        )
        air_gap_flux_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_air_gap_flux_density"
        ]
        yoke_flux_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_yoke_flux_density"
        ]
        surface_current_density = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_surface_current_density"
        ]
        x2p_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio"]
        num_pole_pairs = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number"
        ]
        mu_0 = 4.0 * np.pi * 1e-7  # Magnetic permeability [H/m]
        total_flux_density = np.sqrt(
            (mu_0 * surface_current_density * x2p_ratio) ** 2.0 + air_gap_flux_density**2.0
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
        ] = (1.0 / (2.0 * num_pole_pairs)) * total_flux_density / np.abs(yoke_flux_density)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":pole_pairs_number",
        ] = -(bore_radius / num_pole_pairs**2.0) * total_flux_density / np.abs(yoke_flux_density)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio",
        ] = (
            (bore_radius / num_pole_pairs)
            * ((mu_0 * surface_current_density) ** 2.0 * x2p_ratio)
            / (np.abs(yoke_flux_density) * total_flux_density)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_air_gap_flux_density",
        ] = (
            (bore_radius / num_pole_pairs)
            * air_gap_flux_density
            / (np.abs(yoke_flux_density) * total_flux_density)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":design_yoke_flux_density",
        ] = (
            -(bore_radius / num_pole_pairs)
            * total_flux_density
            / (yoke_flux_density * np.abs(yoke_flux_density))
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:SM_PMSM:"
            + motor_id
            + ":design_surface_current_density",
        ] = (
            (bore_radius / num_pole_pairs)
            * ((mu_0 * x2p_ratio) ** 2.0 * surface_current_density)
            / (np.abs(yoke_flux_density) * total_flux_density)
        )
