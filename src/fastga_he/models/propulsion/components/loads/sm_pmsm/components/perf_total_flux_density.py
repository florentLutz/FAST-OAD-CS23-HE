# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

MAGNETIC_PERMEABILITY = 4.0 * np.pi * 1.0e-7  # Magnetic permeability [H/m]


class PerformancesTotalFluxDensity(om.ExplicitComponent):
    """
    Computation of the Maximum total magnetic flux density in air gap of the motor,
    including the magnetic flux from electromagnetism. The formula is obtained
    from equation (II-21) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        motor_id = self.options["motor_id"]

        self.add_input(
            name="air_gap_flux_density",
            units="T",
            desc="The magnetic flux density provided by the permanent magnets",
            shape=number_of_points,
            val=np.nan,
        )
        self.add_input(
            name="surface_current_density",
            val=np.nan,
            units="A/m",
            shape=number_of_points,
            desc="The maximum surface current density of the winding conductor cable",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio",
            val=np.nan,
        )

        self.add_output(
            name="total_flux_density",
            val=0.89,
            units="T",
            shape=number_of_points,
            desc="The total magnetic flux density in air gap including electromagnetism",
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="*",
            wrt=["surface_current_density", "air_gap_flux_density"],
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
            method="exact",
        )

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        air_gap_flux_density = inputs["air_gap_flux_density"]
        surface_current_density = inputs["surface_current_density"]
        x2p_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio"]

        unclipped_flux_density = np.sqrt(
            air_gap_flux_density**2.0
            + (MAGNETIC_PERMEABILITY * surface_current_density * x2p_ratio) ** 2.0
        )

        outputs["total_flux_density"] = np.clip(unclipped_flux_density, 0.5, 2.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        air_gap_flux_density = inputs["air_gap_flux_density"]
        surface_current_density = inputs["surface_current_density"]
        x2p_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio"]
        total_flux_density = np.sqrt(
            air_gap_flux_density**2.0
            + (MAGNETIC_PERMEABILITY * surface_current_density * x2p_ratio) ** 2.0
        )
        clipped_flux_density = np.clip(total_flux_density, 0.5, 2.0)

        partials["total_flux_density", "air_gap_flux_density"] = np.where(
            total_flux_density == clipped_flux_density,
            air_gap_flux_density / total_flux_density,
            1e-6,
        )

        partials["total_flux_density", "surface_current_density"] = np.where(
            total_flux_density == clipped_flux_density,
            (MAGNETIC_PERMEABILITY * x2p_ratio) ** 2.0
            * surface_current_density
            / total_flux_density,
            1e-6,
        )

        partials[
            "total_flux_density",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":x2p_ratio",
        ] = np.where(
            total_flux_density == clipped_flux_density,
            (MAGNETIC_PERMEABILITY * surface_current_density) ** 2.0
            * x2p_ratio
            / total_flux_density,
            1e-6,
        )
