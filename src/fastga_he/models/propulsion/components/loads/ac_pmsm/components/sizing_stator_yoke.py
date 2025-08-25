# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorYokeHeight(om.ExplicitComponent):
    """
    Computation of the stator yoke thickness of the PMSM. The formula is obtained from
    equation (II-45) in :cite:`touhami:2020.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
            val=np.nan,
            desc="Number of the north and south pairs in the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Stator bore diameter of the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":yoke_flux_density",
            val=np.nan,
            units="T",
            desc="Magnetic flux density at the stator yoke layer",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density",
            val=np.nan,
            units="A/m",
            desc="The surface current density of the winding conductor cable",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
            val=np.nan,
            units="T",
            desc="The magnetic flux density provided by the permanent magnets",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            units="m",
            desc="Stator yoke thickness of the PMSM",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        r = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"] / 2.0
        b_m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density"]
        b_sy = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":yoke_flux_density"]
        k_m = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density"
        ]
        p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
        x2p_ratio = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p"]
        mu_0 = 4.0 * np.pi * 1e-7  # Magnetic permeability [H/m]
        max_total_airgap_flux_density = np.sqrt((mu_0 * k_m * x2p_ratio) ** 2.0 + b_m**2.0)

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height"] = (
            r * max_total_airgap_flux_density / (p * np.abs(b_sy))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        r = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"] / 2.0
        b_m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density"]
        b_sy = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":yoke_flux_density"]
        k_m = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density"
        ]
        x2p_ratio = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p"]
        p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
        mu_0 = 4.0 * np.pi * 1e-7  # Magnetic permeability [H/m]
        max_total_airgap_flux_density = np.sqrt((mu_0 * k_m * x2p_ratio) ** 2.0 + b_m**2.0)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
        ] = (1.0 / (2.0 * p)) * max_total_airgap_flux_density / np.abs(b_sy)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
        ] = -(r / p**2.0) * max_total_airgap_flux_density / np.abs(b_sy)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p",
        ] = (
            (r / p)
            * ((mu_0 * k_m) ** 2.0 * x2p_ratio)
            / (np.abs(b_sy) * max_total_airgap_flux_density)
        )

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
        ] = (r / p) * b_m / (np.abs(b_sy) * max_total_airgap_flux_density)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":yoke_flux_density",
        ] = -(r / p) * max_total_airgap_flux_density / (b_sy * np.abs(b_sy))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density",
        ] = (
            (r / p)
            * ((mu_0 * x2p_ratio) ** 2.0 * k_m)
            / (np.abs(b_sy) * max_total_airgap_flux_density)
        )
