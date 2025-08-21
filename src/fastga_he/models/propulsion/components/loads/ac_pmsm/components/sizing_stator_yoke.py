# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorYokeHeight(om.ExplicitComponent):
    """Computation of the slot height of the PMSM."""

    def initialize(self):
        # Reference motor : HASTECS project, Sarah Touhami

        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        # self.options.declare(
        # "diameter_ref",
        # default=0.268,
        # desc="Diameter of the reference motor in [m]",
        # )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":yoke_flux_density",
            val=np.nan,
            units="T",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density",
            val=np.nan,
            units="A/m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
            val=np.nan,
            units="T",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            units="m",
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":yoke_flux_density",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
            method="exact",
        )

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

        # Equation II-46: Slot height hs

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height"] = (
            r / p
        ) * np.sqrt((b_m / b_sy) ** 2.0 + ((mu_0 * k_m * x2p_ratio / b_sy) ** 2.0))

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

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
        ] = (1.0 / (2.0 * p)) * np.sqrt(
            (b_m / b_sy) ** 2.0 + ((mu_0 * k_m * x2p_ratio / b_sy) ** 2.0)
        )

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
        ] = -(r / p**2.0) * np.sqrt((b_m / b_sy) ** 2.0 + ((mu_0 * k_m * x2p_ratio / b_sy) ** 2.0))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p",
        ] = (
            (r / p)
            * ((mu_0 * k_m) ** 2.0 * x2p_ratio)
            / (np.abs(b_sy) * np.sqrt(b_m**2.0 + (mu_0 * k_m * x2p_ratio) ** 2.0))
        )

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
        ] = (r / p) * b_m / (np.abs(b_sy) * np.sqrt(b_m**2.0 + (mu_0 * k_m * x2p_ratio) ** 2.0))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":yoke_flux_density",
        ] = -(r / p) * np.sqrt(b_m**2.0 + (mu_0 * k_m * x2p_ratio) ** 2.0) / (b_sy * np.abs(b_sy))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density",
        ] = (
            (r / p)
            * ((mu_0 * x2p_ratio) ** 2.0 * k_m)
            / (np.abs(b_sy) * np.sqrt(b_m**2.0 + (mu_0 * k_m * x2p_ratio) ** 2.0))
        )
