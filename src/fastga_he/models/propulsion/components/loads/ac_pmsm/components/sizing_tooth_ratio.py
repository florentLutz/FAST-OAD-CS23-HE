# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingToothRatio(om.ExplicitComponent):
    """Computation of the ratio between tooth length to bore radius."""

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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_flux_density",
            val=np.nan,
            units="T",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
            val=np.nan,
            units="T",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density",
            val=np.nan,
            units="A/m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p",
            val=np.nan,
            units="m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio", val=0.04
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        b_m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density"]
        b_st = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_flux_density"]
        k_m = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density"
        ]
        x2p_ratio = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p"]
        mu_0 = 4.0 * np.pi * 1.0e-7  # Magnetic permeability [H/m]

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio"] = (
            2.0 / np.pi
        ) * np.sqrt((b_m / b_st) ** 2.0 + ((mu_0 * k_m * x2p_ratio / b_st) ** 2.0))

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        b_m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density"]
        b_st = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_flux_density"]
        k_m = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density"
        ]
        x2p_ratio = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p"]
        mu_0 = 4.0 * np.pi * 1e-7  # Magnetic permeability [H/m]
        common_term = np.sqrt((mu_0 * k_m * x2p_ratio) ** 2.0 + b_m**2.0)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p",
        ] = 2.0 * (mu_0 * k_m) ** 2.0 * x2p_ratio / (np.pi * np.abs(b_st) * common_term)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
        ] = 2.0 * b_m / (np.pi * np.abs(b_st) * common_term)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_flux_density",
        ] = -2.0 * common_term / (np.pi * b_st * np.abs(b_st))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density",
        ] = 2.0 * (mu_0 * x2p_ratio) ** 2.0 * k_m / (np.pi * np.abs(b_st) * common_term)
