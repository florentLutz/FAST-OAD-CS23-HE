# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingToothRatio(om.ExplicitComponent):
    """
    Computation of the ratio between tooth length to bore radius. The formula is obtained from
    equation (II-48) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":tooth_flux_density",
            val=np.nan,
            units="T",
            desc="Magnetic flux density at the stator teeth(slot) layer",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_flux_density",
            val=np.nan,
            units="T",
            desc="The magnetic flux density provided by the permanent magnets",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":surface_current_density",
            val=np.nan,
            units="A/m",
            desc="The surface current density of the winding conductor cable",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":ratiox2p",
            val=np.nan,
            units="m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":tooth_ratio",
            val=0.04,
            desc="The fraction between overall tooth length and stator bore circumference",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        b_m = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_flux_density"]
        b_st = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":tooth_flux_density"]
        k_m = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":surface_current_density"
        ]
        x2p_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":ratiox2p"]
        mu_0 = 4.0 * np.pi * 1.0e-7  # Magnetic permeability [H/m]
        max_total_airgap_flux_density = np.sqrt((mu_0 * k_m * x2p_ratio) ** 2.0 + b_m**2.0)

        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":tooth_ratio"] = (
            2.0 * max_total_airgap_flux_density / (np.pi * np.abs(b_st))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        b_m = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_flux_density"]
        b_st = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":tooth_flux_density"]
        k_m = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":surface_current_density"
        ]
        x2p_ratio = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":ratiox2p"]
        mu_0 = 4.0 * np.pi * 1e-7  # Magnetic permeability [H/m]
        max_total_airgap_flux_density = np.sqrt((mu_0 * k_m * x2p_ratio) ** 2.0 + b_m**2.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":tooth_ratio",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":ratiox2p",
        ] = (
            2.0
            * (mu_0 * k_m) ** 2.0
            * x2p_ratio
            / (np.pi * np.abs(b_st) * max_total_airgap_flux_density)
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":tooth_ratio",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":airgap_flux_density",
        ] = 2.0 * b_m / (np.pi * np.abs(b_st) * max_total_airgap_flux_density)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":tooth_ratio",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":tooth_flux_density",
        ] = -2.0 * max_total_airgap_flux_density / (np.pi * b_st * np.abs(b_st))

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":tooth_ratio",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":surface_current_density",
        ] = (
            2.0
            * (mu_0 * x2p_ratio) ** 2.0
            * k_m
            / (np.pi * np.abs(b_st) * max_total_airgap_flux_density)
        )
