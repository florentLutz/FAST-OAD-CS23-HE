# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingStatorYokeHeight(om.ExplicitComponent):
    """ Computation of the slot height of the PMSM."""

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
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ratiox2p",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":yoke_flux_density",
            val=np.nan,
            units="T",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":surface_current_density",
            val=np.nan,
            units="A/m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":airgap_flux_density",
            val=np.nan,
            units="T",
        )


        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ratiox2p",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":yoke_flux_density",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":surface_current_density",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":airgap_flux_density",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        R = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter"] / 2
        B_m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":airgap_flux_density"]
        B_sy = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":yoke_flux_density"]
        K_m = inputs[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":surface_current_density"
        ]
        p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]

        x2p_ratio = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ratiox2p"]
        mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability [H/m]

        # Equation II-46: Slot height hs

        hy = (R / p) * np.sqrt(
            (B_m / B_sy) ** 2 + ((mu_0 * K_m / B_sy) ** 2) * x2p_ratio)

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height"] = hy

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        pmsm_id = self.options["pmsm_id"]
        R = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter"] / 2
        B_m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":airgap_flux_density"]
        B_sy = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":yoke_flux_density"]
        K_m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":surface_current_density"]
        x2p_ratio = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ratiox2p"]
        p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
        mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability [H/m]
        #
        # hy = (R / p) * np.sqrt(
        #     (B_m / B_sy) ** 2 + ((mu_0 * K_m / B_sy) ** 2) * x2p_ratio)

        dhy_dD = (1 / (2 * p)) * np.sqrt(
            (B_m / B_sy) ** 2 + ((mu_0 * K_m / B_sy) ** 2) * x2p_ratio)

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
        ] = dhy_dD

        dhy_dp = - (R / (p**2)) * np.sqrt(
            (B_m / B_sy) ** 2 + ((mu_0 * K_m / B_sy) ** 2) * x2p_ratio)

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
        ] = dhy_dp

        dhy_dx2pratio =  (R / (p)) * ((mu_0 * K_m / B_sy) ** 2) /(2 * np.sqrt(
            (B_m / B_sy) ** 2 + ((mu_0 * K_m / B_sy) ** 2) * x2p_ratio))

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ratiox2p",
        ] = dhy_dx2pratio

        dhy_dBm = (R / (p)) * (B_m /( B_sy ** 2)) / ( np.sqrt(
            (B_m / B_sy) ** 2 + ((mu_0 * K_m / B_sy) ** 2) * x2p_ratio))

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":airgap_flux_density",
        ] = dhy_dBm

        dhy_dBsy = - (R / (p)) * np.sqrt((B_m** 2+ (mu_0 * K_m) ** 2)* x2p_ratio) / ((B_sy) ** 2)

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":yoke_flux_density",
        ] = dhy_dBsy

        dhy_dKm =  (R / (p)) * (((mu_0/B_sy)**2)* x2p_ratio*K_m)/ np.sqrt((B_m ** 2 + (mu_0 * K_m) ** 2)) / ((B_sy) ** 2)

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":surface_current_density",
        ] = dhy_dKm
