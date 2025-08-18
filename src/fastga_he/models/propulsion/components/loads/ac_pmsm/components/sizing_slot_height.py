# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingSlotHeight(om.ExplicitComponent):
    """Computation of the slot height of the PMSM."""

    def initialize(self):
        # Reference motor : EMRAX 268

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
            name="data:propulsion:he_power_train:AC_PMSM:"
            + pmsm_id
            + ":density_current_ac_max",  # doubt I don't know it
            val=np.nan,
            units="A/m**2",
        )

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":winding_factor",
            val=np.nan,
        )

        # self.add_input(
        #     name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
        #     val=np.nan,
        # )

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_conductor_factor",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_flux_density",
            val=np.nan,
            units="T",
        )

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density",
            val=np.nan,
            units="A/m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Tangential_stress",
            val=np.nan,
            units="N/m**2",
        )

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
            val=np.nan,
            units="T",
        )

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p", val=np.nan
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":density_current_ac_max",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":winding_factor",
            method="fd",
        )

        # self.declare_partials(
        #     of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
        #     wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
        #     method="fd",
        # )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_conductor_factor",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_flux_density",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Tangential_stress",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        sigma = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Tangential_stress"]
        k_w = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":winding_factor"]
        B_m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density"]
        j_rms = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":density_current_ac_max"
        ]
        k_sc = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_conductor_factor"
        ]
        k_fill = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor"]
        B_st = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_flux_density"]
        K_m = inputs[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density"
        ]
        x2p_ratio = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p"]
        # p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
        mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability [H/m]

        # Equation II-46: Slot height hs
        numerator = np.sqrt(2) * sigma
        denominator = k_w * B_m * j_rms
        second_term = 1 / (
            k_sc
            * k_fill
            * (
                1
                - (
                    (2 / np.pi)
                    * np.sqrt((B_m / B_st) ** 2 + ((mu_0 * K_m / B_st) ** 2) * x2p_ratio)
                )
            )
        )
        hs = (numerator / denominator) * second_term

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height"] = hs

    """def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        sigma = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Tangential_stress"]
        k_w = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":winding_factor"]
        B_m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":airgap_flux_density"]
        j_rms = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":density_current_ac_max"]
        k_sc = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_conductor_factor"]
        k_fill = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_fill_factor"]
        B_st = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_flux_density"]
        K_m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":surface_current_density"]
        x = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio"]
        p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
        x_2p = x ** (2 * p)
        mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability [H/m]

        # Equation II-46: Slot height hs
        numerator = np.sqrt(2)                  #* sigma
        denominator = k_w * B_m * j_rms
        second_term = 1 / (k_sc * k_fill * (1 - ((2 / np.pi) * np.sqrt(
            (B_m / B_st) ** 2 + ((mu_0 * K_m / B_st) ** 2) * (((1 + x_2p) / (1 - x_2p)) ** 2)))))
        dhs_dsigma = (numerator / denominator) * second_term

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Tangential_stress",
        ] = dhs_dsigma
"""
