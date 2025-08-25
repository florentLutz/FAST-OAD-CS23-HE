# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingSlotWidth(om.ExplicitComponent):
    """ Computation of the slot width of the PMSM."""

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
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductors_number",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":tooth_flux_density",
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

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ratiox2p", val=np.nan
        )

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductors_number",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":tooth_flux_density",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":surface_current_density",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":airgap_flux_density",
            method="fd",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ratiox2p",
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        R = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter"] / 2
        B_m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":airgap_flux_density"]
        B_st = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":tooth_flux_density"]
        K_m = inputs[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":surface_current_density"
        ]
        x2p_ratio = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ratiox2p"]
        p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
        Nc = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":conductors_number"]
        Ns = Nc  # Ncs = 1
        mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability [H/m]

        # Equation II-46: Slot height hs

        r_tooth = (2 / np.pi) * np.sqrt(
            (B_m / B_st) ** 2 + ((mu_0 * K_m / B_st) ** 2) * x2p_ratio
        )
        ls = (1 - r_tooth) * 2 * np.pi * R / Ns

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":slot_width"] = ls

    """def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        pmsm_id = self.options["pmsm_id"]
        R = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter"] / 2
        B_m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":airgap_flux_density"]
        B_sy = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":yoke_flux_density"]
        K_m = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":surface_current_density"]
        x = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":radius_ratio"]
        p = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":pole_pairs_number"]
        x_2p = x ** (2 * p)
        mu_0 = 4 * np.pi * 1e-7  # Magnetic permeability [H/m]

        # Equation II-46: Slot height hs

        dhy_dD = (1 / (2 * p)) * np.sqrt(
            (B_m / B_sy) ** 2 + ((mu_0 * K_m / B_sy) ** 2) * (((1 + x_2p) / (1 - x_2p)) ** 2))

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_yoke_height",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":diameter",
        ] = dhy_dD"""
