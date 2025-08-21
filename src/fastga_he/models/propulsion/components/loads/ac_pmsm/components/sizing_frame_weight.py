# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingFrameWeight(om.ExplicitComponent):
    """Computation of the frame weight of the PMSM."""

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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_density",
            val=np.nan,
            units="kg/m**3",
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_diameter",
            units="m",
        )
        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_weight",
            units="kg",
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_diameter",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_density",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_weight",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        r_out = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter"]
            / 2.0
        )
        l_m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"]
        k_tb = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff"]
        rho_fr = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_density"]

        r_out_mm = r_out * 1000.0

        tau_r_ = 0.7371 * r_out**2.0 - 0.580 * r_out + 1.1599 if r_out_mm <= 400.0 else 1.04
        r_fr = tau_r_ * r_out

        # Frame weight
        w_frame = rho_fr * (
            np.pi * l_m * k_tb * (r_fr**2.0 - r_out**2.0)
            + 2.0 * np.pi * (tau_r_ - 1.0) * r_out * r_fr**2.0
        )

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_diameter"] = (
            2.0 * r_fr
        )
        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_weight"] = w_frame

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        r_out = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter"]
            / 2.0
        )
        l_m = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"]
        k_tb = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff"]
        rho_fr = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_density"]

        r_out_mm = r_out * 1000.0
        tau_r_ = 0.7371 * r_out**2 - 0.580 * r_out + 1.1599 if r_out_mm <= 400 else 1.04
        dt_dd = 0.7371 * r_out - 0.29 if r_out_mm <= 400 else 0.0
        r_fr = tau_r_ * r_out
        drfr_dd = tau_r_ / 2.0 + dt_dd * r_out

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_diameter",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter",
        ] = 2.0 * drfr_dd

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ext_stator_diameter",
        ] = rho_fr * (
            np.pi * l_m * k_tb * (2.0 * r_fr * drfr_dd - r_out)
            + 2.0 * np.pi * dt_dd * r_out * r_fr**2.0
            + 2.0 * np.pi * (tau_r_ - 1.0) * 0.5 * r_fr**2.0
            + 2.0 * np.pi * (tau_r_ - 1.0) * r_out * 2.0 * r_fr * drfr_dd
        )

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
        ] = rho_fr * np.pi * k_tb * (r_fr**2.0 - r_out**2.0)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff",
        ] = rho_fr * np.pi * l_m * (r_fr**2.0 - r_out**2.0)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":frame_density",
        ] = (
            np.pi * l_m * k_tb * (r_fr**2.0 - r_out**2.0)
            + 2.0 * np.pi * (tau_r_ - 1.0) * r_out * r_fr**2.0
        )
