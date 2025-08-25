# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingFrameWeight(om.ExplicitComponent):
    """ Computation of the frame weight of the PMSM."""

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
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff",
            val=np.nan,
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_density",
            val=np.nan,
            units="kg/m**3",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_diameter",
            units="m",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight",
            units="kg",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_diameter",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_density",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight",
            wrt="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        R_out = (
            inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter"] / 2
        )
        Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
        k_tb = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff"]
        rho_fr = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_density"]

        R_out_mm = R_out * 1000

        tau_r_ = 0.7371 * R_out**2 - 0.580 * R_out + 1.1599 if R_out_mm <= 400 else 1.04
        R_fr = tau_r_ * R_out

        # Frame weight
        W_frame = rho_fr * (
            np.pi * Lm * k_tb * (R_fr**2 - R_out**2) + 2 * np.pi * (tau_r_ - 1) * R_out * R_fr**2
        )

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_diameter"] = 2 * R_fr
        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight"] = W_frame

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]
        R_out = (
            inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter"] / 2
        )
        Lm = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length"]
        k_tb = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff"]
        rho_fr = inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_density"]

        R_out_mm = R_out * 1000

        tau_r_ = 0.7371 * R_out**2 - 0.580 * R_out + 1.1599 if R_out_mm <= 400 else 1.04
        dt_dD = 0.7371 * R_out - 0.580 / 2 if R_out_mm <= 400 else 0
        R_fr = tau_r_ * R_out
        dRfr_dD = tau_r_ / 2 + dt_dD * R_out

        # Frame weight
        W_frame = rho_fr * (
            np.pi * Lm * k_tb * (R_fr**2 - R_out**2) + 2 * np.pi * (tau_r_ - 1) * R_out * R_fr**2
        )
        dW_dD = rho_fr * (
            np.pi * Lm * k_tb * (2 * R_fr * dRfr_dD - R_out)
            + 2 * np.pi * dt_dD * R_out * R_fr**2
            + 2 * np.pi * (tau_r_ - 1) * 0.5 * R_fr**2
            + 2 * np.pi * (tau_r_ - 1) * R_out * 2 * R_fr * dRfr_dD
        )
        dW_drho = (
            np.pi * Lm * k_tb * (R_fr**2 - R_out**2) + 2 * np.pi * (tau_r_ - 1) * R_out * R_fr**2
        )
        dW_dLm = rho_fr * (
            np.pi * k_tb * (R_fr**2 - R_out**2) + 2 * np.pi * (tau_r_ - 1) * R_out * R_fr**2
        )
        dW_dktb = rho_fr * (
            np.pi * Lm * (R_fr**2 - R_out**2) + 2 * np.pi * (tau_r_ - 1) * R_out * R_fr**2
        )

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_diameter",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
        ] = 2 * dRfr_dD

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":ext_stator_diameter",
        ] = dW_dD

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":active_length",
        ] = dW_dLm

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":end_winding_coeff",
        ] = dW_dktb

        partials[
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_density",
        ] = dW_drho
