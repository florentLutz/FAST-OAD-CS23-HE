# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingFrameGeometry(om.ExplicitComponent):
    """
    Computation of the frame diameter and weight of the PMSM. The formula is obtained from
    equation (II-53) and (II-59) respectively in :cite:`touhami:2020.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
            desc="The stator length of PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":end_winding_coeff",
            val=np.nan,
            desc="The factor to account extra length from end winding",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_diameter",
            val=np.nan,
            units="m",
            desc="The outer stator diameter of the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_density",
            val=np.nan,
            units="kg/m**3",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_diameter",
            units="m",
            val=0.23,
        )
        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_weight",
            units="kg",
            val=6.0,
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_diameter",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_weight",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_density",
                "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":active_length",
                "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":end_winding_coeff",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        r_out = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_diameter"] / 2.0
        )
        l_m = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":active_length"]
        k_tb = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":end_winding_coeff"]
        rho_fr = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_density"]

        r_out_mm = r_out * 1000.0

        tau_r_ = 0.7371 * r_out**2.0 - 0.580 * r_out + 1.1599 if r_out_mm <= 400.0 else 1.04
        r_fr = tau_r_ * r_out

        # Frame weight
        w_frame = rho_fr * (
            np.pi * l_m * k_tb * (r_fr**2.0 - r_out**2.0)
            + 2.0 * np.pi * (tau_r_ - 1.0) * r_out * r_fr**2.0
        )

        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_diameter"] = (
            2.0 * r_fr
        )
        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_weight"] = w_frame

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        r_out = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_diameter"] / 2.0
        )
        l_m = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":active_length"]
        k_tb = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":end_winding_coeff"]
        rho_fr = inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_density"]

        r_out_mm = r_out * 1000.0
        tau_r_ = 0.7371 * r_out**2 - 0.580 * r_out + 1.1599 if r_out_mm <= 400 else 1.04
        dt_dd = 0.7371 * r_out - 0.29 if r_out_mm <= 400 else 0.0
        r_fr = tau_r_ * r_out
        drfr_dd = tau_r_ / 2.0 + dt_dd * r_out

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_diameter",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_diameter",
        ] = 2.0 * drfr_dd

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_diameter",
        ] = rho_fr * (
            np.pi * l_m * k_tb * (2.0 * r_fr * drfr_dd - r_out)
            + 2.0 * np.pi * dt_dd * r_out * r_fr**2.0
            + 2.0 * np.pi * (tau_r_ - 1.0) * 0.5 * r_fr**2.0
            + 2.0 * np.pi * (tau_r_ - 1.0) * r_out * 2.0 * r_fr * drfr_dd
        )

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":active_length",
        ] = rho_fr * np.pi * k_tb * (r_fr**2.0 - r_out**2.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":end_winding_coeff",
        ] = rho_fr * np.pi * l_m * (r_fr**2.0 - r_out**2.0)

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_weight",
            "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_density",
        ] = (
            np.pi * l_m * k_tb * (r_fr**2.0 - r_out**2.0)
            + 2.0 * np.pi * (tau_r_ - 1.0) * r_out * r_fr**2.0
        )
