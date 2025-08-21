# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingConductorLength(om.ExplicitComponent):
    """
    Computation of the Conductor length.

    """

    def initialize(self):
        # Reference motor : HASTECS project, Sarah Touhami

        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":cond_twisting_coeff",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff",
            val=np.nan,
        )

        self.add_output(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length", units="m"
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length",
            wrt=[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":cond_twisting_coeff",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        lm = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"]
        k_lc = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":cond_twisting_coeff"]
        k_tb = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff"]

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length"] = (
            lm * k_lc * k_tb
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]
        lm = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"]
        k_lc = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":cond_twisting_coeff"]
        k_tb = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
        ] = k_lc * k_tb

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":cond_twisting_coeff",
        ] = lm * k_tb

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductor_length",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":end_winding_coeff",
        ] = lm * k_lc
