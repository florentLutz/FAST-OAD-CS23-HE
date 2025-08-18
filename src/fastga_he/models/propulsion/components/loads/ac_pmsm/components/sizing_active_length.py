# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingActiveLength(om.ExplicitComponent):
    """Computation of the Active Length of a cylindrical PMSM."""

    def initialize(self):
        #  Reference motor : HASTECS project, Sarah Touhami

        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
        )

        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Form_coefficient",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            method="exact",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Form_coefficient",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        # Equation II-43: Stator inner radius R

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length"] = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"]
            / inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Form_coefficient"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
        ] = 1 / inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Form_coefficient"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":active_length",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Form_coefficient",
        ] = (
            -inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"]
            / inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":Form_coefficient"] ** 2
        )
