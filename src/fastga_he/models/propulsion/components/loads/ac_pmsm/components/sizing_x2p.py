# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class Sizingx2p(om.ExplicitComponent):
    """Computation of the contribute x^2p used for sizing the slot width and yoke height ."""

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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio", val=np.nan
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p",
            units="m",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]
        x = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio"]
        p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
        x_2p = x ** (2.0 * p)

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p"] = x_2p

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        x = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio"]
        p = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number"]
        x_2p = x ** (2.0 * p)

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":radius_ratio",
        ] = (2.0 * p) * (x ** (2.0 * p - 1.0))

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":pole_pairs_number",
        ] = 2.0 * x_2p * np.log(x)
