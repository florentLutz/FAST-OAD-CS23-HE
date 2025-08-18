# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingRatioX2p(om.ExplicitComponent):
    """Computation of the slot height of the PMSM."""

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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p",
            units="m",
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p",
            wrt="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        # Equation II-46: Slot height hs

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p"] = (
            (1.0 + inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p"])
            / (1.0 - inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p"])
        ) ** 2.0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":ratiox2p",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p",
        ] = (
            4.0
            * (1.0 + inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p"])
            / ((1.0 - inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":x2p"]) ** 3.0)
        )
