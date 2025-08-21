# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingSlotWidth(om.ExplicitComponent):
    """Computation of the slot width of the PMSM."""

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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
            units="m",
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        d = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"]
        tooth_ratio = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio"]
        ns = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number"]

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width"] = (
            (1.0 - tooth_ratio) * np.pi * d / ns
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        d = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter"] / 2.0
        tooth_ratio = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio"]
        ns = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":diameter",
        ] = (1.0 - tooth_ratio) * np.pi / ns

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":tooth_ratio",
        ] = -np.pi * d / ns

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":conductors_number",
        ] = -(1.0 - tooth_ratio) * np.pi * d / ns**2.0
