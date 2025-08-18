# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingSlotSection(om.ExplicitComponent):
    """
    Computation of the Slot section.

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
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
            val=np.nan,
            units="m",
        )
        self.add_input(
            name="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
            val=np.nan,
            units="m",
        )

        self.add_output(
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_section",
            units="m**2",
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_section",
            wrt=[
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
                "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_section"] = (
            inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height"]
            * inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pmsm_id = self.options["pmsm_id"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_section",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height",
        ] = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width"]

        partials[
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_section",
            "data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_width",
        ] = inputs["data:propulsion:he_power_train:AC_PMSM:" + pmsm_id + ":slot_height"]
