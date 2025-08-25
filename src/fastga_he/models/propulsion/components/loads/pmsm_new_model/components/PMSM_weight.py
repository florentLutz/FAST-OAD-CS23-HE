# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingMotorWeight(om.ExplicitComponent):
    """Computation of the PMSM weight."""

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
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight",
            val=np.nan,
            units="kg",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight",
            val=np.nan,
            units="kg",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight",
            val=np.nan,
            units="kg",
        )

        self.add_input(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight",
            val=np.nan,
            units="kg",
        )

        self.add_output(
            name="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mass",
            units="kg",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mass",
            wrt="*",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":mass"] = (
            inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_core_weight"]
            + inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":stator_winding_weight"]
            + inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":rotor_weight"]
            + inputs["data:propulsion:he_power_train:ACPMSM:" + pmsm_id + ":frame_weight"]
        )  # /2
