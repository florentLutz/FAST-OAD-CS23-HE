# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingMotorWeight(om.ExplicitComponent):
    """
    Computation of the PMSM total weight with summing all the component mass.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_core_weight",
            val=np.nan,
            units="kg",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_winding_weight",
            val=np.nan,
            units="kg",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":rotor_weight",
            val=np.nan,
            units="kg",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_weight",
            val=np.nan,
            units="kg",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":mass",
            units="kg",
            val=55.0,
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":mass",
            wrt="*",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":mass"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_core_weight"]
            + inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_winding_weight"]
            + inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":rotor_weight"]
            + inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":frame_weight"]
        )
