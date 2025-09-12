# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingExtStatorDiameter(om.ExplicitComponent):
    """
    Computation of the external stator diameter of a cylindrical PMSM. The formula is obtained from
    equation (II-47) in :cite:`touhami:2020.
    """

    def initialize(self):
        self.options.declare(
            name="pmsm_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        pmsm_id = self.options["pmsm_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":bore_diameter",
            val=np.nan,
            units="m",
            desc="Stator bore diameter of the PMSM",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_height",
            val=np.nan,
            units="m",
            desc="Single stator slot height (radial)",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_yoke_height",
            val=np.nan,
            units="m",
            desc="Stator yoke thickness of the PMSM",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_diameter",
            units="m",
            desc="The outer stator diameter of the PMSM",
            val=0.2,
        )

    def setup_partials(self):
        pmsm_id = self.options["pmsm_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_diameter",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_height",
                "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_yoke_height",
            ],
            val=2.0,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_diameter",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":bore_diameter",
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pmsm_id = self.options["pmsm_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_diameter"] = (
            2.0
            * (
                inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":slot_height"]
                + inputs[
                    "data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":stator_yoke_height"
                ]
            )
            + inputs["data:propulsion:he_power_train:SM_PMSM:" + pmsm_id + ":bore_diameter"]
        )
