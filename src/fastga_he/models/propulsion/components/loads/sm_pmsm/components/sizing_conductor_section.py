# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingConductorSection(om.ExplicitComponent):
    """
    Computation of the conductor material area coverage in one stator slot of the PMSM. The
    formula  is obtained from part II.2.3a in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section",
            val=np.nan,
            units="m**2",
            desc="Single slot cross section area on the motor stator",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor",
            val=np.nan,
            desc="The factor describes the conductor material fullness inside the stator slots",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor",
            val=np.nan,
            desc="The area factor considers the cross-section shape twist due to wire bunching",
        )

        self.add_output(
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_section",
            units="m**2",
            val=0.0001,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_section",
            wrt=[
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor",
                "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        s_slot = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section"]
        k_fill = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor"]
        k_sc = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor"
        ]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_section"] = (
            s_slot * k_sc * k_fill
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        motor_id = self.options["motor_id"]

        s_slot = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section"]
        k_fill = inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor"]
        k_sc = inputs[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor"
        ]

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_section",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_section",
        ] = k_fill * k_sc

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_section",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_fill_factor",
        ] = s_slot * k_sc

        partials[
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":conductor_section",
            "data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":slot_conductor_factor",
        ] = s_slot * k_fill
