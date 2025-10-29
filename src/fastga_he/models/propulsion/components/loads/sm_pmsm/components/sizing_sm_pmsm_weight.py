# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class SizingMotorWeight(om.ExplicitComponent):
    """
    Computation of the SM PMSM total weight by summing all the component mass.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass",
            val=np.nan,
            units="kg",
            desc="The weight of the stator excluding the wire weight",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_winding_mass",
            val=np.nan,
            units="kg",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass",
            val=np.nan,
            units="kg",
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass",
            val=np.nan,
            units="kg",
            desc="The weight of the motor casing",
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass",
            units="kg",
            val=55.0,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":mass"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_core_mass"]
            + inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":stator_winding_mass"]
            + inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_mass"]
            + inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":frame_mass"]
        )
