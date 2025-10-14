# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingRotorDiameter(om.ExplicitComponent):
    """
    Computation of the rotor diameterof a cylindrical PMSM. The formulas are obtained from
    equation (II-85) in :cite:`touhami:2020`.
    """

    def initialize(self):
        self.options.declare(
            name="motor_id", default=None, desc="Identifier of the motor", allow_none=False
        )

    def setup(self):
        motor_id = self.options["motor_id"]

        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness",
            units="m",
            desc="The distance between the rotor and the stator bore",
            val=0.00075,
        )

        self.add_output(
            name="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter",
            units="m",
            val=0.11,
        )

    def setup_partials(self):
        motor_id = self.options["motor_id"]

        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter",
            val=1.0,
        )
        self.declare_partials(
            of="*",
            wrt="data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness",
            val=-2.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        motor_id = self.options["motor_id"]

        outputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":rotor_diameter"] = (
            inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":bore_diameter"]
            - 2.0
            * inputs["data:propulsion:he_power_train:SM_PMSM:" + motor_id + ":air_gap_thickness"]
        )
