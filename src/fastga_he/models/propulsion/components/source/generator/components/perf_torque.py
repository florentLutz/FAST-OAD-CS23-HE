# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesTorque(om.ExplicitComponent):
    """
    Computation of the torque at the input of the generator based on the RMS current and torque
    constant.
    """

    def initialize(self):

        self.options.declare(
            name="generator_id", default=None, desc="Identifier of the motor", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        generator_id = self.options["generator_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "ac_current_rms_out",
            val=np.full(number_of_points, np.nan),
            units="A",
            desc="Current at the output side of the generator",
        )
        self.add_input(
            name="data:propulsion:he_power_train:generator:" + generator_id + ":torque_constant",
            val=1.0,
            units="N*m/A",
        )

        self.add_output("torque_in", units="N*m", val=400.0, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        generator_id = self.options["generator_id"]

        current_rms_out = inputs["ac_current_rms_out"]
        k_t = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":torque_constant"
        ]

        outputs["torque_in"] = current_rms_out * k_t

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        generator_id = self.options["generator_id"]
        number_of_points = self.options["number_of_points"]

        current_rms_out = inputs["ac_current_rms_out"]
        k_t = inputs[
            "data:propulsion:he_power_train:generator:" + generator_id + ":torque_constant"
        ]

        partials["torque_in", "ac_current_rms_out"] = k_t * np.eye(number_of_points)
        partials[
            "torque_in",
            "data:propulsion:he_power_train:generator:" + generator_id + ":torque_constant",
        ] = current_rms_out
