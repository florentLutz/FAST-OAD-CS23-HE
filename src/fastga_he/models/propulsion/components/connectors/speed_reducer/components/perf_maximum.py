# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum torques of the reducer
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="speed_reducer_id",
            default=None,
            desc="Identifier of the speed reducer",
            allow_none=False,
        )

    def setup(self):
        speed_reducer_id = self.options["speed_reducer_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("torque_out", units="N*m", val=np.nan, shape=number_of_points)
        self.add_input("torque_in", units="N*m", val=np.nan, shape=number_of_points)

        self.add_output(
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":torque_in_max",
            units="N*m",
            val=200.0,
            desc="Maximum value of the input torque of the gearbox",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_in_max",
            wrt="torque_in",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.add_output(
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":torque_out_max",
            units="N*m",
            val=200.0,
            desc="Maximum value of the output torque of the gearbox",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:speed_reducer:"
            + speed_reducer_id
            + ":torque_out_max",
            wrt="torque_out",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        speed_reducer_id = self.options["speed_reducer_id"]

        outputs[
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":torque_in_max"
        ] = np.max(inputs["torque_in"])
        outputs[
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":torque_out_max"
        ] = np.max(inputs["torque_out"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        speed_reducer_id = self.options["speed_reducer_id"]

        partials[
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":torque_in_max",
            "torque_in",
        ] = np.where(inputs["torque_in"] == np.max(inputs["torque_in"]), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":torque_out_max",
            "torque_out",
        ] = np.where(inputs["torque_out"] == np.max(inputs["torque_out"]), 1.0, 0.0)
