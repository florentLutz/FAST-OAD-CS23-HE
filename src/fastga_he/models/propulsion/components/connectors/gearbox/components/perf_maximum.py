# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum torques of the planetary gear
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="gearbox_id",
            default=None,
            desc="Identifier of the gearbox",
            allow_none=False,
        )

    def setup(self):

        gearbox_id = self.options["gearbox_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("torque_out_1", units="N*m", val=np.nan, shape=number_of_points)
        self.add_input("torque_out_2", units="N*m", val=np.nan, shape=number_of_points)

        self.add_output(
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_max",
            units="N*m",
            val=200.0,
            desc="Maximum value of the output torque of the gearbox",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_max",
            wrt=["torque_out_1", "torque_out_2"],
            method="exact",
            cols=np.arange(number_of_points),
            rows=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        gearbox_id = self.options["gearbox_id"]

        torque_out = inputs["torque_out_1"] + inputs["torque_out_2"]

        outputs[
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_max"
        ] = np.max(torque_out)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        gearbox_id = self.options["gearbox_id"]

        torque_out = inputs["torque_out_1"] + inputs["torque_out_2"]

        partials[
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_max",
            "torque_out_1",
        ] = np.where(torque_out == np.max(torque_out), 1.0, 0.0)
        partials[
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_max",
            "torque_out_2",
        ] = np.where(torque_out == np.max(torque_out), 1.0, 0.0)
