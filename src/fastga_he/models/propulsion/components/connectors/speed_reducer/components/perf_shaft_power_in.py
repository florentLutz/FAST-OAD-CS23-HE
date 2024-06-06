# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesShaftPowerIn(om.ExplicitComponent):
    """
    Component which computes the input shaft power based on the output shaft power and an assumed
    constant efficiency. Default value for the efficiency are taken from literature (see
    :cite:`thauvin:2018` and :cite:`pettes:2021`).
    """

    def initialize(self):

        self.options.declare(
            name="speed_reducer_id",
            default=None,
            desc="Identifier of the speed reducer",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        speed_reducer_id = self.options["speed_reducer_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":efficiency",
            val=0.98,
            desc="Efficiency of the speed reducer",
        )
        self.add_input("shaft_power_out", units="kW", val=np.nan, shape=number_of_points)

        self.add_output("shaft_power_in", units="kW", val=100.0, shape=number_of_points)

        self.declare_partials(
            of="shaft_power_in",
            wrt="shaft_power_out",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="shaft_power_in",
            wrt="data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":efficiency",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        speed_reducer_id = self.options["speed_reducer_id"]

        outputs["shaft_power_in"] = (
            inputs["shaft_power_out"]
            / inputs[
                "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":efficiency"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        speed_reducer_id = self.options["speed_reducer_id"]
        number_of_points = self.options["number_of_points"]

        partials["shaft_power_in", "shaft_power_out"] = (
            np.ones(number_of_points)
            / inputs[
                "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":efficiency"
            ]
        )
        partials[
            "shaft_power_in",
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":efficiency",
        ] = (
            -inputs["shaft_power_out"]
            / inputs[
                "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":efficiency"
            ]
            ** 2.0
        )
