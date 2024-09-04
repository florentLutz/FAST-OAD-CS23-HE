# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesRPMIn(om.ExplicitComponent):
    """
    Component which computes the input rpm based on the output rpm and gear ratio.
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
            name="data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":gear_ratio",
            val=np.nan,
            desc="Gear ratio of the speed reducer",
        )
        self.add_input("rpm_out", units="min**-1", val=np.nan, shape=number_of_points)

        self.add_output("rpm_in", units="min**-1", val=5000.0, shape=number_of_points)

        self.declare_partials(
            of="rpm_in",
            wrt="rpm_out",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="rpm_in",
            wrt="data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":gear_ratio",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        speed_reducer_id = self.options["speed_reducer_id"]

        outputs["rpm_in"] = (
            inputs["rpm_out"]
            * inputs[
                "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":gear_ratio"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        speed_reducer_id = self.options["speed_reducer_id"]
        number_of_points = self.options["number_of_points"]

        partials["rpm_in", "rpm_out"] = (
            np.ones(number_of_points)
            * inputs[
                "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":gear_ratio"
            ]
        )
        partials[
            "rpm_in",
            "data:propulsion:he_power_train:speed_reducer:" + speed_reducer_id + ":gear_ratio",
        ] = inputs["rpm_out"]
