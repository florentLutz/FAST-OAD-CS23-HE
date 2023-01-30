# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesModuleCurrent(om.ExplicitComponent):
    """
    Computation of the current going into each module, assume each module provide an equal
    amount of current.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )

    def setup(self):

        battery_pack_id = self.options["battery_pack_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("dc_current_out", units="A", val=np.full(number_of_points, np.nan))
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            val=np.nan,
            desc="Number of modules in parallel inside the battery pack",
        )

        self.add_output("current_one_module", units="A", val=np.full(number_of_points, 20.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        outputs["current_one_module"] = (
            inputs["dc_current_out"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        battery_pack_id = self.options["battery_pack_id"]
        number_of_points = self.options["number_of_points"]

        partials["current_one_module", "dc_current_out"] = (
            np.eye(number_of_points)
            / inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules"
            ]
        )
        partials[
            "current_one_module",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
        ] = (
            -inputs["dc_current_out"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules"
            ]
            ** 2.0
        )
