# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesModuleCRate(om.ExplicitComponent):
    """
    Computation of the C-rate of each module, assume each module provide an equal amount of
    current and the capacity of one module is equal to the capacity of one cell.
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

        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        self.add_input("current_one_module", units="A", val=np.full(number_of_points, np.nan))
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:capacity",
            val=np.nan,
            units="A*h",
            desc="Capacity of the cell used for the assembly of the battery pack",
        )

        self.add_output("c_rate", units="h**-1", val=np.full(number_of_points, 1.0))

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        battery_pack_id = self.options["battery_pack_id"]

        outputs["c_rate"] = (
            inputs["current_one_module"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:capacity"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        partials["c_rate", "current_one_module"] = (
            np.eye(number_of_points)
            / inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:capacity"
            ]
        )
        partials[
            "c_rate",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:capacity",
        ] = (
            -inputs["current_one_module"]
            / inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":cell:capacity"
            ]
            ** 2.0
        )
