# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesBatteryLosses(om.ExplicitComponent):
    """
    Computation of the total losses inside the battery based on the losses at cell level and the
    number of cells.
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

        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
            val=np.nan,
            desc="Number of cells in series inside one battery module",
        )
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
            val=np.nan,
            desc="Number of modules in parallel inside the battery pack",
        )
        self.add_input("losses_cell", units="W", val=np.full(number_of_points, np.nan))

        self.add_output("losses_battery", units="W", val=np.full(number_of_points, 1e3))

        self.declare_partials(
            of="losses_battery",
            wrt="losses_cell",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="losses_battery",
            wrt=[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":number_modules",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs["losses_battery"] = (
            inputs["losses_cell"]
            * inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells"
            ]
            * inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        partials["losses_battery", "losses_cell"] = (
            np.ones(number_of_points)
            * inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells"
            ]
            * inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules"
            ]
        )
        partials[
            "losses_battery",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
        ] = (
            inputs["losses_cell"]
            * inputs[
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules"
            ]
        )
        partials[
            "losses_battery",
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":number_modules",
        ] = (
            inputs["losses_cell"]
            * inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells"
            ]
        )
