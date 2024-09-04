# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesModuleVoltage(om.ExplicitComponent):
    """
    Computation of the voltage provided by each module, assume each module has the same voltage
    and that each cell in the module (all in series) has the same terminal voltage.
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

        self.add_input("terminal_voltage", units="V", val=np.full(number_of_points, np.nan))
        self.add_input(
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
            val=np.nan,
            desc="Number of cells in series inside one battery module",
        )

        self.add_output("module_voltage", units="V", val=np.full(number_of_points, 500.0))

        self.declare_partials(
            of="module_voltage",
            wrt="terminal_voltage",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="module_voltage",
            wrt="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs["module_voltage"] = (
            inputs["terminal_voltage"]
            * inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        number_of_points = self.options["number_of_points"]
        battery_pack_id = self.options["battery_pack_id"]

        partials["module_voltage", "terminal_voltage"] = np.full(
            number_of_points,
            *inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":module:number_cells"
            ],
        )
        partials[
            "module_voltage",
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":module:number_cells",
        ] = inputs["terminal_voltage"]
