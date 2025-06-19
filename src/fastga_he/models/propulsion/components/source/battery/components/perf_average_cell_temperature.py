#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAverageCellTemperature(om.ExplicitComponent):
    """
    Computation of the time step weighted average temperature of the cell.
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

        self.add_input("cell_temperature", units="degK", val=np.full(number_of_points, np.nan))
        self.add_input("time_step", units="h", val=np.full(number_of_points, np.nan))

        self.add_output(
            name="data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":average_cell_temperature",
            val=288.15,
            units="degK",
            desc="Time step averaged temperature of the cell during the mission",
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        self.declare_partials(
            of="*",
            wrt="*",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        outputs[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":average_cell_temperature"
        ] = np.sum(inputs["cell_temperature"] * inputs["time_step"]) / np.sum(inputs["time_step"])

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]

        sum_time_step = np.sum(inputs["time_step"])

        partials[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":average_cell_temperature",
            "cell_temperature",
        ] = inputs["time_step"] / sum_time_step

        partials[
            "data:propulsion:he_power_train:battery_pack:"
            + battery_pack_id
            + ":average_cell_temperature",
            "time_step",
        ] = (
            sum_time_step * inputs["cell_temperature"]
            - np.sum(inputs["cell_temperature"] * inputs["time_step"])
        ) / sum_time_step**2.0
