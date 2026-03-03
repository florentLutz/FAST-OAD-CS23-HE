# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesHeatCapacityRatio(om.ExplicitComponent):
    """
    Computation of the heat capacity ratio in heat exchange system.
    """

    def setup(self):
        self.add_input(
            name="air_heat_capacity",
            units="W/K",
            val=np.nan,
        )
        self.add_input(
            name="coolant_heat_capacity",
            units="W/K",
            val=np.nan,
        )

        self.add_output(
            name="min_heat_capacity",
            units="W/K",
            val=2.64,
        )
        self.add_output(
            name="heat_capacity_ratio",
            units=None,
            val=0.8,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if inputs["air_heat_capacity"] < inputs["coolant_heat_capacity"]:
            outputs["min_heat_capacity"] = inputs["air_heat_capacity"]
            outputs["heat_capacity_ratio"] = (
                inputs["air_heat_capacity"] / inputs["coolant_heat_capacity"]
            )

        else:
            outputs["min_heat_capacity"] = inputs["coolant_heat_capacity"]
            outputs["heat_capacity_ratio"] = (
                inputs["coolant_heat_capacity"] / inputs["air_heat_capacity"]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None, discrete_outputs=None):
        if inputs["air_heat_capacity"] < inputs["coolant_heat_capacity"]:
            partials["min_heat_capacity", "air_heat_capacity"] = 1.0

            partials["min_heat_capacity", "coolant_heat_capacity"] = 0.0

            partials["heat_capacity_ratio", "air_heat_capacity"] = (
                1.0 / inputs["coolant_heat_capacity"]
            )

            partials["heat_capacity_ratio", "coolant_heat_capacity"] = -inputs[
                "air_heat_capacity"
            ] / (inputs["coolant_heat_capacity"] ** 2.0)

        else:
            partials["min_heat_capacity", "air_heat_capacity"] = 0.0

            partials["min_heat_capacity", "coolant_heat_capacity"] = 1.0

            partials["heat_capacity_ratio", "air_heat_capacity"] = -inputs[
                "coolant_heat_capacity"
            ] / (inputs["air_heat_capacity"] ** 2.0)

            partials["heat_capacity_ratio", "coolant_heat_capacity"] = (
                1.0 / inputs["air_heat_capacity"]
            )
