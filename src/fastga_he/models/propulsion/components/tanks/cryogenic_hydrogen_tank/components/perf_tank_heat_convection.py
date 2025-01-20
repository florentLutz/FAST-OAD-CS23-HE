# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesCryogenicHydrogenTankConvection(om.ExplicitComponent):
    """
    Computation of the heat convection at the outer surface of the cryogenic tank
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        self.options.declare(
            name="cryogenic_hydrogen_tank_id",
            default=None,
            desc="Identifier of the cryogenic hydrogen tank",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        self.add_input(
            "tank_nusselt_number",
            val=np.full(number_of_points, np.nan),
            desc="Tank Nusselt number at each time step",
        )

        self.add_input(
            name="exterior_temperature",
            units="K",
            val=np.full(number_of_points, np.nan),
        )

        self.add_input(
            name="skin_temperature",
            units="K",
            val=np.full(number_of_points, np.nan),
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:length",
            val=np.nan,
            units="m",
        )

        self.add_input(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:"
            + cryogenic_hydrogen_tank_id
            + ":dimension:outer_diameter",
            units="m",
            val=np.nan,
            desc="Outer diameter of the hydrogen gas tank",
        )

        self.add_input(
            "air_thermal_conductivity",
            units="W/m/K",
            val=np.full(number_of_points, np.nan),
            desc="air thermal conductivity at each time step",
        )

        self.add_output(
            "heat_convection",
            units="W",
            val=np.full(number_of_points, 10.2185),
            desc="Hydrogen boil-off in the tank at each time step",
        )

        self.declare_partials(
            of="heat_convection",
            wrt=[
                "air_thermal_conductivity",
                "skin_temperature",
                "exterior_temperature",
                "tank_nusselt_number",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

        self.declare_partials(
            of="heat_convection",
            wrt=[
                input_prefix + ":dimension:outer_diameter",
                input_prefix + ":dimension:length",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        d = inputs[input_prefix + ":dimension:outer_diameter"]
        area = np.pi * d ** 2 + np.pi * d * inputs[input_prefix + ":dimension:length"]

        h = inputs["air_thermal_conductivity"] * inputs["tank_nusselt_number"] / d

        outputs["heat_convection"] = (
            h * area * (inputs["exterior_temperature"] - inputs["skin_temperature"])
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        cryogenic_hydrogen_tank_id = self.options["cryogenic_hydrogen_tank_id"]
        number_of_points = self.options["number_of_points"]
        input_prefix = (
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:" + cryogenic_hydrogen_tank_id
        )

        d = inputs[input_prefix + ":dimension:outer_diameter"]
        area = np.pi * d ** 2 + np.pi * d * inputs[input_prefix + ":dimension:length"]
        h = inputs["air_thermal_conductivity"] * inputs["tank_nusselt_number"] / d

        partials["heat_convection", "air_thermal_conductivity"] = (
            inputs["tank_nusselt_number"]
            / d
            * area
            * (inputs["exterior_temperature"] - inputs["skin_temperature"])
        )
        partials["heat_convection", "tank_nusselt_number"] = (
            inputs["air_thermal_conductivity"]
            / d
            * area
            * (inputs["exterior_temperature"] - inputs["skin_temperature"])
        )
        partials["heat_convection", "exterior_temperature"] = h * area * np.ones(number_of_points)
        partials["heat_convection", "skin_temperature"] = -h * area * np.ones(number_of_points)
        partials["heat_convection", input_prefix + ":dimension:length"] = (
            h * np.pi * d * (inputs["exterior_temperature"] - inputs["skin_temperature"])
        )
        partials["heat_convection", input_prefix + ":dimension:outer_diameter"] = (
            inputs["air_thermal_conductivity"]
            * inputs["tank_nusselt_number"]
            * np.pi
            * (inputs["exterior_temperature"] - inputs["skin_temperature"])
        )
