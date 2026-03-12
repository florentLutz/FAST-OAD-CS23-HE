# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
from CoolProp.CoolProp import PropsSI

from .constant import fluid_name_dict


class FluidThermalConductivity(om.ExplicitComponent):
    """
    Fluid specific heat capacity calculation for heat transfer models.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "fluid",
            default="air",
            types=str,
            desc="Fluid type: air, water, hydrogen, ammonia, etc.",
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("fluid_temperature", val=np.nan, units="K", shape=number_of_points)
        self.add_input("fluid_pressure", val=np.nan, units="Pa", shape=number_of_points)

        self.add_output(
            "fluid_thermal_conductivity", val=0.024, units="W/m/K", shape=number_of_points
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fluid = self.options["fluid"]
        number_of_points = self.options["number_of_points"]

        temperature = inputs["fluid_temperature"]
        pressure = inputs["fluid_pressure"]

        if fluid not in fluid_name_dict:
            raise ValueError(f"Unknown fluid: {fluid}")

        default_thermal_conductivity = np.array(
            [
                PropsSI("L", "T", t, "P", p, fluid_name_dict[fluid])
                for t, p in zip(temperature, pressure)
            ]
        )

        conditions = [
            (fluid == "hydrogen") & (temperature < 20.325),
            (fluid == "potassium formate") & ((temperature < 173.15) | (temperature > 313.15)),
            fluid == "liquid hydrogen",
        ]

        choices = [
            np.full(number_of_points, 1.323),
            np.array([PropsSI("L", "T", 313.15, "P", p, fluid_name_dict[fluid]) for p in pressure]),
            np.full(number_of_points, 70.78),
        ]

        outputs["fluid_thermal_conductivity"] = np.select(
            conditions, choices, default=default_thermal_conductivity
        )
