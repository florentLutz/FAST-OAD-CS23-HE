# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
from CoolProp.CoolProp import PropsSI

from .constant import fluid_name_dict


class FluidPrandtlNumber(om.Group):
    """
    Fluid Prandtl number calculation for heat transfer models.
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
        fluid = self.options["fluid"]

        self.add_subsystem(
            "property_check",
            _PropertyCheck(number_of_points=number_of_points),
            promotes=["fluid_temperature", "fluid_pressure"],
        )
        self.add_subsystem(
            "prandtl_number",
            _PrandtlNumber(number_of_points=number_of_points, fluid=fluid),
            promotes=["fluid_prandtl_number"],
        )

        self.connect("property_check.temperature", "prandtl_number.temperature")
        self.connect("property_check.pressure", "prandtl_number.pressure")


class _PropertyCheck(om.ExplicitComponent):
    """
    Fluid property check.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input("fluid_temperature", val=np.nan, units="K", shape=number_of_points)
        self.add_input("fluid_pressure", val=np.nan, units="Pa", shape=number_of_points)

        self.add_output("temperature", val=300.0, units="K", shape=number_of_points)
        self.add_output("pressure", val=101325.0, units="Pa", shape=number_of_points)

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        if number_of_points > 1:
            self.declare_partials(
                of="temperature",
                wrt="fluid_temperature",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
            )
            self.declare_partials(
                of="pressure",
                wrt="fluid_pressure",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
            )
        else:
            self.declare_partials(of="temperature", wrt="fluid_temperature", method="exact")
            self.declare_partials(of="pressure", wrt="fluid_pressure", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["temperature"] = np.clip(inputs["fluid_temperature"], 200.0, 450.0)
        outputs["pressure"] = np.clip(inputs["fluid_pressure"], 1e3, 1e8)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        clipped_temperature = np.clip(inputs["fluid_temperature"], 200.0, 450.0)
        clipped_pressure = np.clip(inputs["fluid_pressure"], 1e3, 1e8)

        partials["temperature", "fluid_temperature"] = np.where(
            clipped_temperature == inputs["fluid_temperature"], 1.0, 0.0
        )

        partials["pressure", "fluid_pressure"] = np.where(
            clipped_pressure == inputs["fluid_pressure"], 1.0, 0.0
        )


class _PrandtlNumber(om.ExplicitComponent):
    """
    Fluid Prandtl number calculation for heat transfer models.
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

        self.add_input("temperature", val=np.nan, units="K", shape=number_of_points)
        self.add_input("pressure", val=np.nan, units="Pa", shape=number_of_points)

        self.add_output(
            "fluid_prandtl_number", val=1006.0, units="unitless", shape=number_of_points
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fluid = self.options["fluid"]
        number_of_points = self.options["number_of_points"]

        temperature = inputs["temperature"]
        pressure = inputs["pressure"]

        if fluid not in fluid_name_dict:
            raise ValueError(f"Unknown fluid: {fluid}")

        default_prandtl_number = np.array(
            [
                PropsSI("PRANDTL", "T", t, "P", p, fluid_name_dict[fluid])
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
            np.array(
                [PropsSI("PRANDTL", "T", 313.15, "P", p, fluid_name_dict[fluid]) for p in pressure]
            ),
            np.full(number_of_points, 70.78),
        ]

        outputs["fluid_prandtl_number"] = np.select(
            conditions, choices, default=default_prandtl_number
        )
