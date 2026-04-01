# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
from CoolProp.CoolProp import PropsSI

from .constant import fluid_name_dict, fluid_specific_heat_capacity_dict


class FluidSpecificHeatCapacity(om.Group):
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
        fluid = self.options["fluid"]

        self.add_subsystem(
            "property_check",
            _PropertyCheck(number_of_points=number_of_points),
            promotes=["fluid_temperature", "fluid_pressure"],
        )
        self.add_subsystem(
            "specific_heat_capacity",
            _SpecificHeatCapacity(number_of_points=number_of_points, fluid=fluid),
            promotes=["fluid_specific_heat_capacity"],
        )

        self.connect("property_check.temperature", "specific_heat_capacity.temperature")
        self.connect("property_check.pressure", "specific_heat_capacity.pressure")


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
        temperature = inputs["fluid_temperature"]
        pressure = inputs["fluid_pressure"]

        temperature_conditions = [
            temperature < 250.0,
            temperature > 450.0,
            np.isnan(temperature),
        ]
        pressure_conditions = [pressure < 1e3, pressure > 1e8, np.isnan(pressure)]

        clipped_temperature = [250.0, 450.0, 300.0]
        clipped_pressure = [1e3, 1e8, 101325.0]

        outputs["temperature"] = np.select(
            temperature_conditions, clipped_temperature, default=temperature
        )
        outputs["pressure"] = np.select(pressure_conditions, clipped_pressure, default=pressure)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        temperature = inputs["fluid_temperature"]
        pressure = inputs["fluid_pressure"]

        temperature_conditions = [
            temperature < 250.0,
            temperature > 450.0,
            np.isnan(temperature),
        ]
        pressure_conditions = [pressure < 1e3, pressure > 1e8, np.isnan(pressure)]

        clipped_temperature = [250.0, 450.0, 300.0]
        clipped_pressure = [1e3, 1e8, 101325.0]

        partials["temperature", "fluid_temperature"] = np.where(
            np.select(temperature_conditions, clipped_temperature, default=temperature)
            == inputs["fluid_temperature"],
            1.0,
            1e-6,
        )

        partials["pressure", "fluid_pressure"] = np.where(
            np.select(pressure_conditions, clipped_pressure, default=pressure)
            == inputs["fluid_pressure"],
            1.0,
            1e-6,
        )


class _SpecificHeatCapacity(om.ExplicitComponent):
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
        fluid = self.options["fluid"]

        self.add_input("temperature", val=np.nan, units="K", shape=number_of_points)
        self.add_input("pressure", val=np.nan, units="Pa", shape=number_of_points)

        self.add_output(
            "fluid_specific_heat_capacity",
            val=fluid_specific_heat_capacity_dict[fluid],
            units="J/kg/K",
            shape=number_of_points,
        )

    def setup_partials(self):
        number_of_points = self.options["number_of_points"]

        if number_of_points > 1:
            self.declare_partials(
                of="*",
                wrt="*",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
            )
        else:
            self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fluid = self.options["fluid"]
        number_of_points = self.options["number_of_points"]

        temperature = inputs["temperature"]
        pressure = inputs["pressure"]

        if fluid not in fluid_name_dict:
            raise ValueError(f"Unknown fluid: {fluid}")

        default_specific_heat_capacity = np.array(
            [
                PropsSI("C", "T", t, "P", p, fluid_name_dict[fluid])
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
            np.array([PropsSI("C", "T", 313.15, "P", p, fluid_name_dict[fluid]) for p in pressure]),
            np.full(number_of_points, 70.78),
        ]

        outputs["fluid_specific_heat_capacity"] = np.select(
            conditions, choices, default_specific_heat_capacity
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fluid = self.options["fluid"]
        number_of_points = self.options["number_of_points"]

        temperature = inputs["temperature"]
        pressure = inputs["pressure"]

        fluid_string = fluid_name_dict[fluid]
        is_incompressible = fluid_string.startswith("INCOMP::")

        if is_incompressible:
            dt = 1e-3
            dp = 1.0

            default_d_cp_dt = np.array(
                [
                    (
                        PropsSI("C", "T", t + dt, "P", p, fluid_string)
                        - PropsSI("C", "T", t - dt, "P", p, fluid_string)
                    )
                    / (2.0 * dt)
                    for t, p in zip(temperature, pressure)
                ]
            )
            default_d_cp_dp = np.array(
                [
                    (
                        PropsSI("C", "T", t, "P", p + dp, fluid_string)
                        - PropsSI("C", "T", t, "P", p - dp, fluid_string)
                    )
                    / (2.0 * dp)
                    for t, p in zip(temperature, pressure)
                ]
            )
        else:
            default_d_cp_dt = np.array(
                [
                    PropsSI("d(C)/d(T)|P", "T", t, "P", p, fluid_string)
                    for t, p in zip(temperature, pressure)
                ]
            )
            default_d_cp_dp = np.array(
                [
                    PropsSI("d(C)/d(P)|T", "T", t, "P", p, fluid_string)
                    for t, p in zip(temperature, pressure)
                ]
            )

        conditions = [
            (fluid == "hydrogen") & (temperature < 20.325),
            (fluid == "potassium formate") & ((temperature < 173.15) | (temperature > 313.15)),
            fluid == "liquid hydrogen",
        ]

        choice_temperature = [
            np.zeros(number_of_points),
            np.zeros(number_of_points),
            np.zeros(number_of_points),
        ]

        choice_pressure = [
            np.zeros(number_of_points),
            np.array(
                [
                    (
                        (
                            PropsSI("C", "T", 313.15, "P", p + 1.0, fluid_string)
                            - PropsSI("C", "T", 313.15, "P", p - 1.0, fluid_string)
                        )
                        / 2.0
                        if is_incompressible
                        else PropsSI("d(C)/d(P)|T", "T", 313.15, "P", p, fluid_string)
                    )
                    for p in pressure
                ]
            ),
            np.zeros(number_of_points),
        ]

        partials["fluid_specific_heat_capacity", "temperature"] = np.select(
            conditions, choice_temperature, default=default_d_cp_dt
        )

        partials["fluid_specific_heat_capacity", "pressure"] = np.select(
            conditions, choice_pressure, default=default_d_cp_dp
        )
