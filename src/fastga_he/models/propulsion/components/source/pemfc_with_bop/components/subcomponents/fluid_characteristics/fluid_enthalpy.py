# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
from CoolProp.CoolProp import PropsSI

from .constant import fluid_name_dict, fluid_enthalpy_dict


class FluidEnthalpy(om.Group):
    """
    Fluid enthalpy calculation for heat transfer models.
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
            "enthalpy",
            _Enthalpy(number_of_points=number_of_points, fluid=fluid),
            promotes=["fluid_enthalpy"],
        )

        self.connect("property_check.temperature", "enthalpy.temperature")
        self.connect("property_check.pressure", "enthalpy.pressure")


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

        condition = (
            (temperature > 250.0)
            & (temperature < 373.1)
            & (pressure > 1e3)
            & (pressure < 1e8)
            & ~np.isnan(temperature)
            & ~np.isnan(pressure)
        )

        outputs["temperature"] = np.where(condition, temperature, 300.0)
        outputs["pressure"] = np.where(condition, pressure, 101325.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        temperature = inputs["fluid_temperature"]
        pressure = inputs["fluid_pressure"]

        condition = (
            (temperature > 250.0)
            & (temperature < 373.1)
            & (pressure > 1e3)
            & (pressure < 1e8)
            & ~np.isnan(temperature)
            & ~np.isnan(pressure)
        )

        partials["temperature", "fluid_temperature"] = np.where(condition, 1.0, 1e-6)

        partials["pressure", "fluid_pressure"] = np.where(condition, 1.0, 1e-6)


class _Enthalpy(om.ExplicitComponent):
    """
    Fluid enthalpy calculation for heat transfer models.
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
            "fluid_enthalpy", val=fluid_enthalpy_dict[fluid], units="J/kg", shape=number_of_points
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

        temperature = np.clip(inputs["temperature"], 250.0, 373.1)
        pressure = np.clip(inputs["pressure"], 1e3, 1e8)

        if fluid not in fluid_name_dict:
            raise ValueError(f"Unknown fluid: {fluid}")

        default_enthalpy = np.array(
            [
                PropsSI("H", "T", t, "P", p, fluid_name_dict[fluid])
                for t, p in zip(temperature, pressure)
            ]
        )

        conditions = [
            (fluid == "hydrogen") & (temperature < 20.325),
            (fluid == "potassium formate") & ((temperature < 173.15) | (temperature > 313.15)),
            fluid == "liquid hydrogen",
        ]

        choices = [
            np.full(number_of_points, 0.6866),
            np.array([PropsSI("D", "T", 313.15, "P", p, fluid_name_dict[fluid]) for p in pressure]),
            np.full(number_of_points, 0.0),
        ]

        outputs["fluid_enthalpy"] = np.select(conditions, choices, default=default_enthalpy)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fluid = self.options["fluid"]
        number_of_points = self.options["number_of_points"]

        temperature = np.clip(inputs["temperature"], 250.0, 373.1)
        pressure = np.clip(inputs["pressure"], 1e3, 1e8)

        fluid_string = fluid_name_dict[fluid]
        is_incompressible = fluid_string.startswith("INCOMP::")

        if is_incompressible:
            dt = 1e-3
            dp = 1.0

            default_d_h_dt = np.array(
                [
                    (
                        PropsSI("H", "T", t + dt, "P", p, fluid_string)
                        - PropsSI("H", "T", t - dt, "P", p, fluid_string)
                    )
                    / (2.0 * dt)
                    for t, p in zip(temperature, pressure)
                ]
            )
            default_d_h_dp = np.array(
                [
                    (
                        PropsSI("H", "T", t, "P", p + dp, fluid_string)
                        - PropsSI("H", "T", t, "P", p - dp, fluid_string)
                    )
                    / (2.0 * dp)
                    for t, p in zip(temperature, pressure)
                ]
            )
        else:
            default_d_h_dt = np.array(
                [
                    PropsSI("d(H)/d(T)|P", "T", t, "P", p, fluid_name_dict[fluid])
                    for t, p in zip(temperature, pressure)
                ]
            )
            default_d_h_dp = np.array(
                [
                    PropsSI("d(H)/d(P)|T", "T", t, "P", p, fluid_name_dict[fluid])
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
                            PropsSI("H", "T", 313.15, "P", p + 1.0, fluid_string)
                            - PropsSI("H", "T", 313.15, "P", p - 1.0, fluid_string)
                        )
                        / 2.0
                        if is_incompressible
                        else PropsSI("d(H)/d(P)|T", "T", 313.15, "P", p, fluid_string)
                    )
                    for p in pressure
                ]
            ),
            np.zeros(number_of_points),
        ]

        partials["fluid_enthalpy", "temperature"] = np.select(
            conditions, choice_temperature, default=default_d_h_dt
        )

        partials["fluid_enthalpy", "pressure"] = np.select(
            conditions, choice_pressure, default=default_d_h_dp
        )
