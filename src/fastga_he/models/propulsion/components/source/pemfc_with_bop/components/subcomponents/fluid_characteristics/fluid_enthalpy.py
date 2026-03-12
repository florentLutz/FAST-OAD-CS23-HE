# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
from CoolProp.CoolProp import PropsSI

from .constant import fluid_name_dict


class FluidEnthalpy(om.ExplicitComponent):
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

        self.add_input("fluid_temperature", val=np.nan, units="K", shape=number_of_points)
        self.add_input("fluid_pressure", val=np.nan, units="Pa", shape=number_of_points)

        self.add_output("fluid_enthalpy", val=250000.0, units="J/kg", shape=number_of_points)

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

        temperature = inputs["fluid_temperature"]
        pressure = inputs["fluid_pressure"]

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

        temperature = inputs["fluid_temperature"]
        pressure = inputs["fluid_pressure"]

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
                    PropsSI("d(H)/d(P)|T", "T", 313.15, "P", p, fluid_name_dict[fluid])
                    for p in pressure
                ]
            ),
            np.zeros(number_of_points),
        ]

        partials["fluid_enthalpy", "fluid_temperature"] = np.select(
            conditions, choice_temperature, default=default_d_h_dt
        )

        partials["fluid_enthalpy", "fluid_pressure"] = np.select(
            conditions, choice_pressure, default=default_d_h_dp
        )
