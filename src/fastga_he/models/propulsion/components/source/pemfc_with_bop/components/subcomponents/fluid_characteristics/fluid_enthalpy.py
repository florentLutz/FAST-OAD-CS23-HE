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
            "fluid",
            default="air",
            types=str,
            desc="Fluid type: air, water, hydrogen, ammonia, etc.",
        )

    def setup(self):
        self.add_input("fluid_temperature", val=np.nan, units="K")
        self.add_input("fluid_pressure", val=np.nan, units="Pa")

        self.add_output("fluid_enthalpy", val=250000.0, units="J/kg")

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        temperature = inputs["fluid_temperature"]
        pressure = inputs["fluid_pressure"]
        fluid = self.options["fluid"]

        if fluid == "hydrogen" and temperature < 20.325:
            outputs["fluid_enthalpy"] = 0.6866
        elif fluid == "potassium formate" and not 173.15 <= temperature <= 313.15:
            outputs["fluid_enthalpy"] = PropsSI(
                "H", "T", 313.15, "P", pressure, fluid_name_dict[fluid]
            )
        elif fluid == "liquid hydrogen":
            outputs["fluid_enthalpy"] = 0.0
        else:
            if fluid not in fluid_name_dict:
                raise ValueError(f"Unknown fluid: {fluid}")

            outputs["fluid_enthalpy"] = PropsSI(
                "H", "T", temperature, "P", pressure, fluid_name_dict[fluid]
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        temperature = inputs["fluid_temperature"]
        pressure = inputs["fluid_pressure"]
        fluid = self.options["fluid"]

        if fluid == "hydrogen" and temperature < 20.325:
            partials["fluid_enthalpy", "fluid_temperature"] = 0.0
            partials["fluid_enthalpy", "fluid_pressure"] = 0.0
        elif fluid == "potassium formate" and not 173.15 <= temperature <= 313.15:
            partials["fluid_enthalpy", "fluid_temperature"] = 0.0
            partials["fluid_enthalpy", "fluid_pressure"] = PropsSI(
                "d(H)/d(P)|T", "T", temperature, "P", pressure, fluid_name_dict[fluid]
            )
        elif fluid == "liquid hydrogen":
            partials["fluid_enthalpy", "fluid_temperature"] = 0.0
            partials["fluid_enthalpy", "fluid_pressure"] = 0.0
        else:
            partials["fluid_enthalpy", "fluid_temperature"] = PropsSI(
                "d(H)/d(T)|P", "T", temperature, "P", pressure, fluid_name_dict[fluid]
            )
            partials["fluid_enthalpy", "fluid_pressure"] = PropsSI(
                "d(H)/d(P)|T", "T", temperature, "P", pressure, fluid_name_dict[fluid]
            )
