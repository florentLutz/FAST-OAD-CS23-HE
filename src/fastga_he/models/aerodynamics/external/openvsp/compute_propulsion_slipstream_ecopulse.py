# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

import fastoad.api as oad

from stdatm import Atmosphere

from fastga.models.aerodynamics.constants import SUBMODEL_THRUST_POWER_SLIPSTREAM


@oad.RegisterSubmodel(
    SUBMODEL_THRUST_POWER_SLIPSTREAM,
    "fastga.submodel.aerodynamics.wing.slipstream.thrust_power_computation.ecopulse",
)
class PropulsionForDPComputationEcoPulse(om.ExplicitComponent):
    """Computes thrust and shaft power for Ecopulse computation fixed values for now."""

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        # Dummy variables
        self.add_input("data:geometry:wing:area", val=np.nan, units="ft**2")
        self.add_input("mach", val=np.nan)
        self.add_input("altitude", val=np.nan, units="m")

        self.add_output("thrust", val=0, units="N")
        self.add_output("shaft_power", val=1, units="W")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # 6 engine of 100 kW
        shaft_power = 100e3
        true_airspeed = (
            Atmosphere(inputs["altitude"], altitude_in_feet=False).speed_of_sound * inputs["mach"]
        )

        # Assumed efficiency of 0.75
        outputs["thrust"] = shaft_power * 0.75 / true_airspeed
        outputs["shaft_power"] = shaft_power
