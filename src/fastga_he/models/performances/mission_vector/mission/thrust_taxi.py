# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om
from fastoad.constants import EngineSetting

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from stdatm import Atmosphere

from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet


class ThrustTaxi(om.ExplicitComponent):
    """Computes the fuel consumed during the taxi phases."""

    def setup(self):

        self.add_input("data:mission:sizing:taxi_out:thrust_rate", np.nan)
        self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units="m/s")
        self.add_output("data:mission:sizing:taxi_out:thrust", 1500, units="N")

        self.add_input("data:mission:sizing:taxi_in:thrust_rate", np.nan)
        self.add_input("data:mission:sizing:taxi_in:speed", np.nan, units="m/s")
        self.add_output("data:mission:sizing:taxi_in:thrust", 1500, units="N")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # TODO: Proper formula based on the speed, AoA = 0.0 and aerodynamics

        outputs["data:mission:sizing:taxi_out:thrust"] = 300.0
        outputs["data:mission:sizing:taxi_in:thrust"] = 300.0
