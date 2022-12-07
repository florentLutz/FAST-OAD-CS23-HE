# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
import fastoad.api as oad
from stdatm import Atmosphere

from ..constants import HE_SUBMODEL_ENERGY_CONSUMPTION


@oad.RegisterSubmodel(
    HE_SUBMODEL_ENERGY_CONSUMPTION, "fastga_he.submodel.performances.energy_consumption.basic"
)
class FuelConsumed(om.ExplicitComponent):
    """Computes the fuel consumed at each time step."""

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "thrust_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="N",
        )
        self.add_input(
            "altitude_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="m",
        )
        self.add_input(
            "time_step_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="s",
        )
        self.add_input(
            "true_airspeed_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="m/s",
        )
        self.add_input(
            "engine_setting_econ", shape=number_of_points + 2, val=np.full(number_of_points + 2, 1)
        )
        self.add_input(
            "data:propulsion:IC_engine:tsfc",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, 7.3e-6),
            units="s",
        )

        self.add_output(
            "fuel_consumed_t_econ",
            val=np.full(number_of_points + 2, 0.0),
            desc="fuel consumed at each time step",
            units="kg",
        )
        self.add_output(
            "non_consumable_energy_t_econ",
            val=np.full(number_of_points + 2, 0.0),
            desc="fuel consumed at each time step",
            units="W*h",
        )
        self.add_output(
            "thrust_rate_t_econ",
            val=np.full(number_of_points + 2, 0.5),
            desc="thrust ratio at each time step",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        time_step = inputs["time_step_econ"]
        thrust = inputs["thrust_econ"]

        tsfc = inputs["data:propulsion:IC_engine:tsfc"]

        outputs["fuel_consumed_t_econ"] = thrust * tsfc * time_step
        outputs["thrust_rate_t_econ"] = 0.0
