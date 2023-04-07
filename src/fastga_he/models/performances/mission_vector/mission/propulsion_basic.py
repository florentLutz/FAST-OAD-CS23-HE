# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from ..constants import HE_SUBMODEL_ENERGY_CONSUMPTION

oad.RegisterSubmodel.active_models[
    HE_SUBMODEL_ENERGY_CONSUMPTION
] = "fastga_he.submodel.performances.energy_consumption.basic"


@oad.RegisterSubmodel(
    HE_SUBMODEL_ENERGY_CONSUMPTION, "fastga_he.submodel.performances.energy_consumption.basic"
)
class FuelConsumed(om.ExplicitComponent):
    """Computes the fuel consumed at each time step."""

    def initialize(self):

        # We have to declare them even if not used to preserve compatibility
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare(
            name="power_train_file_path",
            default="",
            desc="Path to the file containing the description of the power",
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

        ########## NOT USED BUT NEEDED TO ENSURE COMPATIBILITY #####################################
        self.options.declare(
            name="pre_condition_voltage",
            default=False,
            desc="Boolean to pre_condition the voltages of the different components of the PT, "
            "can save some time in specific cases",
            allow_none=False,
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
        # Unused in this module but needed in the power train builder
        self.add_input(
            "exterior_temperature_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="degK",
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
            units="kg/s/N",
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
