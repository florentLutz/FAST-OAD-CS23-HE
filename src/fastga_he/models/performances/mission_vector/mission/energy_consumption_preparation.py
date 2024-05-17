# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import logging

import numpy as np
import openmdao.api as om
from stdatm import Atmosphere

_LOGGER = logging.getLogger(__name__)

RHO_SL = Atmosphere(0.0).density


class PrepareForEnergyConsumption(om.ExplicitComponent):
    """
    Prepare the different vector for the energy consumption computation, which means some name
    will be changed because we need to add the point corresponding to the taxi computation.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units="m/s")
        self.add_input("data:mission:sizing:taxi_out:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:taxi_out:thrust", 1500, units="N")

        self.add_input("data:mission:sizing:taxi_in:speed", np.nan, units="m/s")
        self.add_input("data:mission:sizing:taxi_in:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:taxi_in:thrust", 1500, units="N")

        self.add_input(
            "thrust", shape=number_of_points, val=np.full(number_of_points, np.nan), units="N"
        )
        self.add_input(
            "altitude", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "density",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="kg/m**3",
        )
        self.add_input(
            "exterior_temperature",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="degK",
        )
        self.add_input(
            "time_step", shape=number_of_points, val=np.full(number_of_points, np.nan), units="s"
        )
        self.add_input(
            "true_airspeed",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="m/s",
        )
        self.add_input(
            "engine_setting", shape=number_of_points, val=np.full(number_of_points, np.nan)
        )

        # Econ stands for Energy Consumption, this way we separate the vectors used for the
        # computation of the equilibrium from the one used for the computation of the energy
        # consumption
        self.add_output("thrust_econ", shape=number_of_points + 2, units="N")
        self.add_output("altitude_econ", shape=number_of_points + 2, units="m")
        self.add_output("density_econ", shape=number_of_points + 2, units="kg/m**3")
        self.add_output("exterior_temperature_econ", shape=number_of_points + 2, units="degK")
        self.add_output("time_step_econ", shape=number_of_points + 2, units="s")
        self.add_output("true_airspeed_econ", shape=number_of_points + 2, units="m/s")
        self.add_output("engine_setting_econ", shape=number_of_points + 2)

        self.declare_partials(
            of="thrust_econ",
            wrt=[
                "thrust",
                "data:mission:sizing:taxi_out:thrust",
                "data:mission:sizing:taxi_in:thrust",
            ],
            method="exact",
        )

        self.declare_partials(of="altitude_econ", wrt="altitude", method="exact")
        self.declare_partials(of="density_econ", wrt="density", method="exact")
        self.declare_partials(of="engine_setting_econ", wrt="engine_setting", method="exact")

        self.declare_partials(
            of="exterior_temperature_econ", wrt="exterior_temperature", method="exact"
        )

        self.declare_partials(
            of="time_step_econ",
            wrt=[
                "time_step",
                "data:mission:sizing:taxi_out:duration",
                "data:mission:sizing:taxi_in:duration",
            ],
            method="exact",
        )

        self.declare_partials(
            of="true_airspeed_econ",
            wrt=[
                "true_airspeed",
                "data:mission:sizing:taxi_out:speed",
                "data:mission:sizing:taxi_in:speed",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        thrust_taxi_out = float(inputs["data:mission:sizing:taxi_out:thrust"])
        thrust_taxi_in = float(inputs["data:mission:sizing:taxi_in:thrust"])

        thrust_econ = np.concatenate(
            (np.array([thrust_taxi_out]), inputs["thrust"], np.array([thrust_taxi_in]))
        )
        if np.any(thrust_econ) < 50.0:

            thrust_econ = np.maximum(
                thrust_econ,
                np.full_like(thrust_econ, 50.0),
            )

        outputs["thrust_econ"] = thrust_econ

        outputs["altitude_econ"] = np.concatenate((np.zeros(1), inputs["altitude"], np.zeros(1)))
        outputs["density_econ"] = np.concatenate(
            (np.array([RHO_SL]), inputs["density"], np.array([RHO_SL]))
        )
        outputs["engine_setting_econ"] = np.concatenate(
            (np.ones(1), inputs["engine_setting"], np.ones(1))
        )

        temp_sl = Atmosphere(np.array([0]), altitude_in_feet=True).temperature
        outputs["exterior_temperature_econ"] = np.concatenate(
            (temp_sl, inputs["exterior_temperature"], temp_sl)
        )

        time_step_taxi_out = float(inputs["data:mission:sizing:taxi_out:duration"])
        time_step_taxi_in = float(inputs["data:mission:sizing:taxi_in:duration"])
        # Here we have to do an additional change. Since time step is computed for point i based
        # on time(i+1) - time(i) the last time step of climb will be computed with the first time
        # of cruise which means, since the cruise time step is very wide, that it will be very
        # wide and lead to an overestimation of climb fuel. For this reason we will replace the
        # last time step of climb with the precedent to get a good estimate. This will only serve
        # for the energy consumption calculation. Same remark holds for the end of descent and
        # start of reserve.
        time_step = inputs["time_step"]
        outputs["time_step_econ"] = np.concatenate(
            (np.array([time_step_taxi_out]), time_step, np.array([time_step_taxi_in]))
        )

        tas_taxi_out = float(inputs["data:mission:sizing:taxi_out:speed"])
        tas_taxi_in = float(inputs["data:mission:sizing:taxi_in:speed"])
        outputs["true_airspeed_econ"] = np.concatenate(
            (np.array([tas_taxi_out]), inputs["true_airspeed"], np.array([tas_taxi_in]))
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]

        d_thrust_econ_d_thrust = np.zeros((number_of_points + 2, number_of_points))
        d_thrust_econ_d_thrust_diagonal = np.where(
            inputs["thrust"] > 50.0, np.ones_like(inputs["thrust"]), np.zeros_like(inputs["thrust"])
        )
        d_thrust_econ_d_thrust[1 : number_of_points + 1, :] = np.diag(
            d_thrust_econ_d_thrust_diagonal
        )
        partials["thrust_econ", "thrust"] = d_thrust_econ_d_thrust

        d_thrust_econ_d_thrust_to = np.zeros(number_of_points + 2)
        d_thrust_econ_d_thrust_to[0] = 1.0
        partials["thrust_econ", "data:mission:sizing:taxi_out:thrust"] = d_thrust_econ_d_thrust_to

        d_thrust_econ_d_thrust_ti = np.zeros(number_of_points + 2)
        d_thrust_econ_d_thrust_ti[-1] = 1.0
        partials["thrust_econ", "data:mission:sizing:taxi_in:thrust"] = d_thrust_econ_d_thrust_ti

        d_altitude_econ_d_altitude = np.zeros((number_of_points + 2, number_of_points))
        d_altitude_econ_d_altitude[1 : number_of_points + 1, :] = np.eye(number_of_points)
        partials["altitude_econ", "altitude"] = d_altitude_econ_d_altitude

        # In value it is gonna be the same so we avoid over-burdening the compute_partials function
        partials["density_econ", "density"] = d_altitude_econ_d_altitude
        partials["engine_setting_econ", "engine_setting"] = d_altitude_econ_d_altitude

        d_temp_econ_d_temp = np.zeros((number_of_points + 2, number_of_points))
        d_temp_econ_d_temp[1 : number_of_points + 1, :] = np.eye(number_of_points)
        partials["exterior_temperature_econ", "exterior_temperature"] = d_temp_econ_d_temp

        d_ts_econ_d_ts = np.zeros((number_of_points + 2, number_of_points))
        d_ts_econ_d_ts[1 : number_of_points + 1, :] = np.eye(number_of_points)
        partials["time_step_econ", "time_step"] = d_ts_econ_d_ts

        d_ts_econ_d_ts_to = np.zeros(number_of_points + 2)
        d_ts_econ_d_ts_to[0] = 1.0
        partials["time_step_econ", "data:mission:sizing:taxi_out:duration"] = d_ts_econ_d_ts_to

        d_ts_econ_d_ts_ti = np.zeros(number_of_points + 2)
        d_ts_econ_d_ts_ti[-1] = 1.0
        partials["time_step_econ", "data:mission:sizing:taxi_in:duration"] = d_ts_econ_d_ts_ti

        d_tas_econ_d_tas = np.zeros((number_of_points + 2, number_of_points))
        d_tas_econ_d_tas[1 : number_of_points + 1, :] = np.eye(number_of_points)
        partials["true_airspeed_econ", "true_airspeed"] = d_tas_econ_d_tas

        d_tas_econ_d_tas_to = np.zeros(number_of_points + 2)
        d_tas_econ_d_tas_to[0] = 1.0
        partials["true_airspeed_econ", "data:mission:sizing:taxi_out:speed"] = d_tas_econ_d_tas_to

        d_tas_econ_d_tas_ti = np.zeros(number_of_points + 2)
        d_tas_econ_d_tas_ti[-1] = 1.0
        partials["true_airspeed_econ", "data:mission:sizing:taxi_in:speed"] = d_tas_econ_d_tas_ti
