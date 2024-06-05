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

        t_econ_to_t = np.zeros((number_of_points + 2, number_of_points))
        t_econ_to_t[1 : number_of_points + 1, :] = np.eye(number_of_points)

        self.declare_partials(
            of="thrust_econ",
            wrt="thrust",
            method="exact",
            rows=np.where(t_econ_to_t != 0)[0],
            cols=np.where(t_econ_to_t != 0)[1],
        )
        self.declare_partials(
            of="thrust_econ",
            wrt="data:mission:sizing:taxi_out:thrust",
            method="exact",
            rows=np.array([0]),
            cols=np.array([0]),
            val=1.0,
        )
        self.declare_partials(
            of="thrust_econ",
            wrt="data:mission:sizing:taxi_in:thrust",
            method="exact",
            rows=np.array([number_of_points + 1]),
            cols=np.array([0]),
            val=1.0,
        )

        self.declare_partials(
            of="altitude_econ",
            wrt="altitude",
            method="exact",
            rows=np.where(t_econ_to_t != 0)[0],
            cols=np.where(t_econ_to_t != 0)[1],
            val=np.ones(len(np.where(t_econ_to_t != 0)[0])),
        )
        self.declare_partials(
            of="density_econ",
            wrt="density",
            method="exact",
            rows=np.where(t_econ_to_t != 0)[0],
            cols=np.where(t_econ_to_t != 0)[1],
            val=np.ones(len(np.where(t_econ_to_t != 0)[0])),
        )
        self.declare_partials(
            of="engine_setting_econ",
            wrt="engine_setting",
            method="exact",
            rows=np.where(t_econ_to_t != 0)[0],
            cols=np.where(t_econ_to_t != 0)[1],
            val=np.ones(len(np.where(t_econ_to_t != 0)[0])),
        )

        self.declare_partials(
            of="exterior_temperature_econ",
            wrt="exterior_temperature",
            method="exact",
            rows=np.where(t_econ_to_t != 0)[0],
            cols=np.where(t_econ_to_t != 0)[1],
            val=np.ones(len(np.where(t_econ_to_t != 0)[0])),
        )

        self.declare_partials(
            of="time_step_econ",
            wrt="time_step",
            method="exact",
            rows=np.where(t_econ_to_t != 0)[0],
            cols=np.where(t_econ_to_t != 0)[1],
            val=np.ones(len(np.where(t_econ_to_t != 0)[0])),
        )
        self.declare_partials(
            of="time_step_econ",
            wrt="data:mission:sizing:taxi_out:duration",
            method="exact",
            rows=np.array([0]),
            cols=np.array([0]),
            val=1.0,
        )
        self.declare_partials(
            of="time_step_econ",
            wrt="data:mission:sizing:taxi_in:duration",
            method="exact",
            rows=np.array([number_of_points + 1]),
            cols=np.array([0]),
            val=1.0,
        )

        self.declare_partials(
            of="true_airspeed_econ",
            wrt="true_airspeed",
            method="exact",
            rows=np.where(t_econ_to_t != 0)[0],
            cols=np.where(t_econ_to_t != 0)[1],
            val=np.ones(len(np.where(t_econ_to_t != 0)[0])),
        )
        self.declare_partials(
            of="true_airspeed_econ",
            wrt="data:mission:sizing:taxi_out:speed",
            method="exact",
            rows=np.array([0]),
            cols=np.array([0]),
            val=1.0,
        )
        self.declare_partials(
            of="true_airspeed_econ",
            wrt="data:mission:sizing:taxi_in:speed",
            method="exact",
            rows=np.array([number_of_points + 1]),
            cols=np.array([0]),
            val=1.0,
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

        d_thrust_econ_d_thrust_diagonal = np.where(
            inputs["thrust"] > 50.0, np.ones_like(inputs["thrust"]), np.zeros_like(inputs["thrust"])
        )
        partials["thrust_econ", "thrust"] = d_thrust_econ_d_thrust_diagonal
