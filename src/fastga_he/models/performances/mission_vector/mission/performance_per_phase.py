# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import numpy as np
import openmdao.api as om


class PerformancePerPhase(om.ExplicitComponent):
    """
    Computes the fuel consumed time spent and ground distance travelled for each phase to
    match the outputs of the previous performance module.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points_climb", default=1, desc="number of equilibrium to be treated in climb"
        )
        self.options.declare(
            "number_of_points_cruise",
            default=1,
            desc="number of equilibrium to be treated in cruise",
        )
        self.options.declare(
            "number_of_points_descent",
            default=1,
            desc="number of equilibrium to be treated in descent",
        )
        self.options.declare(
            "number_of_points_reserve",
            default=1,
            desc="number of equilibrium to be treated in reserve",
        )

    def setup(self):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        self.add_input(
            "time", shape=number_of_points, val=np.full(number_of_points, np.nan), units="s"
        )
        self.add_input(
            "position", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "fuel_consumed_t_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="kg",
        )
        self.add_input(
            "fuel_mass_t_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="kg",
        )
        self.add_input(
            "fuel_lever_arm_t_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="kg*m",
        )
        self.add_input(
            "non_consumable_energy_t_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
            units="W*h",
        )
        self.add_input(
            "thrust_rate_t_econ",
            shape=number_of_points + 2,
            val=np.full(number_of_points + 2, np.nan),
        )

        self.add_output("data:mission:sizing:main_route:climb:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:climb:energy", units="W*h")
        self.add_output("data:mission:sizing:main_route:climb:distance", units="m")
        self.add_output("data:mission:sizing:main_route:climb:duration", units="s")

        self.add_output("data:mission:sizing:main_route:cruise:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:cruise:energy", units="W*h")
        self.add_output("data:mission:sizing:main_route:cruise:distance", units="m")
        self.add_output("data:mission:sizing:main_route:cruise:duration", units="s")

        self.add_output("data:mission:sizing:main_route:descent:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:descent:energy", units="W*h")
        self.add_output("data:mission:sizing:main_route:descent:distance", units="m")
        self.add_output("data:mission:sizing:main_route:descent:duration", units="s")

        self.add_output("data:mission:sizing:taxi_out:fuel", units="kg")
        self.add_output("data:mission:sizing:taxi_out:energy", units="W*h")
        self.add_output("data:mission:sizing:taxi_in:fuel", units="kg")
        self.add_output("data:mission:sizing:taxi_in:energy", units="W*h")

        self.add_output("data:mission:sizing:main_route:reserve:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:reserve:energy", units="W*h")

        self.add_output("fuel_consumed_t", shape=number_of_points, units="kg")
        self.add_output("fuel_mass_t", shape=number_of_points, units="kg")
        self.add_output("fuel_lever_arm_t", shape=number_of_points, units="kg*m")
        self.add_output("non_consumable_energy_t", shape=number_of_points, units="W*h")
        self.add_output("thrust_rate_t", shape=number_of_points)

    def setup_partials(self):

        self.declare_partials(
            of="data:mission:sizing:main_route:climb:fuel",
            wrt="fuel_consumed_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:cruise:fuel",
            wrt="fuel_consumed_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:descent:fuel",
            wrt="fuel_consumed_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_out:fuel", wrt="fuel_consumed_t_econ", method="exact"
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_in:fuel", wrt="fuel_consumed_t_econ", method="exact"
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:reserve:fuel",
            wrt="fuel_consumed_t_econ",
            method="exact",
        )
        self.declare_partials(of="fuel_consumed_t", wrt="fuel_consumed_t_econ", method="exact")

        self.declare_partials(of="fuel_mass_t", wrt="fuel_mass_t_econ", method="exact")

        self.declare_partials(of="fuel_lever_arm_t", wrt="fuel_lever_arm_t_econ", method="exact")

        self.declare_partials(
            of="data:mission:sizing:main_route:climb:energy",
            wrt="non_consumable_energy_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:cruise:energy",
            wrt="non_consumable_energy_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:descent:energy",
            wrt="non_consumable_energy_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_out:energy",
            wrt="non_consumable_energy_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:taxi_in:energy",
            wrt="non_consumable_energy_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:reserve:energy",
            wrt="non_consumable_energy_t_econ",
            method="exact",
        )
        self.declare_partials(
            of="non_consumable_energy_t", wrt="non_consumable_energy_t_econ", method="exact"
        )

        self.declare_partials(of="thrust_rate_t", wrt="thrust_rate_t_econ", method="exact")

        self.declare_partials(
            of="data:mission:sizing:main_route:climb:distance", wrt="position", method="exact"
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:cruise:distance", wrt="position", method="exact"
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:descent:distance", wrt="position", method="exact"
        )

        self.declare_partials(
            of="data:mission:sizing:main_route:climb:duration", wrt="time", method="exact"
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:cruise:duration", wrt="time", method="exact"
        )
        self.declare_partials(
            of="data:mission:sizing:main_route:descent:duration", wrt="time", method="exact"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        time = inputs["time"]
        position = inputs["position"]
        # This one is two element longer than the other array since it includes the fuel consumed
        # for the taxi phases, hence why we stop at -2 for the descent fuel consumption
        fuel_consumed_t_econ = inputs["fuel_consumed_t_econ"]
        fuel_mass_t_econ = inputs["fuel_mass_t_econ"]
        fuel_lever_arm_t_econ = inputs["fuel_lever_arm_t_econ"]
        non_consumable_energy = inputs["non_consumable_energy_t_econ"]
        thrust_rate_t_econ = inputs["thrust_rate_t_econ"]

        outputs["data:mission:sizing:main_route:climb:fuel"] = np.sum(
            fuel_consumed_t_econ[1 : number_of_points_climb + 1]
        )
        outputs["data:mission:sizing:main_route:climb:energy"] = np.sum(
            non_consumable_energy[1 : number_of_points_climb + 1]
        )
        outputs["data:mission:sizing:main_route:climb:distance"] = position[number_of_points_climb]
        outputs["data:mission:sizing:main_route:climb:duration"] = time[number_of_points_climb]

        outputs["data:mission:sizing:main_route:cruise:fuel"] = np.sum(
            fuel_consumed_t_econ[
                number_of_points_climb + 1 : number_of_points_climb + number_of_points_cruise + 1
            ]
        )
        outputs["data:mission:sizing:main_route:cruise:energy"] = np.sum(
            non_consumable_energy[
                number_of_points_climb + 1 : number_of_points_climb + number_of_points_cruise + 1
            ]
        )
        outputs["data:mission:sizing:main_route:cruise:distance"] = (
            position[number_of_points_climb + number_of_points_cruise]
            - position[number_of_points_climb]
        )
        outputs["data:mission:sizing:main_route:cruise:duration"] = (
            time[number_of_points_climb + number_of_points_cruise] - time[number_of_points_climb]
        )

        outputs["data:mission:sizing:main_route:descent:fuel"] = np.sum(
            fuel_consumed_t_econ[
                number_of_points_climb
                + number_of_points_cruise
                + 1 : number_of_points_climb
                + number_of_points_cruise
                + number_of_points_descent
                + 1
            ]
        )
        outputs["data:mission:sizing:main_route:descent:energy"] = np.sum(
            non_consumable_energy[
                number_of_points_climb
                + number_of_points_cruise
                + 1 : number_of_points_climb
                + number_of_points_cruise
                + number_of_points_descent
                + 1
            ]
        )
        outputs["data:mission:sizing:main_route:descent:distance"] = (
            position[number_of_points_climb + number_of_points_cruise + number_of_points_descent]
            - position[number_of_points_climb + number_of_points_cruise]
        )

        outputs["data:mission:sizing:main_route:descent:duration"] = (
            time[number_of_points_climb + number_of_points_cruise + number_of_points_descent]
            - time[number_of_points_climb + number_of_points_cruise]
        )

        outputs["data:mission:sizing:main_route:reserve:fuel"] = np.sum(
            fuel_consumed_t_econ[
                number_of_points_climb
                + number_of_points_cruise
                + number_of_points_descent
                + 1 : number_of_points_climb
                + number_of_points_cruise
                + number_of_points_descent
                + number_of_points_reserve
                + 1
            ]
        )
        outputs["data:mission:sizing:main_route:reserve:energy"] = np.sum(
            non_consumable_energy[
                number_of_points_climb
                + number_of_points_cruise
                + number_of_points_descent
                + 1 : number_of_points_climb
                + number_of_points_cruise
                + number_of_points_descent
                + number_of_points_reserve
                + 1
            ]
        )

        outputs["data:mission:sizing:taxi_out:fuel"] = fuel_consumed_t_econ[0]
        outputs["data:mission:sizing:taxi_out:energy"] = non_consumable_energy[0]
        outputs["data:mission:sizing:taxi_in:fuel"] = fuel_consumed_t_econ[-1]
        outputs["data:mission:sizing:taxi_in:energy"] = non_consumable_energy[-1]

        outputs["fuel_consumed_t"] = fuel_consumed_t_econ[1:-1]
        outputs["fuel_mass_t"] = fuel_mass_t_econ[1:-1]
        outputs["fuel_lever_arm_t"] = fuel_lever_arm_t_econ[1:-1]
        outputs["non_consumable_energy_t"] = non_consumable_energy[1:-1]
        outputs["thrust_rate_t"] = thrust_rate_t_econ[1:-1]

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        partials[
            "data:mission:sizing:main_route:climb:fuel", "fuel_consumed_t_econ"
        ] = np.concatenate(
            (
                np.zeros(1),
                np.ones(number_of_points_climb),
                np.zeros(
                    number_of_points_cruise
                    + number_of_points_descent
                    + number_of_points_reserve
                    + 1
                ),
            )
        )
        partials[
            "data:mission:sizing:main_route:climb:energy", "non_consumable_energy_t_econ"
        ] = np.concatenate(
            (
                np.zeros(1),
                np.ones(number_of_points_climb),
                np.zeros(
                    number_of_points_cruise
                    + number_of_points_descent
                    + number_of_points_reserve
                    + 1
                ),
            )
        )

        partials[
            "data:mission:sizing:main_route:cruise:fuel", "fuel_consumed_t_econ"
        ] = np.concatenate(
            (
                np.zeros(number_of_points_climb + 1),
                np.ones(number_of_points_cruise),
                np.zeros(number_of_points_descent + number_of_points_reserve + 1),
            )
        )
        partials[
            "data:mission:sizing:main_route:cruise:energy", "non_consumable_energy_t_econ"
        ] = np.concatenate(
            (
                np.zeros(number_of_points_climb + 1),
                np.ones(number_of_points_cruise),
                np.zeros(number_of_points_descent + number_of_points_reserve + 1),
            )
        )

        partials[
            "data:mission:sizing:main_route:descent:fuel", "fuel_consumed_t_econ"
        ] = np.concatenate(
            (
                np.zeros(number_of_points_climb + number_of_points_cruise + 1),
                np.ones(number_of_points_descent),
                np.zeros(number_of_points_reserve + 1),
            )
        )
        partials[
            "data:mission:sizing:main_route:descent:energy", "non_consumable_energy_t_econ"
        ] = np.concatenate(
            (
                np.zeros(number_of_points_climb + number_of_points_cruise + 1),
                np.ones(number_of_points_descent),
                np.zeros(number_of_points_reserve + 1),
            )
        )

        partials[
            "data:mission:sizing:main_route:reserve:fuel", "fuel_consumed_t_econ"
        ] = np.concatenate(
            (
                np.zeros(
                    number_of_points_climb + number_of_points_cruise + number_of_points_descent + 1
                ),
                np.full(number_of_points_reserve, 1.0),
                np.zeros(1),
            )
        )
        partials[
            "data:mission:sizing:main_route:reserve:energy", "non_consumable_energy_t_econ"
        ] = np.concatenate(
            (
                np.zeros(
                    number_of_points_climb + number_of_points_cruise + number_of_points_descent + 1
                ),
                np.full(number_of_points_reserve, 1.0),
                np.zeros(1),
            )
        )

        d_taxi_out_d_fuel = np.zeros(number_of_points + 2)
        d_taxi_out_d_fuel[0] = 1
        partials["data:mission:sizing:taxi_out:fuel", "fuel_consumed_t_econ"] = d_taxi_out_d_fuel
        partials[
            "data:mission:sizing:taxi_out:energy", "non_consumable_energy_t_econ"
        ] = d_taxi_out_d_fuel

        d_taxi_in_d_fuel = np.zeros(number_of_points + 2)
        d_taxi_in_d_fuel[-1] = 1
        partials["data:mission:sizing:taxi_in:fuel", "fuel_consumed_t_econ"] = d_taxi_in_d_fuel
        partials[
            "data:mission:sizing:taxi_in:energy", "non_consumable_energy_t_econ"
        ] = d_taxi_in_d_fuel

        d_fc_d_fc_t = np.zeros((number_of_points, number_of_points + 2))
        d_fc_d_fc_t[:, 1 : number_of_points + 1] = np.eye(number_of_points)
        partials["fuel_consumed_t", "fuel_consumed_t_econ"] = d_fc_d_fc_t
        partials["non_consumable_energy_t", "non_consumable_energy_t_econ"] = d_fc_d_fc_t
        partials["fuel_mass_t", "fuel_mass_t_econ"] = d_fc_d_fc_t
        partials["fuel_lever_arm_t", "fuel_lever_arm_t_econ"] = d_fc_d_fc_t

        d_tr_d_tr_t = np.zeros((number_of_points, number_of_points + 2))
        d_tr_d_tr_t[:, 1 : number_of_points + 1] = np.eye(number_of_points)
        partials["thrust_rate_t", "thrust_rate_t_econ"] = d_tr_d_tr_t

        d_climb_d_d_pos = np.zeros(number_of_points)
        d_climb_d_d_pos[number_of_points_climb] = 1.0
        partials["data:mission:sizing:main_route:climb:distance", "position"] = d_climb_d_d_pos

        d_cruise_d_d_pos = np.zeros(number_of_points)
        d_cruise_d_d_pos[number_of_points_climb + number_of_points_cruise] = 1.0
        d_cruise_d_d_pos[number_of_points_climb] = -1.0
        partials["data:mission:sizing:main_route:cruise:distance", "position"] = d_cruise_d_d_pos

        d_descent_d_d_pos = np.zeros(number_of_points)
        d_descent_d_d_pos[
            number_of_points_climb + number_of_points_cruise + number_of_points_descent
        ] = 1.0
        d_descent_d_d_pos[number_of_points_climb + number_of_points_cruise] = -1.0
        partials["data:mission:sizing:main_route:descent:distance", "position"] = d_descent_d_d_pos

        partials["data:mission:sizing:main_route:climb:duration", "time"] = d_climb_d_d_pos

        partials["data:mission:sizing:main_route:cruise:duration", "time"] = d_cruise_d_d_pos

        partials["data:mission:sizing:main_route:descent:duration", "time"] = d_descent_d_d_pos
