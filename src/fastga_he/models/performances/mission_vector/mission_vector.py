# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import logging

import numpy as np
import openmdao.api as om

from scipy.constants import g

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

from stdatm import Atmosphere
from typing import Tuple

from .initialization.initialize import Initialize
from .mission.mission_core import MissionCore
from .to_csv import ToCSV
from fastga_he.models.weight.cg.cg_variation import InFlightCGVariation

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator
from fastga_he.models.propulsion.assemblers.performances_watcher import (
    PowerTrainPerformancesWatcher,
)

from fastga_he.models.performances.mission_vector.constants import (
    HE_SUBMODEL_ENERGY_CONSUMPTION,
    HE_SUBMODEL_DEP_EFFECT,
)
from fastga_he.models.propulsion.assemblers.energy_consumption_mission_vector import (
    ENERGY_CONSUMPTION_FROM_PT_FILE,
)
from fastga_he.models.propulsion.assemblers.delta_from_pt_file import DEP_EFFECT_FROM_PT_FILE

from fastga_he.models.performances.mission_vector.mission.thrust_taxi import MIN_POWER_TAXI

_LOGGER = logging.getLogger(__name__)

DENSITY_SL = Atmosphere(0.0).density
DUMMY_FUEL_CONSUMED = 200.0
DUMMY_ENERGY_CONSUMED = 200e3


@oad.RegisterOpenMDAOSystem("fastga_he.performances.mission_vector", domain=ModelDomain.OTHER)
class MissionVector(om.Group):
    """Computes and potentially save mission based on options."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 100
        self.nonlinear_solver.options["rtol"] = 1e-5
        self.linear_solver = om.LinearBlockGS()

        self.configurator = FASTGAHEPowerTrainConfigurator()

    def initialize(self):

        self.options.declare("out_file", default="", types=str)
        self.options.declare(
            "number_of_points_climb",
            default=100,
            desc="number of equilibrium to be treated in climb",
        )
        self.options.declare(
            "number_of_points_cruise",
            default=100,
            desc="number of equilibrium to be treated in cruise",
        )
        self.options.declare(
            "number_of_points_descent",
            default=50,
            desc="number of equilibrium to be treated in descent",
        )
        self.options.declare(
            "number_of_points_reserve",
            default=1,
            desc="number of equilibrium to be treated in reserve",
        )
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare(
            name="power_train_file_path",
            default="",
            desc="Path to the file containing the description of the power",
        )
        self.options.declare(
            "use_linesearch",
            default=True,
            types=bool,
            desc="boolean to turn off the use of a linesearch algorithm during the mission."
            "Can be turned off to speed up the process but might not converge.",
        )
        self.options.declare(
            name="pre_condition_voltage",
            default=False,
            desc="Boolean to pre_condition the voltages of the different components of the PT, "
            "can save some time in specific cases",
            allow_none=False,
        )

    def setup(self):

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        pt_file_path = self.options["power_train_file_path"]

        self.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
        self.add_subsystem(
            "initialization",
            Initialize(
                number_of_points_climb=number_of_points_climb,
                number_of_points_cruise=number_of_points_cruise,
                number_of_points_descent=number_of_points_descent,
                number_of_points_reserve=number_of_points_reserve,
            ),
            promotes_inputs=["data:*", "settings:*"],
            promotes_outputs=[],
        )
        self.add_subsystem(
            "solve_equilibrium",
            MissionCore(
                number_of_points_climb=number_of_points_climb,
                number_of_points_cruise=number_of_points_cruise,
                number_of_points_descent=number_of_points_descent,
                number_of_points_reserve=number_of_points_reserve,
                propulsion_id=self.options["propulsion_id"],
                power_train_file_path=self.options["power_train_file_path"],
                use_linesearch=self.options["use_linesearch"],
                pre_condition_voltage=self.options["pre_condition_voltage"],
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "to_csv",
            ToCSV(
                number_of_points_climb=number_of_points_climb,
                number_of_points_cruise=number_of_points_cruise,
                number_of_points_descent=number_of_points_descent,
                number_of_points_reserve=number_of_points_reserve,
                out_file=self.options["out_file"],
            ),
            promotes_inputs=["data:*"],
            promotes_outputs=[],
        )

        self.connect(
            "initialization.initialize_engine_setting.engine_setting",
            [
                "solve_equilibrium.compute_dep_equilibrium.engine_setting",
                "to_csv.engine_setting",
            ],
        )

        self.connect(
            "initialization.initialize_temperature.exterior_temperature",
            [
                "solve_equilibrium.compute_dep_equilibrium.exterior_temperature",
                "to_csv.exterior_temperature",
            ],
        )

        self.connect(
            "initialization.initialize_center_of_gravity.x_cg",
            [
                "solve_equilibrium.compute_dep_equilibrium.x_cg",
                "to_csv.x_cg",
            ],
        )
        self.connect(
            "initialization.initialize_density.density",
            [
                "solve_equilibrium.compute_dep_equilibrium.density",
            ],
        )

        self.connect(
            "initialization.initialize_time_and_distance.position",
            ["solve_equilibrium.performance_per_phase.position", "to_csv.position"],
        )

        self.connect(
            "initialization.initialize_time_and_distance.time",
            [
                "solve_equilibrium.performance_per_phase.time",
                "to_csv.time",
            ],
        )

        self.connect(
            "initialization.initialize_time_step.time_step",
            ["to_csv.time_step", "solve_equilibrium.time_step"],
        )

        self.connect(
            "initialization.initialize_airspeed_time_derivatives.d_vx_dt",
            [
                "solve_equilibrium.compute_dep_equilibrium.d_vx_dt",
                "to_csv.d_vx_dt",
            ],
        )

        self.connect(
            "initialization.initialize_airspeed.true_airspeed",
            [
                "solve_equilibrium.compute_dep_equilibrium.true_airspeed",
                "to_csv.true_airspeed",
            ],
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.delta_Cl",
            "to_csv.delta_Cl",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.delta_Cd",
            "to_csv.delta_Cd",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.delta_Cm",
            "to_csv.delta_Cm",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.alpha",
            "to_csv.alpha",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.thrust",
            "to_csv.thrust",
        )

        self.connect(
            "solve_equilibrium.compute_dep_equilibrium.delta_m",
            "to_csv.delta_m",
        )

        self.connect(
            "initialization.initialize_airspeed.equivalent_airspeed", "to_csv.equivalent_airspeed"
        )

        self.connect(
            "solve_equilibrium.mass",
            [
                "to_csv.mass",
                "initialization.mass",
            ],
        )

        self.connect(
            "solve_equilibrium.fuel_consumed_t",
            [
                "to_csv.fuel_consumed_t",
                "initialization.initialize_center_of_gravity.fuel_consumed_t",
            ],
        )

        self.connect(
            "solve_equilibrium.performance_per_phase.non_consumable_energy_t",
            [
                "to_csv.non_consumable_energy_t",
            ],
        )

        self.connect(
            "solve_equilibrium.performance_per_phase.thrust_rate_t", "to_csv.thrust_rate_t"
        )

        self.connect(
            "initialization.initialize_gamma.gamma",
            [
                "to_csv.gamma",
                "solve_equilibrium.compute_dep_equilibrium.gamma",
            ],
        )

        self.connect(
            "initialization.altitude",
            [
                "to_csv.altitude",
                "solve_equilibrium.compute_dep_equilibrium.altitude",
            ],
        )

        # Add the powertrain watcher here to avoid opening and closing csv all the time. We will
        # add here a check to ensure that the module that computes the performances base on the
        # powertrain builder is used. Because if it is not used, no pt file will be provided
        # meaning the load instruction of the configurator will fail and so will the connects.

        if (
            oad.RegisterSubmodel.active_models[HE_SUBMODEL_ENERGY_CONSUMPTION]
            == ENERGY_CONSUMPTION_FROM_PT_FILE
        ):

            self.configurator.load(pt_file_path)

            if self.configurator.get_watcher_file_path():

                number_of_points = (
                    1
                    + number_of_points_climb
                    + number_of_points_cruise
                    + number_of_points_descent
                    + number_of_points_reserve
                    + 1
                )
                self.add_subsystem(
                    "performances_watcher",
                    PowerTrainPerformancesWatcher(
                        power_train_file_path=self.options["power_train_file_path"],
                        number_of_points=number_of_points,
                    ),
                )

                (
                    components_name,
                    components_performances_watchers_names,
                    _,
                ) = self.configurator.get_performance_watcher_elements_list()

                for (component_name, component_performances_watcher_name) in zip(
                    components_name, components_performances_watchers_names
                ):

                    self.connect(
                        "solve_equilibrium.compute_dep_equilibrium.compute_energy_consumed.power_train_performances."
                        + component_name
                        + "."
                        + component_performances_watcher_name,
                        "performances_watcher"
                        + "."
                        + component_name
                        + "_"
                        + component_performances_watcher_name,
                    )

                # This is starting to become confusing: The next bit should only be executed if
                # the right model for the computation of slipstream effect is used. BUT we want
                # to be able to "turn off" the slipstream effects and still register results in
                # the powertrain watcher ...

                if (
                    oad.RegisterSubmodel.active_models[HE_SUBMODEL_DEP_EFFECT]
                    == DEP_EFFECT_FROM_PT_FILE
                ):

                    (
                        components_slip_name,
                        components_slip_performances_watchers_names,
                        _,
                    ) = self.configurator.get_slipstream_performance_watcher_elements_list()

                    for (component_slip_name, component_slip_performances_watcher_name) in zip(
                        components_slip_name, components_slip_performances_watchers_names
                    ):
                        self.connect(
                            "solve_equilibrium.compute_dep_equilibrium.compute_dep_effect."
                            + component_slip_name
                            + "."
                            + component_slip_performances_watcher_name,
                            "performances_watcher"
                            + "."
                            + component_slip_name
                            + "_"
                            + component_slip_performances_watcher_name,
                        )

                self.connect(
                    "solve_equilibrium.compute_dep_equilibrium.thrust_econ",
                    "performances_watcher.thrust",
                )
                self.connect(
                    "solve_equilibrium.compute_dep_equilibrium.altitude_econ",
                    "performances_watcher.altitude",
                )
                self.connect(
                    "solve_equilibrium.compute_dep_equilibrium.time_step_econ",
                    "performances_watcher.time_step",
                )
                self.connect(
                    "solve_equilibrium.compute_dep_equilibrium.true_airspeed_econ",
                    "performances_watcher.true_airspeed",
                )
                self.connect(
                    "solve_equilibrium.compute_dep_equilibrium.exterior_temperature_econ",
                    "performances_watcher.exterior_temperature",
                )

        else:

            _LOGGER.warning(
                "Power train builder is not used for the performances computation. If "
                "this was intended, you can ignore this warning. Else, make sure to select the "
                + ENERGY_CONSUMPTION_FROM_PT_FILE
                + " submodel for the "
                + HE_SUBMODEL_ENERGY_CONSUMPTION
                + " service"
            )

    def guess_nonlinear(
        self, inputs, outputs, residuals, discrete_inputs=None, discrete_outputs=None
    ):
        """
        Against my better judgement, I will do the guess_nonlinear at the top most component
        because for some reasons the guess nonlinear function are first executed from the
        bottom-most component to the topmost component. Additionally, the variables I'd need (
        MTOW to get thrust through assumed finesse, cruise TAS for speed) are inputs of component
        only in the initialization and high level mission component. Consequently we will do
        everything here. But I'm not happy about it.
        """

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]
        number_of_points_total = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        ###########################################################################################
        # MISSION INITIAL GUESS, RAN REGARDLESS OF WHETHER WE USE THE PT FILE OR NOT ##############
        ###########################################################################################

        # For the initialization of the fuel consumed we can be smart and set it at 0.0 if we
        # only have electric components

        self.set_initial_guess_mass(outputs=outputs, inputs=inputs)
        self.set_initial_guess_energies(outputs=outputs)
        self.set_initial_guess_thrust(outputs=outputs, inputs=inputs)
        self.set_initial_guess_alpha(outputs=outputs)
        self.set_initial_guess_delta_m(outputs=outputs)
        self.set_initial_guess_speed(inputs=inputs, outputs=outputs)
        self.set_initial_guess_altitude(inputs=inputs, outputs=outputs)
        self.set_initial_guess_density(outputs=outputs)
        self.set_initial_guess_temperature(outputs=outputs)
        self.set_initial_guess_taxi_thrust(inputs=inputs, outputs=outputs)
        self.set_initial_guess_speed_econ(inputs=inputs, outputs=outputs)
        self.set_initial_guess_thrust_econ(outputs=outputs)

        ###########################################################################################
        # PT FILE INITIAL GUESS ###################################################################
        ###########################################################################################

        # This one will be passed in before going into the first pt components

        # Only trigger if we actually have a pt file, don't check for proper submodels because.

        # Let's first check the coherence of the voltage. Have to have the +2 because contrarily
        # as where it was before where the taxi phases were included here, it is not
        if self.options["power_train_file_path"]:
            self.configurator.check_voltage_coherence(
                inputs=inputs, number_of_points=number_of_points_total + 2
            )

        # Only trigger if the options is used, if we actually have a pt file and if the right
        # submodels are used

        if self.options["pre_condition_voltage"] and self.options["power_train_file_path"]:

            # Then we check that there is indeed a powertrain and that the right submodels are used

            voltage_to_set = self.configurator.get_voltage_to_set(
                inputs=inputs, number_of_points=number_of_points_total + 2
            )

            # First we pre-condition voltage
            for sub_graphs in voltage_to_set:
                for voltage in sub_graphs:
                    output_name = (
                        "solve_equilibrium.compute_dep_equilibrium.compute_energy_consumed.power_train_performances."
                        + voltage
                    )
                    outputs[output_name] = sub_graphs[voltage]

            # Then we compute the propulsive power required that each propulsor has to produce.
            # We need the true airspeed, and since all propulsor will likely need it and they'll
            # be the same, we'll take the tas from the first propulsor in the list.
            propulsor_names = self.configurator.get_thrust_element_list()

            propulsive_power_dict = get_propulsive_power(
                propulsor_names,
                inputs[
                    "solve_equilibrium.compute_dep_equilibrium.compute_energy_consumed.power_train_performances.thrust_splitter.data:propulsion:he_power_train:thrust_distribution"
                ],
                outputs[
                    "solve_equilibrium.compute_dep_equilibrium.preparation_for_energy_consumption.thrust_econ"
                ],
                outputs[
                    "solve_equilibrium.compute_dep_equilibrium.preparation_for_energy_consumption.true_airspeed_econ"
                ],
            )

            # So that we can set the power
            power_to_set = self.configurator.get_power_to_set(inputs, propulsive_power_dict)[1]

            for sub_graphs in power_to_set:
                for power in sub_graphs:
                    output_name = (
                        "solve_equilibrium.compute_dep_equilibrium.compute_energy_consumed.power_train_performances."
                        + power
                    )
            outputs[output_name] = sub_graphs[power]

    def _get_initial_guess_fuel_consumed(self) -> np.ndarray:
        """
        Provides an educated guess of the variation of fuel consumed during the flight. It is a
        mere initial guess, the end results will still be accurate. Does not set it.
        """

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        # The following fuel repartition will be adopted for now, 15% for climb, 50% for cruise,
        # 5% for descent, 30% for reserve
        fuel_climb = np.full(
            number_of_points_climb, 0.15 * DUMMY_FUEL_CONSUMED / number_of_points_climb
        )
        fuel_cruise = np.full(
            number_of_points_cruise, 0.50 * DUMMY_FUEL_CONSUMED / number_of_points_cruise
        )
        fuel_descent = np.full(
            number_of_points_descent, 0.15 * DUMMY_FUEL_CONSUMED / number_of_points_descent
        )
        fuel_reserve = np.full(
            number_of_points_reserve, 0.20 * DUMMY_FUEL_CONSUMED / number_of_points_reserve
        )

        dummy_fuel_consumed = np.concatenate((fuel_climb, fuel_cruise, fuel_descent, fuel_reserve))

        return dummy_fuel_consumed

    def set_initial_guess_mass(self, inputs, outputs):
        """
        Provides and sets educated guess of the variation of mass during the flight. It is a mere
        initial guess, the end results will still be accurate.

        :param inputs: OpenMDAO vector containing the value of inputs
        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        mtow = inputs["data:weight:aircraft:MTOW"]

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        number_of_points_total = (
            number_of_points_climb
            + number_of_points_cruise
            + number_of_points_descent
            + number_of_points_reserve
        )

        dummy_fuel_consumed = self._get_initial_guess_fuel_consumed()

        if self.options["power_train_file_path"]:
            if self.configurator.will_aircraft_mass_vary():

                outputs["solve_equilibrium.update_mass.mass"] = np.full(
                    number_of_points_total, mtow
                ) - np.cumsum(dummy_fuel_consumed)

            else:
                outputs["solve_equilibrium.update_mass.mass"] = np.full(
                    number_of_points_total, mtow
                )
        else:

            outputs["solve_equilibrium.update_mass.mass"] = np.full(
                number_of_points_total, mtow
            ) - np.cumsum(dummy_fuel_consumed)

    def _get_initial_guess_thrust(self, mtow) -> np.ndarray:
        """
        Provides an educated guess of the thrust required during the flight. It is a mere initial
        guess, the end results will still be accurate. Does not set it. Assumes a lift to drag
        ratio of 13 during cruise and reserve.

        :param mtow: mtow of the aircraft at that iteration of the sizing process
        """

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        thrust_climb = np.full(number_of_points_climb, mtow * 2.0)
        thrust_cruise = np.full(number_of_points_cruise, mtow / 1.3)
        thrust_descent = np.full(number_of_points_descent, mtow * 0.5)
        thrust_reserve = np.full(number_of_points_reserve, mtow / 1.3)

        dummy_thrust = np.concatenate((thrust_climb, thrust_cruise, thrust_descent, thrust_reserve))

        return dummy_thrust

    def set_initial_guess_thrust(self, inputs, outputs):
        """
        Provides and sets educated guess of the  thrust required during the flight. It is a mere
        initial guess, the end results will still be accurate.

        :param inputs: OpenMDAO vector containing the value of inputs
        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        mtow = inputs["data:weight:aircraft:MTOW"]

        dummy_thrust = self._get_initial_guess_thrust(mtow=mtow)

        outputs[
            "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium_thrust.thrust"
        ] = dummy_thrust

    def _get_initial_guess_alpha(self) -> np.ndarray:
        """
        Provides an educated guess of the AoA required during the flight. It is a mere initial
        guess, the end results will still be accurate. Does not set it. Could be improved based
        on the weight, wing area and wing aerodynamics with a simple lift equation.
        """

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        aoa_climb = np.full(number_of_points_climb, 3.0)
        aoa_cruise = np.full(number_of_points_cruise, 2.0)
        aoa_descent = np.full(number_of_points_descent, 1.0)
        aoa_reserve = np.full(number_of_points_reserve, 7.0)

        dummy_aoa = np.concatenate((aoa_climb, aoa_cruise, aoa_descent, aoa_reserve))

        return dummy_aoa

    def set_initial_guess_alpha(self, outputs):
        """
        Provides and sets educated guess of the  aoa required during the flight. It is a mere
        initial guess, the end results will still be accurate.

        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        dummy_aoa = self._get_initial_guess_alpha()

        outputs[
            "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium_alpha.alpha"
        ] = dummy_aoa

    def _get_initial_guess_delta_m(self) -> np.ndarray:
        """
        Provides an educated guess of the elevator deflection required during the flight. It is a
        mere initial guess, the end results will still be accurate. Does not set it.
        """

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        delta_m_climb = np.full(number_of_points_climb, -10.0)
        delta_m_cruise = np.full(number_of_points_cruise, -2.0)
        delta_m_descent = np.full(number_of_points_descent, -5.0)
        delta_m_reserve = np.full(number_of_points_reserve, -2.0)

        dummy_delta_m = np.concatenate(
            (delta_m_climb, delta_m_cruise, delta_m_descent, delta_m_reserve)
        )

        return dummy_delta_m

    def set_initial_guess_delta_m(self, outputs):
        """
        Provides and sets educated guess of the elevator deflection required during the flight.
        It is a mere initial guess, the end results will still be accurate.

        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        dummy_delta_m = self._get_initial_guess_delta_m()

        outputs[
            "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium_delta_m.delta_m"
        ] = dummy_delta_m

    def _get_initial_guess_true_airspeed(
        self,
        mass: np.ndarray,
        wing_area: float,
        cruise_altitude: float,
        cl_max_clean: float,
        cruise_tas: float,
        reserve_altitude: float,
    ) -> np.ndarray:
        """
        Provides an educated guess of the airspeed during the flight. It is a mere initial guess,
        the end results will still be accurate. Does not set it. It is a bit redundant since it
        mostly redoes the computation done in the initialization based on the mass initial guess,
        but it will allow us to have an initial guess of the propulsive power during flight and
        also reduce residuals

        :param mass: evolution of mass during the flight
        :param wing_area: aircraft wing area
        :param cruise_altitude: altitude set for cruise
        :param cl_max_clean: maximum lift coefficient in clean configuration
        :param cruise_tas: cruise true airspeed
        :param reserve_altitude: reserve_altitude
        """

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        mtow = mass[0]

        density_cruise = Atmosphere(cruise_altitude, altitude_in_feet=False).density

        vs1_climb = np.sqrt((mtow * g) / (0.5 * DENSITY_SL * wing_area * cl_max_clean))
        climb_eas = 1.3 * vs1_climb
        end_of_climb_tas = climb_eas * np.sqrt(DENSITY_SL / density_cruise)
        speed_array_climb = np.linspace(climb_eas, end_of_climb_tas, number_of_points_climb)

        speed_array_cruise = np.full(number_of_points_cruise, cruise_tas)

        mass_end_cruise = mass[number_of_points_climb + number_of_points_cruise - 1]
        # To check but I assume that since I ask it in m in the initialization, it will be in m here
        vs1_descent = np.sqrt(
            (mass_end_cruise * g) / (0.5 * density_cruise * wing_area * cl_max_clean)
        )
        descent_eas = 1.3 * vs1_descent
        start_of_descent_tas = descent_eas * np.sqrt(DENSITY_SL / density_cruise)
        speed_array_descent = np.linspace(
            start_of_descent_tas, descent_eas, number_of_points_descent
        )

        mass_start_reserve = mass[
            number_of_points_climb + number_of_points_cruise + number_of_points_descent
        ]
        # To check but I assume that since I ask it in m in the initialization, it will be in m here
        density_reserve = Atmosphere(reserve_altitude, altitude_in_feet=False).density
        vs1_reserve = np.sqrt(
            (mass_start_reserve * g) / (0.5 * density_reserve * wing_area * cl_max_clean)
        )
        speed_array_reserve = np.full(number_of_points_reserve, 1.3 * vs1_reserve)

        dummy_tas_array = np.concatenate(
            (speed_array_climb, speed_array_cruise, speed_array_descent, speed_array_reserve)
        )

        return dummy_tas_array

    def set_initial_guess_speed(self, inputs, outputs):
        """
        Provides and sets educated guess of the true airspeed during the flight.
        It is a mere initial guess, the end results will still be computed.

        :param inputs: OpenMDAO vector containing the value of inputs
        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        # For mass, the initial guess must have been set beforehand. A downside of doing the initial
        # guess like that to access the data, we need the submodel to actually exist. E.g if the
        # climb speed submodel is set to null, we won't be able to access the cl_max_clean.
        # One point to be careful about is that the non_linear guessing is done before any model is
        # actually ran. Therefore it relies on initial values for other variables and problem
        # inputs. This should be kept in mind !
        dummy_tas_array = self._get_initial_guess_true_airspeed(
            mass=outputs["solve_equilibrium.update_mass.mass"],
            wing_area=float(
                inputs[
                    "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium_alpha.data:geometry:wing:area"
                ]
            ),
            cruise_altitude=float(
                inputs[
                    "initialization.initialize_altitude.data:mission:sizing:main_route:cruise:altitude"
                ]
            ),
            cl_max_clean=float(
                inputs[
                    "initialization.initialize_reserve_speed.data:aerodynamics:wing:low_speed:CL_max_clean"
                ]
            ),
            cruise_tas=float(inputs["initialization.initialize_airspeed.data:TLAR:v_cruise"]),
            reserve_altitude=float(
                inputs[
                    "initialization.initialize_altitude.data:mission:sizing:main_route:reserve:altitude"
                ]
            ),
        )

        outputs["initialization.initialize_airspeed.true_airspeed"] = dummy_tas_array

    @staticmethod
    def _get_initial_guess_taxi_thrust(
        speed_to: float,
        speed_ti: float,
    ) -> Tuple[float, float]:
        """
        Provides an initial guess of the taxi thrust. The scope of this initial guess is reduced
        due to the fact that I realized it's wiser to only use problem inputs and not results
        from other module

        :param speed_to: target speed during taxi out
        :param speed_ti: target speed during taxi in
        """

        thrust_to = MIN_POWER_TAXI / speed_to
        thrust_ti = MIN_POWER_TAXI / speed_ti

        return thrust_to, thrust_ti

    def set_initial_guess_taxi_thrust(self, inputs, outputs):
        """
        Provides and sets educated guess of the thrust required during taxi. Is actually more
        than an initial guess.

        :param inputs: OpenMDAO vector containing the value of inputs
        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        thrust_to, thrust_ti = self._get_initial_guess_taxi_thrust(
            speed_to=inputs[
                "solve_equilibrium.compute_taxi_thrust.data:mission:sizing:taxi_out:speed"
            ],
            speed_ti=inputs[
                "solve_equilibrium.compute_taxi_thrust.data:mission:sizing:taxi_in:speed"
            ],
        )

        outputs[
            "solve_equilibrium.compute_taxi_thrust.data:mission:sizing:taxi_out:thrust"
        ] = thrust_to
        outputs[
            "solve_equilibrium.compute_taxi_thrust.data:mission:sizing:taxi_in:thrust"
        ] = thrust_ti

    @staticmethod
    def _get_initial_guess_speed_econ(
        speed_mission: np.ndarray, speed_to: float, speed_ti: float
    ) -> np.ndarray:
        """
        Provides an initial guess of the speed econ. Basically the same as the component itself,
        we'll simply concatenate data.

        :param speed_mission: speed during mission
        :param speed_to: target speed during taxi out
        :param speed_ti: target speed during taxi in
        """

        return np.concatenate((speed_to, speed_mission, speed_ti))

    def set_initial_guess_speed_econ(self, inputs, outputs):
        """
        Provides and sets educated guess of the speed for energy consumption. Is actually more
        than an initial guess. Needs to be run after the initialization of TAS

        :param inputs: OpenMDAO vector containing the value of inputs
        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        dummy_speed_econ = self._get_initial_guess_speed_econ(
            speed_mission=outputs["initialization.initialize_airspeed.true_airspeed"],
            speed_to=inputs[
                "solve_equilibrium.compute_taxi_thrust.data:mission:sizing:taxi_out:speed"
            ],
            speed_ti=inputs[
                "solve_equilibrium.compute_taxi_thrust.data:mission:sizing:taxi_in:speed"
            ],
        )

        outputs[
            "solve_equilibrium.compute_dep_equilibrium.preparation_for_energy_consumption.true_airspeed_econ"
        ] = dummy_speed_econ

    @staticmethod
    def _get_initial_guess_thrust_econ(
        thrust_mission: np.ndarray, thrust_to: float, thrust_ti: float
    ) -> np.ndarray:
        """
        Provides an initial guess of the thrust econ. Basically the same as the component itself,
        we'll simply concatenate data.

        :param thrust_mission: speed during mission
        :param thrust_to: target speed during taxi out
        :param thrust_ti: target speed during taxi in
        """

        return np.concatenate((thrust_to, thrust_mission, thrust_ti))

    def set_initial_guess_thrust_econ(self, outputs):
        """
        Provides and sets educated guess of the thrust for energy consumption. Is actually more
        than an initial guess. Needs to be run after the initialization of thrust mission and taxi thrust

        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        dummy_thrust_econ = self._get_initial_guess_thrust_econ(
            thrust_mission=outputs[
                "solve_equilibrium.compute_dep_equilibrium.compute_equilibrium_thrust.thrust"
            ],
            thrust_to=outputs[
                "solve_equilibrium.compute_taxi_thrust.data:mission:sizing:taxi_out:thrust"
            ],
            thrust_ti=outputs[
                "solve_equilibrium.compute_taxi_thrust.data:mission:sizing:taxi_in:thrust"
            ],
        )

        outputs[
            "solve_equilibrium.compute_dep_equilibrium.preparation_for_energy_consumption.thrust_econ"
        ] = dummy_thrust_econ

    def _get_initial_guess_altitude(
        self,
        cruise_altitude: float,
        reserve_altitude: float,
    ) -> np.ndarray:
        """
        Provides an educated guess of the altitude during the flight. It is a mere initial guess,
        the end results will still be accurate. Does not set it.

        :param cruise_altitude: altitude set for cruise
        :param reserve_altitude: altitude set for reserve
        """

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        altitude_array_climb = np.linspace(0.0, cruise_altitude, number_of_points_climb)
        altitude_array_cruise = np.full(number_of_points_cruise, cruise_altitude)
        altitude_array_descent = np.linspace(cruise_altitude, 0.0, number_of_points_descent)
        altitude_array_reserve = np.full(number_of_points_reserve, reserve_altitude)

        dummy_altitude = np.concatenate(
            (
                altitude_array_climb,
                altitude_array_cruise,
                altitude_array_descent,
                altitude_array_reserve,
            )
        )

        return dummy_altitude

    def set_initial_guess_altitude(self, inputs, outputs):
        """
        Provides and sets educated guess of the altitude during flight.

        :param inputs: OpenMDAO vector containing the value of inputs
        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        dummy_altitude = self._get_initial_guess_altitude(
            cruise_altitude=float(
                inputs[
                    "initialization.initialize_altitude.data:mission:sizing:main_route:cruise:altitude"
                ]
            ),
            reserve_altitude=float(
                inputs[
                    "initialization.initialize_altitude.data:mission:sizing:main_route:reserve:altitude"
                ]
            ),
        )

        outputs["initialization.initialize_altitude.altitude"] = dummy_altitude

    @staticmethod
    def _get_initial_guess_density(altitude: np.ndarray) -> np.ndarray:
        """
        Provides an educated guess of the density during the flight. It is a mere initial guess,
        the end results will still be accurate. Does not set it. Is based on the initial guess on
        altitude

        :param altitude: altitude during the flight
        """

        return Atmosphere(altitude=altitude, altitude_in_feet=False).density

    def set_initial_guess_density(self, outputs):
        """
        Provides and sets educated guess of the density during flight.

        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        dummy_density = self._get_initial_guess_density(
            altitude=outputs["initialization.initialize_altitude.altitude"]
        )

        outputs["initialization.initialize_density.density"] = dummy_density

    @staticmethod
    def _get_initial_guess_temperature(altitude: np.ndarray) -> np.ndarray:
        """
        Provides an educated guess of the exterior temperature during the flight. It is a mere
        initial guess, the end results will still be accurate. Does not set it. Is based on the
        initial guess on altitude

        :param altitude: altitude during the flight
        """

        return Atmosphere(altitude=altitude, altitude_in_feet=False).temperature

    def set_initial_guess_temperature(self, outputs):
        """
        Provides and sets educated guess of the exterior temperature during flight.

        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        dummy_temperature = self._get_initial_guess_temperature(
            altitude=outputs["initialization.initialize_altitude.altitude"]
        )

        outputs["initialization.initialize_temperature.exterior_temperature"] = dummy_temperature

    def set_initial_guess_energies(self, outputs):
        """
        Provides and sets educated guess of the energy consumed during flight, whether it is
        under the form of fuel or unconsumable energy. Actually, it is only an improvement in the
        sense that if we do all fuel, unconsumable will be set at zero and vice-versa. Doesn't do
        much for hybrid.

        :param outputs: OpenMDAO vector containing the value of outputs (and thus their initial
         guesses)
        """

        if self.options["power_train_file_path"]:
            if self.configurator.will_aircraft_mass_vary():
                outputs[
                    "solve_equilibrium.sizing_fuel.data:mission:sizing:fuel"
                ] = DUMMY_FUEL_CONSUMED

            else:
                outputs["solve_equilibrium.sizing_fuel.data:mission:sizing:fuel"] = 0.0

            if self.configurator.has_fuel_non_consumable_energy_source():
                outputs[
                    "solve_equilibrium.sizing_fuel.data:mission:sizing:energy"
                ] = DUMMY_ENERGY_CONSUMED

            else:
                outputs["solve_equilibrium.sizing_fuel.data:mission:sizing:energy"] = 0.0

        else:
            # If there are no PT file we assume full fuel
            outputs["solve_equilibrium.sizing_fuel.data:mission:sizing:fuel"] = DUMMY_FUEL_CONSUMED
            outputs["solve_equilibrium.sizing_fuel.data:mission:sizing:energy"] = 0.0


def get_propulsive_power(
    propulsor_names: list,
    thrust_distributor: np.ndarray,
    thrust: np.ndarray,
    true_airspeed: np.ndarray,
) -> dict:
    """
    Returns a dictionary containing the propulsive power, at each point of the flight, that the
    propulsor will have to produce. Is based on the thrust distributor but unfortunately,
    since this function is ran BEFORE any subsystems, we must do the computation twice.

    :param propulsor_names: names of the propulsor inside the propulsion chain
    :param thrust_distributor: array containing the percent repartition of the thrust among each
    propulsor
    :param thrust: aircraft level thrust, in N ? (I don't see how to check it)
    :param true_airspeed: true airspeed, in m/s ?
    """

    propulsive_power_dict = {}

    normalized_thrust_distribution = thrust_distributor / np.sum(thrust_distributor)

    for propulsor_name in propulsor_names:
        propulsive_power_dict[propulsor_name] = (
            thrust
            * true_airspeed
            * normalized_thrust_distribution[propulsor_names.index(propulsor_name)]
        )

    return propulsive_power_dict
