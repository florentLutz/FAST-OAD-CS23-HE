# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import logging

import numpy as np
import openmdao.api as om

import fastoad.api as oad
from fastoad.module_management.constants import ModelDomain

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

_LOGGER = logging.getLogger(__name__)


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
        self.set_initial_guess_thrust(outputs=outputs, inputs=inputs)
        self.set_initial_guess_alpha(outputs=outputs)
        self.set_initial_guess_delta_m(outputs=outputs)

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

            # propulsive_power_dict = get_propulsive_power(
            #     propulsor_names,
            #     inputs["data:propulsion:he_power_train:thrust_distribution"],
            #     inputs["thrust"],
            #     inputs["energy_consumption.propeller_1.advance_ratio.true_airspeed_econ"],
            # )

    def _get_initial_guess_fuel_consumed(self) -> np.ndarray:
        """
        Provides an educated guess of the variation of fuel consumed during the flight. It is a
        mere initial guess, the end results will still be accurate. Does not set it.
        """

        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        # A mere initial_guess
        fuel_mass_all_flight = 200.0

        # The following fuel repartition will be adopted for now, 15% for climb, 50% for cruise,
        # 5% for descent, 30% for reserve
        fuel_climb = np.full(
            number_of_points_climb, 0.15 * fuel_mass_all_flight / number_of_points_climb
        )
        fuel_cruise = np.full(
            number_of_points_cruise, 0.50 * fuel_mass_all_flight / number_of_points_cruise
        )
        fuel_descent = np.full(
            number_of_points_descent, 0.15 * fuel_mass_all_flight / number_of_points_descent
        )
        fuel_reserve = np.full(
            number_of_points_reserve, 0.20 * fuel_mass_all_flight / number_of_points_reserve
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
        guess, the end results will still be accurate. Does not set it.
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
