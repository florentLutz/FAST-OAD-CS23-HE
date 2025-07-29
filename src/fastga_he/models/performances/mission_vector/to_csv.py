# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO.

import os
import logging

import numpy as np
import openmdao.api as om
import pandas as pd
from stdatm import Atmosphere

CSV_DATA_LABELS = [
    "time",
    "altitude",
    "ground_distance",
    "mass",
    "x_cg",
    "true_airspeed",
    "equivalent_airspeed",
    "mach",
    "d_vx_dt",
    "density",
    "exterior_temperature",
    "gamma",
    "alpha",
    "delta_m",
    "cl_wing",
    "cl_htp",
    "cl_aircraft",
    "cd_aircraft",
    "l_d_aircraft",
    "delta_Cl",
    "delta_Cd",
    "delta_Cm",
    "thrust (N)",
    "thrust_rate",
    "engine_setting",
    "tsfc (kg/s/N)",
    "fuel_flow (kg/s)",
    "energy_consumed (W*h)",
    "time step (s)",
    "name",
]

_LOGGER = logging.getLogger(__name__)  # Logger for this module


class ToCSV(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.previous_iter_count_apply = 0

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
        self.options.declare("out_file", default="", types=str)

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
            "d_vx_dt", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m/s**2"
        )
        self.add_input(
            "time", shape=number_of_points, val=np.full(number_of_points, np.nan), units="s"
        )
        self.add_input(
            "x_cg", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "altitude", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "exterior_temperature",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="degK",
        )
        self.add_input(
            "position", shape=number_of_points, val=np.full(number_of_points, np.nan), units="m"
        )
        self.add_input(
            "mass", val=np.full(number_of_points, np.nan), shape=number_of_points, units="kg"
        )
        self.add_input(
            "true_airspeed",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            units="m/s",
        )
        self.add_input(
            "equivalent_airspeed",
            val=np.full(number_of_points, np.nan),
            shape=number_of_points,
            units="m/s",
        )
        self.add_input(
            "gamma", val=np.full(number_of_points, np.nan), shape=number_of_points, units="deg"
        )
        self.add_input(
            "alpha", val=np.full(number_of_points, np.nan), shape=number_of_points, units="deg"
        )
        self.add_input(
            "delta_m", val=np.full(number_of_points, np.nan), shape=number_of_points, units="deg"
        )
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL0", val=np.nan)
        self.add_input(
            "data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan, units="rad**-1"
        )
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:aerodynamics:elevator:low_speed:CL_delta", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:elevator:low_speed:CD_delta", val=np.nan, units="rad**-2")
        self.add_input("delta_Cl", val=np.full(number_of_points, np.nan))
        self.add_input("delta_Cd", val=np.full(number_of_points, np.nan))
        self.add_input("delta_Cm", val=np.full(number_of_points, np.nan))
        self.add_input(
            "thrust", val=np.full(number_of_points, np.nan), shape=number_of_points, units="N"
        )
        self.add_input(
            "thrust_rate_t", val=np.full(number_of_points, np.nan), shape=number_of_points
        )
        self.add_input("engine_setting", val=np.full(number_of_points, np.nan))
        self.add_input(
            "fuel_consumed_t",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="kg",
        )
        self.add_input(
            "non_consumable_energy_t",
            shape=number_of_points,
            val=np.full(number_of_points, np.nan),
            units="W*h",
        )
        self.add_input(
            "time_step", shape=number_of_points, val=np.full(number_of_points, np.nan), units="s"
        )

        self.add_output(
            "tsfc", shape=number_of_points, val=np.full(number_of_points, 7e-6), units="kg/s/N"
        )

        self.declare_partials(
            of="*", wrt=["fuel_consumed_t", "time_step", "thrust"], method="exact"
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points_climb = self.options["number_of_points_climb"]
        number_of_points_cruise = self.options["number_of_points_cruise"]
        number_of_points_descent = self.options["number_of_points_descent"]
        number_of_points_reserve = self.options["number_of_points_reserve"]

        time = inputs["time"]
        altitude = inputs["altitude"]
        exterior_temperature = inputs["exterior_temperature"]
        distance = inputs["position"]
        mass = inputs["mass"]
        x_cg = inputs["x_cg"]
        v_tas = inputs["true_airspeed"]
        v_eas = inputs["equivalent_airspeed"]
        d_vx_dt = inputs["d_vx_dt"]
        atm = Atmosphere(altitude, altitude_in_feet=False)
        atm.true_airspeed = v_tas
        gamma = inputs["gamma"]
        alpha = inputs["alpha"] * np.pi / 180.0

        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        cl0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        cl_alpha_wing = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        coeff_k_wing = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        cl0_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL0"]
        cl_alpha_htp = inputs["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
        coeff_k_htp = inputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
        delta_cl = inputs["delta_Cl"]
        delta_cd = inputs["delta_Cd"]
        delta_cm = inputs["delta_Cm"]
        delta_m = inputs["delta_m"] * np.pi / 180.0
        cl_delta_m = inputs["data:aerodynamics:elevator:low_speed:CL_delta"]
        cd_delta_m = inputs["data:aerodynamics:elevator:low_speed:CD_delta"]
        cl_wing = cl0_wing + cl_alpha_wing * alpha + delta_cl
        cl_htp = cl0_htp + cl_alpha_htp * alpha + cl_delta_m * delta_m
        cl_aircraft = cl_wing + cl_htp

        cd_tot = (
            cd0
            + delta_cd
            + coeff_k_wing * cl_wing**2.0
            + coeff_k_htp * cl_htp**2.0
            + (cd_delta_m * delta_m**2.0)
        )

        l_d_aircraft = cl_aircraft / cd_tot

        thrust = inputs["thrust"]
        thrust_rate = inputs["thrust_rate_t"]
        engine_setting = inputs["engine_setting"]
        fuel_consumed_t = inputs["fuel_consumed_t"]
        non_consumable_energy_t = inputs["non_consumable_energy_t"]
        time_step = inputs["time_step"]

        tsfc = fuel_consumed_t / time_step / thrust
        fuel_flow = fuel_consumed_t / time_step

        name = np.concatenate(
            (
                np.full(number_of_points_climb, "sizing:main_route:climb"),
                np.full(number_of_points_cruise, "sizing:main_route:cruise"),
                np.full(number_of_points_descent, "sizing:main_route:descent"),
                np.full(number_of_points_reserve, "sizing:main_route:reserve"),
            )
        )

        if self.options["out_file"] != "":
            results_df = pd.DataFrame(columns=CSV_DATA_LABELS)
            results_df["time"] = time
            results_df["altitude"] = altitude
            results_df["ground_distance"] = distance
            results_df["mass"] = mass
            results_df["x_cg"] = x_cg
            results_df["true_airspeed"] = v_tas
            results_df["equivalent_airspeed"] = v_eas
            results_df["mach"] = atm.mach
            results_df["d_vx_dt"] = d_vx_dt
            results_df["density"] = atm.density
            results_df["exterior_temperature"] = exterior_temperature
            results_df["gamma"] = gamma
            results_df["alpha"] = alpha * 180.0 / np.pi
            results_df["delta_m"] = delta_m * 180.0 / np.pi
            results_df["cl_wing"] = cl_wing
            results_df["cl_htp"] = cl_htp
            results_df["cl_aircraft"] = cl_aircraft
            results_df["cd_aircraft"] = cd_tot
            results_df["l_d_aircraft"] = l_d_aircraft
            results_df["delta_Cl"] = delta_cl
            results_df["delta_Cd"] = delta_cd
            results_df["delta_Cm"] = delta_cm
            results_df["thrust (N)"] = thrust
            results_df["thrust_rate"] = thrust_rate
            results_df["engine_setting"] = engine_setting
            results_df["tsfc (kg/s/N)"] = tsfc
            results_df["energy_consumed (W*h)"] = non_consumable_energy_t
            results_df["name"] = name
            results_df["fuel_flow (kg/s)"] = fuel_flow
            results_df["time step (s)"] = time_step

            # If we are not currently using apply non linear, we save results. This allows us to
            # only save the results when the component is actually used

            if self.iter_count_apply == self.previous_iter_count_apply:
                if os.path.exists(self.options["out_file"]):
                    os.remove(self.options["out_file"])

                if not os.path.exists(os.path.dirname(self.options["out_file"])):
                    os.mkdir(os.path.dirname(self.options["out_file"]))

                results_df.to_csv(self.options["out_file"])

                _LOGGER.info("Saved mission results in %s", self.options["out_file"])

            else:
                self.previous_iter_count_apply = self.iter_count_apply

        outputs["tsfc"] = tsfc

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        thrust = inputs["thrust"]
        fuel_consumed_t = inputs["fuel_consumed_t"]
        time_step = inputs["time_step"]

        partials["tsfc", "thrust"] = np.diag(-fuel_consumed_t / time_step / thrust**2.0)
        partials["tsfc", "fuel_consumed_t"] = np.diag(1.0 / time_step / thrust)
        partials["tsfc", "time_step"] = np.diag(-fuel_consumed_t / time_step**2.0 / thrust)
