"""
Computation of wing area update and constraints based on the equilibrium of the aircraft
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2025  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import os

import numpy as np
import openmdao.api as om

from scipy.constants import g

import fastoad.api as oad
from fastoad.openmdao.problem import AutoUnitsDefaultGroup
from fastoad.constants import EngineSetting

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from fastga.models.loops.constants import SUBMODEL_WING_AREA_AERO_LOOP, SUBMODEL_WING_AREA_AERO_CONS

from stdatm import Atmosphere

from fastga_he.command.api import list_inputs_metadata

from fastga_he.models.performances.mission_vector.constants import HE_SUBMODEL_EQUILIBRIUM

_LOGGER = logging.getLogger(__name__)

MIN_WING_AREA = 1.00


@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_AERO_LOOP, "fastga_he.submodel.loop.wing_area.update.aero.dep_equilibrium"
)
class UpdateWingAreaLiftDEPEquilibrium(om.ExplicitComponent):
    """
    Computes needed wing area to reach an equilibrium at required approach speed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()
        self.simplified_file_path = None
        self.control_parameter_list = None

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)
        self.options.declare(
            name="power_train_file_path",
            default="",
            desc="Path to the file containing the description of the power",
        )
        self.options.declare(
            name="sort_component",
            default=False,
            desc="Boolean to sort the component with proper order for adding subsystem operations",
            allow_none=False,
        )
        self.options.declare(
            name="produce_simplified_pt_file",
            default=False,
            desc="Boolean to split powertrain architecture into smaller branches",
            allow_none=False,
        )

    def setup(self):
        if self.options["power_train_file_path"]:
            self.configurator.load(self.options["power_train_file_path"])
            if self.options["produce_simplified_pt_file"]:
                self.simplified_file_path = self.configurator.produce_simplified_pt_file_copy()
            else:
                self.simplified_file_path = self.options["power_train_file_path"]
            self.control_parameter_list = self.configurator.get_control_parameter_list()

        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:CG:fwd:x", val=np.nan, units="m")

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

        input_zip = zip_equilibrium_input(
            self.options["propulsion_id"],
            self.simplified_file_path,
            self.options["sort_component"],
            self.control_parameter_list,
        )[0]
        for (
            var_names,
            var_unit,
            var_value,
            var_shape,
            var_shape_by_conn,
            var_copy_shape,
        ) in input_zip:
            if (
                var_names[:5] == "data:"
                and var_names != "data:geometry:wing:area"
                and var_names != "data:geometry:wing:MAC:length"
            ):
                if var_shape_by_conn:
                    self.add_input(
                        name=var_names,
                        val=np.nan,
                        units=var_unit,
                        shape_by_conn=var_shape_by_conn,
                        copy_shape=var_copy_shape,
                    )
                else:
                    self.add_input(
                        name=var_names,
                        val=var_value,
                        units=var_unit,
                        shape=var_shape,
                    )

        self.add_input(
            "settings:weight:aircraft:CG:fwd:MAC_position:margin",
            val=0.03,
            desc="Added margin for getting most fwd CG position, "
            "as ratio of mean aerodynamic chord",
        )

        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:mission:sizing:landing:elevator_angle", val=np.nan, units="deg")
        self.add_input("data:mission:sizing:takeoff:elevator_angle", val=np.nan, units="deg")

        self.add_output("wing_area", val=10.0, units="m**2")

        self.declare_partials(
            "wing_area",
            "*",
            method="fd",
        )

        if self.options["power_train_file_path"] and self.options["produce_simplified_pt_file"]:
            os.remove(self.simplified_file_path)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options["power_train_file_path"]:
            if self.options["produce_simplified_pt_file"]:
                self.simplified_file_path = self.configurator.produce_simplified_pt_file_copy()
            else:
                self.simplified_file_path = self.options["power_train_file_path"]

        # First, compute a failsafe value, in case the computation crashes because of the wrong
        # initial guesses of the problem
        stall_speed = inputs["data:TLAR:v_approach"] / 1.3
        mlw = inputs["data:weight:aircraft:MLW"]
        max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]

        wing_area_landing_init_guess = 2 * mlw * g / (stall_speed**2) / (1.225 * max_cl)

        wing_area_approach = compute_wing_area(
            inputs,
            self.options["propulsion_id"],
            self.simplified_file_path,
            self.control_parameter_list,
            self.options["sort_component"],
        )

        # Again with the damned optimizer. It can sometimes happen that he simply does not care
        # about being at the equilibrium and he will give a value of 1.0 m**2, consequently,
        # we must also filter out unreasonably low wing area, which is iffy. Indeed DEP can bring
        # significant gain but it might gain so much that it goes below the threshold for
        # unreasonably low values
        if (
            wing_area_approach > 1.2 * wing_area_landing_init_guess
            or wing_area_approach < 1.1 * MIN_WING_AREA
        ):
            wing_area_approach = wing_area_landing_init_guess
            _LOGGER.info(
                "Wing area too far from potential data, taking backup value for this iteration"
            )

        outputs["wing_area"] = wing_area_approach

        if self.options["power_train_file_path"] and self.options["produce_simplified_pt_file"]:
            # We can now delete the temp .yml we created, just to avoid over-clogging the repo
            os.remove(self.simplified_file_path)


@oad.RegisterSubmodel(
    SUBMODEL_WING_AREA_AERO_CONS,
    "fastga_he.submodel.loop.wing_area.constraint.aero.dep_equilibrium",
)
class ConstraintWingAreaLiftDEPEquilibrium(om.ExplicitComponent):
    """
    Computes the difference between the lift coefficient required for the low speed conditions
    and the what the wing can provide while maintaining an equilibrium. Will be an equivalent
    lift coefficient since the maximum one cannot be computed so easily. Equivalence will be
    computed based on the lift equation.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)
        self.options.declare(
            name="power_train_file_path",
            default="",
            desc="Path to the file containing the description of the power",
        )

    def setup(self):
        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        self.add_input("data:weight:aircraft:MLW", val=np.nan, units="kg")

        self.add_input("wing_area:aerodynamic", val=np.nan, units="m**2")

        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("data:constraints:wing:additional_CL_capacity")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        v_stall = inputs["data:TLAR:v_approach"] / 1.3
        mlw = inputs["data:weight:aircraft:MLW"]
        wing_area_actual = inputs["data:geometry:wing:area"]

        wing_area_constraint = inputs["wing_area:aerodynamic"]

        additional_cl = (
            (2.0 * mlw * g)
            / (1.225 * v_stall**2.0)
            * (1.0 / wing_area_constraint - 1.0 / wing_area_actual)
        )

        outputs["data:constraints:wing:additional_CL_capacity"] = additional_cl


class _IDThrustRate(om.ExplicitComponent):
    def setup(self):
        self.add_input("thrust_rate_t_econ", shape=3, val=np.full(3, np.nan))
        self.add_output("thrust_rate", shape=1)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["thrust_rate"] = inputs["thrust_rate_t_econ"][1]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        d_t_r_econ_d_t_r = np.array([1, 0, 1])
        partials["thrust_rate", "thrust_rate_t_econ"] = d_t_r_econ_d_t_r


def compute_wing_area(
    inputs, propulsion_id, pt_file_path, control_parameter_list, sort_component
) -> float:
    # To deactivate all the logging messages from matplotlib
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.pyplot").disabled = True

    # First, setup an initial guess
    stall_speed = inputs["data:TLAR:v_approach"] / 1.3
    mlw = inputs["data:weight:aircraft:MLW"]

    # We will only take the fwd position as it should be the most constraining since the HTP
    # generate negative lift in this case and also we take the value without including the safety
    # margin
    cg_margin = inputs["settings:weight:aircraft:CG:fwd:MAC_position:margin"]
    l0_wing = inputs["data:geometry:wing:MAC:length"]
    cg_max_fwd = inputs["data:weight:aircraft:CG:fwd:x"] + cg_margin * l0_wing

    # To compute the maximum AOA possible, should be done in its own component but oh well
    delta_cl_flaps = inputs["data:aerodynamics:flaps:landing:CL"]
    cl_alpha = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
    cl_0_wing = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
    max_cl = inputs["data:aerodynamics:aircraft:landing:CL_max"]

    alpha_max = (max_cl - delta_cl_flaps - cl_0_wing) / cl_alpha * 180.0 / np.pi

    min_elevator_angle = min(
        inputs["data:mission:sizing:landing:elevator_angle"],
        inputs["data:mission:sizing:takeoff:elevator_angle"],
    )

    wing_area_landing_init_guess = 2 * mlw * g / (stall_speed**2) / (1.225 * max_cl)

    # For some reasons that I don't understand, the driver can sometime fail even with
    # coherent value (at the 6 iteration of the MDA) while it works with absurd value (first
    # and second iteration). Yes I'm salty and I don't know where the debug message linked
    # with matplotlib come from

    try:
        input_zip, inputs_name_for_promotion = zip_equilibrium_input(
            propulsion_id, pt_file_path, sort_component, control_parameter_list
        )

        ivc = om.IndepVarComp()
        for var_names, var_unit, _, _, _, _ in input_zip:
            if var_names[:5] == "data:" and var_names != "data:geometry:wing:area":
                ivc.add_output(
                    name=var_names,
                    val=inputs[var_names],
                    units=var_unit,
                    shape=np.shape(inputs[var_names]),
                )

        ivc.add_output(name="d_vx_dt", val=np.array([0.0]), units="m/s**2")
        ivc.add_output(name="mass", val=np.array([mlw]), units="kg")
        # x_cg should be evaluated at the worst case scenario so either max aft or max fwd
        ivc.add_output(name="x_cg", val=np.array([cg_max_fwd]), units="m")
        ivc.add_output(name="gamma", val=np.array([0.0]), units="deg")
        ivc.add_output(name="altitude", val=np.array([0.0]), units="m")
        ivc.add_output(name="density", val=Atmosphere(np.array([0.0])).density, units="kg/m**3")
        ivc.add_output(name="exterior_temperature", val=Atmosphere(0.0).temperature, units="degK")
        # Time step is not important since we don't care about the fuel consumption
        ivc.add_output(name="time_step", val=np.array([0.1]), units="s")
        ivc.add_output(name="true_airspeed", val=np.array([stall_speed]), units="m/s")
        ivc.add_output(name="engine_setting", val=np.array([EngineSetting.TAKEOFF]))

        problem = om.Problem(reports=False)
        model = problem.model

        model.add_subsystem("ivc", ivc, promotes_outputs=["*"])

        option_equilibrium = {
            "number_of_points": 1,
            "promotes_all_variables": True,
            "propulsion_id": propulsion_id,
            "power_train_file_path": pt_file_path,
            "flaps_position": "landing",
            "sort_component": sort_component,
        }
        model.add_subsystem(
            "equilibrium",
            oad.RegisterSubmodel.get_submodel(HE_SUBMODEL_EQUILIBRIUM, options=option_equilibrium),
            promotes_inputs=inputs_name_for_promotion,
            promotes_outputs=["*"],
        )
        model.add_subsystem("thrust_rate_id", _IDThrustRate(), promotes=["*"])

        if pt_file_path:
            configurator = FASTGAHEPowerTrainConfigurator()
            configurator.load(pt_file_path)
            slip_ins, perf_outs = configurator.get_performances_to_slipstream_element_lists()

            for perf_out, slip_in in zip(perf_outs, slip_ins):
                model.connect("power_train_performances." + perf_out, slip_in)

        # SLSQP uses gradient ?
        problem.driver = om.ScipyOptimizeDriver()
        problem.driver.options["disp"] = False
        problem.driver.options["optimizer"] = "SLSQP"
        problem.driver.options["maxiter"] = 100
        problem.driver.options["tol"] = 1e-4

        problem.model.equilibrium.nonlinear_solver.options["rtol"] = 1e-8
        problem.model.equilibrium.nonlinear_solver.options["atol"] = 1e-8

        problem.model.add_design_var(
            name="data:geometry:wing:area",
            units="m**2",
            lower=MIN_WING_AREA,
            upper=2.0 * wing_area_landing_init_guess,
        )

        problem.model.add_objective(name="data:geometry:wing:area", units="m**2")

        problem.model.add_constraint(name="alpha", units="deg", lower=0.0, upper=alpha_max)
        problem.model.add_constraint(name="thrust_rate", lower=0.0, upper=1.0)
        problem.model.add_constraint(
            name="delta_m",
            lower=min_elevator_angle,
            upper=abs(min_elevator_angle),
        )

        problem.model.approx_totals()

        problem.setup()

        problem["data:geometry:wing:area"] = wing_area_landing_init_guess
        problem["delta_m"] = np.array(0.9 * min_elevator_angle)
        problem["alpha"] = np.array(0.9 * alpha_max)
        problem["thrust"] = np.array(mlw / 1.3)

        # This cause the logger to log a bunch of useless matplotlib information, question is,
        # how to turn it off
        problem.run_driver()

        wing_area_approach = problem.get_val("data:geometry:wing:area", units="m**2")
        print("Wing area in approach conditions", wing_area_approach)
        print(
            "Constraints: alpha/alpha_max; thrust; delta/delta_max: delta_cl",
            problem["alpha"] / alpha_max,
            problem["thrust_rate"],
            problem["delta_m"] / min_elevator_angle,
            problem["delta_Cl"],
        )

    except RuntimeError:
        wing_area_approach = wing_area_landing_init_guess

    # To reactivate them
    logging.getLogger("matplotlib.font_manager").disabled = False
    logging.getLogger("matplotlib.pyplot").disabled = False

    return wing_area_approach


def zip_equilibrium_input(propulsion_id, pt_file_path, sort_component, control_parameter_list=None):
    """
    Returns a list of the variables needed for the computation of the equilibrium. Based on
    the submodel currently registered and the propulsion_id required.

    :param propulsion_id: ID of propulsion wrapped to be used for computation of equilibrium.
    :param pt_file_path: Path to the powertrain file.
    :param sort_component: Option for powertrain component sorting.
    :param control_parameter_list: a list of control parameters to rename.
    :return inputs_zip: a zip containing a list of name, a list of units, a list of shapes,
    a list of shape_by_conn boolean and a list of copy_shape str.
    """
    new_component = AutoUnitsDefaultGroup()
    option_equilibrium = {
        "number_of_points": 1,
        "promotes_all_variables": True,
        "propulsion_id": propulsion_id,
        "power_train_file_path": pt_file_path,
        "flaps_position": "landing",
        "sort_component": sort_component,
    }
    new_component.add_subsystem(
        "system",
        oad.RegisterSubmodel.get_submodel(HE_SUBMODEL_EQUILIBRIUM, options=option_equilibrium),
        promotes=["*"],
    )

    name, unit, value, shape, shape_by_conn, copy_shape = list_inputs_metadata(new_component)
    names_for_promotion = []

    # If there are control parameters
    if control_parameter_list:
        # We check the name we need to add to see if it is the name of a control parameter. If it
        # is we replace it with a new name
        for idx, var_name in enumerate(name):
            if var_name[:5] == "data:" and var_name != "data:geometry:wing:area":
                if var_name in control_parameter_list:
                    # If "_mission" is already in the name we replace it with "_landing", otherwise
                    # we simply add "_landing at the end
                    if "_mission" in var_name:
                        new_var_name = var_name.replace("_mission", "_landing")
                    else:
                        new_var_name = var_name + "_landing"

                    name[idx] = new_var_name
                    names_for_promotion.append((var_name, new_var_name))
                else:
                    names_for_promotion.append(var_name)
            else:
                names_for_promotion.append(var_name)
    else:
        for var_name in name:
            names_for_promotion.append(var_name)

    inputs_zip = zip(name, unit, value, shape, shape_by_conn, copy_shape)

    return inputs_zip, names_for_promotion
