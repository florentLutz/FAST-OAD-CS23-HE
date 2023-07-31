"""
test module for wing area computation.
"""
#  This file is part of FAST-OAD_CS23 : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022  ONERA & ISAE-SUPAERO
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

import os.path as pth

import openmdao.api as om

from numpy.testing import assert_allclose

from ..wing_area_component.wing_area_cl_dep_equilibrium import (
    UpdateWingAreaLiftDEPEquilibrium,
    ConstraintWingAreaLiftDEPEquilibrium,
)

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")


def test_advanced_cl():

    xml_file = "pipistrel_like.xml"

    inputs_list = list_inputs(
        UpdateWingAreaLiftDEPEquilibrium(propulsion_id="fastga.wrapper.propulsion.basicIC_engine")
    )
    # Research independent input value in .xml file
    ivc_loop = get_indep_var_comp(
        inputs_list,
        __file__,
        xml_file,
    )
    ivc_loop.add_output("data:mission:sizing:taxi_in:thrust", val=1500, units="N")
    ivc_loop.add_output("data:mission:sizing:taxi_out:thrust", val=1500, units="N")

    problem_loop = run_system(
        UpdateWingAreaLiftDEPEquilibrium(propulsion_id="fastga.wrapper.propulsion.basicIC_engine"),
        ivc_loop,
    )
    assert_allclose(problem_loop["wing_area"], 17.39, atol=1e-2)

    inputs_list = list_inputs(
        ConstraintWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine"
        )
    )

    inputs_list.remove("data:geometry:wing:area")
    # Research independent input value in .xml file
    ivc_cons = get_indep_var_comp(
        inputs_list,
        __file__,
        xml_file,
    )
    ivc_cons.add_output("data:mission:sizing:taxi_in:thrust", val=1500, units="N")
    ivc_cons.add_output("data:mission:sizing:taxi_out:thrust", val=1500, units="N")
    ivc_cons.add_output("data:geometry:wing:area", val=17.39, units="m**2")
    problem_cons = run_system(
        ConstraintWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine"
        ),
        ivc_cons,
    )
    assert_allclose(
        problem_cons.get_val("data:constraints:wing:additional_CL_capacity"),
        0.0,
        atol=1e-2,
    )


def test_update_wing_area():

    ivc_geom = om.IndepVarComp()
    ivc_geom.add_output("wing_area:geometric", val=20.0, units="m**2")
    ivc_geom.add_output("wing_area:aerodynamic", val=15.0, units="m**2")

    problem_geom = run_system(UpdateWingArea(), ivc_geom)
    assert_allclose(problem_geom["data:geometry:wing:area"], 20.0, atol=1e-3)

    # _ = problem_geom.check_partials(compact_print=True)

    ivc_aero = om.IndepVarComp()
    ivc_aero.add_output("wing_area:geometric", val=10.0, units="m**2")
    ivc_aero.add_output("wing_area:aerodynamic", val=15.0, units="m**2")

    problem_aero = run_system(UpdateWingArea(), ivc_aero)
    assert_allclose(problem_aero["data:geometry:wing:area"], 15.0, atol=1e-3)

    # _ = problem_aero.check_partials(compact_print=True)


def test_update_wing_position():

    ivc = get_indep_var_comp(list_inputs(UpdateWingPosition()), __file__, "beechcraft_76.xml")

    problem = run_system(UpdateWingPosition(), ivc)
    assert_allclose(
        problem.get_val("data:geometry:wing:MAC:at25percent:x", units="m"), 3.4550, atol=1e-3
    )

    problem.check_partials(compact_print=True)
