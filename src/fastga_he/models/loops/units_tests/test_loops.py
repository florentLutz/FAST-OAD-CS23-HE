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
import fastoad.api as oad

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

    problem_loop = run_system(
        UpdateWingAreaLiftDEPEquilibrium(propulsion_id="fastga.wrapper.propulsion.basicIC_engine"),
        ivc_loop,
    )
    assert_allclose(problem_loop["wing_area"], 10.01, atol=1e-2)

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
    ivc_cons.add_output("data:geometry:wing:area", val=10.01, units="m**2")
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


def test_advanced_cl_with_proper_submodels():

    xml_file = "pipistrel_like.xml"
    propulsion_file = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

    oad.RegisterSubmodel.active_models[
        "submodel.performances_he.energy_consumption"
    ] = "fastga_he.submodel.performances.energy_consumption.from_pt_file"
    oad.RegisterSubmodel.active_models[
        "submodel.performances_he.dep_effect"
    ] = "fastga_he.submodel.performances.dep_effect.from_pt_file"

    inputs_list = list_inputs(
        UpdateWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            power_train_file_path=propulsion_file,
        )
    )
    # Research independent input value in .xml file
    ivc_loop = get_indep_var_comp(
        inputs_list,
        __file__,
        xml_file,
    )

    problem_loop = run_system(
        UpdateWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            power_train_file_path=propulsion_file,
        ),
        ivc_loop,
    )
    assert_allclose(problem_loop["wing_area"], 10.01, atol=1e-2)

    inputs_list = list_inputs(
        ConstraintWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            power_train_file_path=propulsion_file,
        )
    )

    inputs_list.remove("data:geometry:wing:area")
    # Research independent input value in .xml file
    ivc_cons = get_indep_var_comp(
        inputs_list,
        __file__,
        xml_file,
    )
    ivc_cons.add_output("data:geometry:wing:area", val=10.01, units="m**2")
    problem_cons = run_system(
        ConstraintWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            power_train_file_path=propulsion_file,
        ),
        ivc_cons,
    )
    assert_allclose(
        problem_cons.get_val("data:constraints:wing:additional_CL_capacity"),
        0.0,
        atol=1e-2,
    )
