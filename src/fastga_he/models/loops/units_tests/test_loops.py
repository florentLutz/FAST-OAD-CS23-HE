"""
test module for wing area computation.
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

import os
import os.path as pth
import copy
import pytest
import fastoad.api as oad

from numpy.testing import assert_allclose
from fastga_he.gui.power_train_network_viewer import power_train_network_viewer
from fastga_he.models.performances.mission_vector.constants import (
    HE_SUBMODEL_ENERGY_CONSUMPTION,
    HE_SUBMODEL_DEP_EFFECT,
)
from ..wing_area_component.wing_area_cl_dep_equilibrium import (
    UpdateWingAreaLiftDEPEquilibrium,
    ConstraintWingAreaLiftDEPEquilibrium,
)
from ..update_wing_area_group import UpdateWingAreaGroupDEP
from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture()
def restore_submodels():
    """
    Since the submodels in the configuration file differ from the defaults, this restore process
    ensures subsequent assembly tests run under default conditions.
    """
    old_submodels = copy.deepcopy(oad.RegisterSubmodel.active_models)
    yield
    oad.RegisterSubmodel.active_models = old_submodels


def test_advanced_cl(restore_submodels):
    xml_file = "pipistrel_like.xml"

    oad.RegisterSubmodel.active_models[HE_SUBMODEL_ENERGY_CONSUMPTION] = (
        "fastga_he.submodel.performances.energy_consumption.basic"
    )
    oad.RegisterSubmodel.active_models[HE_SUBMODEL_DEP_EFFECT] = (
        "fastga_he.submodel.performances.dep_effect.none"
    )

    inputs_list = list_inputs(
        UpdateWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            produce_simplified_pt_file=True,
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
            produce_simplified_pt_file=True,
        ),
        ivc_loop,
    )
    assert_allclose(problem_loop["wing_area"], 9.97, atol=1e-2)

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
    ivc_cons.add_output("data:geometry:wing:area", val=9.97, units="m**2")
    ivc_cons.add_output("wing_area:aerodynamic", val=9.97, units="m**2")
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


def test_advanced_cl_with_proper_submodels_turboshaft():
    xml_file = "kodiak_100.xml"
    propulsion_file = pth.join(DATA_FOLDER_PATH, "turboshaft_propulsion.yml")

    inputs_list = list_inputs(
        UpdateWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            power_train_file_path=propulsion_file,
            produce_simplified_pt_file=True,
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
            produce_simplified_pt_file=True,
        ),
        ivc_loop,
    )
    assert_allclose(problem_loop["wing_area"], 23.36, atol=1e-2)


def test_advanced_cl_with_proper_submodels():
    xml_file = "pipistrel_like.xml"
    propulsion_file = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

    inputs_list = list_inputs(
        UpdateWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            power_train_file_path=propulsion_file,
            produce_simplified_pt_file=True,
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
            produce_simplified_pt_file=True,
        ),
        ivc_loop,
    )
    assert_allclose(problem_loop["wing_area"], 9.97, atol=1e-2)

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
    ivc_cons.add_output("data:geometry:wing:area", val=9.97, units="m**2")
    ivc_cons.add_output("wing_area:aerodynamic", val=9.97, units="m**2")
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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test is not meant to run in Github Actions.")
def test_inspect_octo_propulsion():
    propulsion_file = pth.join(DATA_FOLDER_PATH, "octo_assembly.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "octo_assembly.html")

    if not os.path.exists(network_file_path):
        power_train_network_viewer(propulsion_file, network_file_path)


def test_advanced_cl_octo_propulsion():
    xml_file = "octo_assembly.xml"
    propulsion_file = pth.join(DATA_FOLDER_PATH, "octo_assembly.yml")

    inputs_list = list_inputs(
        UpdateWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            power_train_file_path=propulsion_file,
            produce_simplified_pt_file=True,
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
            produce_simplified_pt_file=True,
        ),
        ivc_loop,
    )
    assert_allclose(problem_loop["wing_area"], 9.36, atol=1e-2)

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
    ivc_cons.add_output("wing_area:aerodynamic", val=9.32, units="m**2")
    problem_cons = run_system(
        ConstraintWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            power_train_file_path=propulsion_file,
        ),
        ivc_cons,
    )
    assert_allclose(
        problem_cons.get_val("data:constraints:wing:additional_CL_capacity"),
        0.126,
        atol=1e-2,
    )


def test_advanced_cl_group():
    xml_file = "pipistrel_like.xml"
    propulsion_file = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

    inputs_list = list_inputs(
        UpdateWingAreaGroupDEP(
            propulsion_id="", power_train_file_path=propulsion_file, produce_simplified_pt_file=True
        )
    )
    # Research independent input value in .xml file
    ivc_loop = get_indep_var_comp(
        inputs_list,
        __file__,
        xml_file,
    )

    problem = run_system(
        UpdateWingAreaGroupDEP(
            propulsion_id="", power_train_file_path=propulsion_file, produce_simplified_pt_file=True
        ),
        ivc_loop,
    )
    assert_allclose(problem.get_val("data:geometry:wing:area", units="m**2"), 9.97, atol=1e-2)
    assert_allclose(problem.get_val("data:constraints:wing:additional_CL_capacity"), 0.0, atol=1e-2)
    assert_allclose(
        problem.get_val("data:constraints:wing:additional_fuel_capacity", units="kg"),
        218.45,
        atol=1e-2,
    )


def test_advanced_cl_group_from_yml():
    # Define used files depending on options
    xml_file_name = "pipistrel_like.xml"
    process_file_name = "update_wing_area.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()

    # om.n2(problem)

    problem.run_model()
    problem.write_outputs()

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)

    assert_allclose(problem.get_val("data:geometry:wing:area", units="m**2"), 9.97, atol=1e-2)
    assert_allclose(problem.get_val("data:constraints:wing:additional_CL_capacity"), 0.0, atol=1e-2)
    assert_allclose(
        problem.get_val("data:constraints:wing:additional_fuel_capacity", units="kg"),
        218.45,
        atol=1e-2,
    )


def test_low_speed(restore_submodels):
    xml_file = "pipistrel_like.xml"
    propulsion_file = pth.join(DATA_FOLDER_PATH, "simple_assembly.yml")

    inputs_list = list_inputs(
        UpdateWingAreaGroupDEP(
            propulsion_id="",
            power_train_file_path=propulsion_file,
            produce_simplified_pt_file=True,
            low_speed_aero=True,
        )
    )
    # Research independent input value in .xml file
    ivc_loop = get_indep_var_comp(
        inputs_list,
        __file__,
        xml_file,
    )

    problem = run_system(
        UpdateWingAreaGroupDEP(
            propulsion_id="",
            power_train_file_path=propulsion_file,
            produce_simplified_pt_file=True,
            low_speed_aero=True,
        ),
        ivc_loop,
    )
    assert_allclose(problem.get_val("data:geometry:wing:area", units="m**2"), 10.03, atol=1e-2)

    oad.RegisterSubmodel.active_models[HE_SUBMODEL_ENERGY_CONSUMPTION] = (
        "fastga_he.submodel.performances.energy_consumption.basic"
    )
    oad.RegisterSubmodel.active_models[HE_SUBMODEL_DEP_EFFECT] = (
        "fastga_he.submodel.performances.dep_effect.none"
    )

    inputs_list = list_inputs(
        UpdateWingAreaLiftDEPEquilibrium(
            propulsion_id="fastga.wrapper.propulsion.basicIC_engine",
            produce_simplified_pt_file=True,
            low_speed_aero=True,
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
            produce_simplified_pt_file=True,
            low_speed_aero=True,
        ),
        ivc_loop,
    )
    assert_allclose(problem_loop["wing_area"], 10.03, atol=1e-2)
