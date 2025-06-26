# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth
from shutil import rmtree
import logging

import pytest

import openmdao.api as om
import fastoad.api as oad
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from fastga.models.aerodynamics.components import ComputeDeltaElevator
from fastga.models.aerodynamics.components.fuselage import ComputeCmAlphaFuselage

from utils.filter_residuals import filter_residuals


DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


@pytest.fixture(scope="module")
def cleanup():
    """Empties results folder to avoid any conflicts."""
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)


def test_sizing_sr22(cleanup):
    # TODO: Check why he needs the propulsion data as inputs
    # ANS: still used for the Z_cg of the aircraft which is assumed to have only a minor influence

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_sr22.xml"
    process_file_name = "full_sizing_fuel.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_cirrus.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()
    problem.setup()

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    problem.write_outputs()

    assert problem.get_val("data:weight:aircraft:MTOW", units="kg") == pytest.approx(
        1601.0, rel=1e-2
    )
    assert problem.get_val("data:weight:aircraft:OWE", units="kg") == pytest.approx(992.0, rel=1e-2)
    assert problem.get_val("data:mission:sizing:fuel", units="kg") == pytest.approx(
        234.00, rel=1e-2
    )


def test_elevator():
    XML_FILE = "problem_outputs_from_RTA.xml"
    # ivc = om.IndepVarComp()
    ivc = get_indep_var_comp(list_inputs(ComputeDeltaElevator()), __file__, XML_FILE)

    ivc.add_output("data:geometry:horizontal_tail:elevator_chord_ratio", val=0.384)
    ivc.add_output(
        "data:mission:sizing:landing:elevator_angle", val=-0.4363323129985824, units="rad"
    )
    ivc.add_output("data:aerodynamics:low_speed:mach", val=0.185)
    ivc.add_output("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", val=6.39, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeDeltaElevator(), ivc)

    print(problem.get_val("data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1"))

    print(problem.get_val("data:aerodynamics:elevator:low_speed:CD_delta", units="1/rad**2"))

    """
    assert problem.get_val(
        "data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1"
    ) == pytest.approx(cl_delta_elev, abs=1e-4)
    assert problem.get_val(
        "data:aerodynamics:elevator:low_speed:CD_delta", units="rad**-2"
    ) == pytest.approx(cd_delta_elev, abs=1e-4)
    """


def test_cm_fus():
    XML_FILE = "problem_outputs_from_RTA.xml"
    # ivc = om.IndepVarComp()
    ivc = get_indep_var_comp(list_inputs(ComputeCmAlphaFuselage()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCmAlphaFuselage(), ivc)

    print(problem.get_val("data:aerodynamics:fuselage:cm_alpha", units="rad**-1"))


### """Tests components @ high speed!"""
"""def test_comp_high_speed():
   

    for mach_interpolation in [True, False]:
        problem = compute_aero(XML_FILE, use_openvsp, mach_interpolation, False)

        # Check obtained value(s) is/(are) correct
        if mach_interpolation:
            assert problem[
                "data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"
            ] == pytest.approx(cl_alpha_vector, abs=1e-2)
            assert problem[
                "data:aerodynamics:aircraft:mach_interpolation:mach_vector"
            ] == pytest.approx(mach_vector, abs=1e-2)
        else:
            assert problem["data:aerodynamics:wing:cruise:CL0_clean"] == pytest.approx(
                cl0_wing, abs=1e-4
            )
            assert problem.get_val(
                "data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1"
            ) == pytest.approx(cl_alpha_wing, abs=1e-3)
            assert problem["data:aerodynamics:wing:cruise:CM0_clean"] == pytest.approx(
                cm0, abs=1e-4
            )
            assert problem[
                "data:aerodynamics:wing:cruise:induced_drag_coefficient"
            ] == pytest.approx(coeff_k_wing, abs=1e-4)
            assert problem["data:aerodynamics:horizontal_tail:cruise:CL0"] == pytest.approx(
                cl0_htp, abs=1e-4
            )
            assert problem[
                "data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"
            ] == pytest.approx(coeff_k_htp, abs=1e-4)"""
