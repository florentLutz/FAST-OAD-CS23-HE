# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pytest

from ..delta_from_pt_file import SlipstreamAirframeLiftClean, SlipstreamAirframeLift
from ..wing_punctual_loads_from_pt_file import PowerTrainPunctualLoadsFromFile
from ..wing_punctual_tanks_from_pt_file import PowerTrainPunctualTanksFromFile
from ..wing_distributed_loads_from_pt_file import PowerTrainDistributedLoadsFromFile
from ..wing_distributed_tanks_from_pt_file import PowerTrainDistributedTanksFromFile
from ..fuel_cg_from_pt_file import FuelCGFromPTFile

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "reference_data.xml"
PROPULSION_FILE = pth.join("data", "sample_power_train_file.yml")
PROPULSION_FILE_TANKS = pth.join("data", "sample_fuel_propulsion_for_assembler.yml")
NB_POINTS_TEST = 10


def test_clean_lift_wing():
    ivc = get_indep_var_comp(
        list_inputs(SlipstreamAirframeLiftClean(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("alpha", val=np.full(NB_POINTS_TEST, 5.0), units="deg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SlipstreamAirframeLiftClean(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("cl_wing_clean") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.6533), rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_airframe_lift():
    flaps_positions = ["cruise", "landing", "takeoff"]
    expected_values = (
        np.full(NB_POINTS_TEST, 0.6533),
        np.full(NB_POINTS_TEST, 1.2620),
        np.full(NB_POINTS_TEST, 0.9536),
    )

    for flaps_position, expected_value in zip(flaps_positions, expected_values):
        ivc = get_indep_var_comp(
            list_inputs(
                SlipstreamAirframeLift(
                    number_of_points=NB_POINTS_TEST,
                    flaps_position=flaps_position,
                )
            ),
            __file__,
            XML_FILE,
        )
        ivc.add_output("cl_wing_clean", val=np.full(NB_POINTS_TEST, 0.6533), units="deg")

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SlipstreamAirframeLift(
                number_of_points=NB_POINTS_TEST,
                flaps_position=flaps_position,
            ),
            ivc,
        )

        assert problem.get_val("cl_airframe") == pytest.approx(expected_value, rel=1e-3)

        problem.check_partials(compact_print=True)


def test_punctual_mass_assembler():
    pt_file_path = pth.join(pth.dirname(__file__), PROPULSION_FILE)

    ivc = get_indep_var_comp(
        list_inputs(PowerTrainPunctualLoadsFromFile(power_train_file_path=pt_file_path)),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PowerTrainPunctualLoadsFromFile(power_train_file_path=pt_file_path),
        ivc,
    )

    assert problem.get_val("data:weight:airframe:wing:punctual_mass:y_ratio") == pytest.approx(
        np.array([0.8, 0.34]), rel=1e-6
    )
    assert problem.get_val(
        "data:weight:airframe:wing:punctual_mass:mass", units="kg"
    ) == pytest.approx(np.array([15.0, 10.0]), rel=1e-6)

    problem.check_partials(compact_print=True)


def test_punctual_tank_assembler():
    pt_file_path = pth.join(pth.dirname(__file__), PROPULSION_FILE_TANKS)

    ivc = get_indep_var_comp(
        list_inputs(PowerTrainPunctualTanksFromFile(power_train_file_path=pt_file_path)),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PowerTrainPunctualTanksFromFile(power_train_file_path=pt_file_path),
        ivc,
    )

    assert problem.get_val("data:weight:airframe:wing:punctual_tanks:y_ratio") == pytest.approx(
        np.array([0.4]), rel=1e-6
    )
    assert problem.get_val(
        "data:weight:airframe:wing:punctual_tanks:fuel_inside", units="kg"
    ) == pytest.approx(np.array([12.0]), rel=1e-6)

    problem.check_partials(compact_print=True)


def test_distributed_mass_assembler():
    pt_file_path = pth.join(pth.dirname(__file__), PROPULSION_FILE)

    ivc = get_indep_var_comp(
        list_inputs(PowerTrainDistributedLoadsFromFile(power_train_file_path=pt_file_path)),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PowerTrainDistributedLoadsFromFile(power_train_file_path=pt_file_path),
        ivc,
    )

    assert problem.get_val(
        "data:weight:airframe:wing:distributed_mass:y_ratio_start"
    ) == pytest.approx(0.3, rel=1e-6)
    assert problem.get_val(
        "data:weight:airframe:wing:distributed_mass:y_ratio_end"
    ) == pytest.approx(0.6, rel=1e-6)
    assert problem.get_val(
        "data:weight:airframe:wing:distributed_mass:start_chord"
    ) == pytest.approx(0.7, rel=1e-6)
    assert problem.get_val(
        "data:weight:airframe:wing:distributed_mass:chord_slope"
    ) == pytest.approx(0.0, rel=1e-6)
    assert problem.get_val("data:weight:airframe:wing:distributed_mass:mass") == pytest.approx(
        120.0, rel=1e-6
    )

    problem.check_partials(compact_print=True)


def test_distributed_tanks_assembler():
    pt_file_path = pth.join(pth.dirname(__file__), PROPULSION_FILE_TANKS)

    ivc = get_indep_var_comp(
        list_inputs(PowerTrainDistributedTanksFromFile(power_train_file_path=pt_file_path)),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        PowerTrainDistributedTanksFromFile(power_train_file_path=pt_file_path),
        ivc,
    )

    assert problem.get_val(
        "data:weight:airframe:wing:distributed_tanks:y_ratio_start"
    ) == pytest.approx(0.3, rel=1e-6)
    assert problem.get_val(
        "data:weight:airframe:wing:distributed_tanks:y_ratio_end"
    ) == pytest.approx(0.6, rel=1e-6)
    assert problem.get_val(
        "data:weight:airframe:wing:distributed_tanks:start_chord"
    ) == pytest.approx(0.7, rel=1e-6)
    assert problem.get_val(
        "data:weight:airframe:wing:distributed_tanks:chord_slope"
    ) == pytest.approx(0.0, rel=1e-6)
    assert problem.get_val(
        "data:weight:airframe:wing:distributed_tanks:fuel_inside"
    ) == pytest.approx(160.0, rel=1e-6)

    problem.check_partials(compact_print=True)


def test_fuel_cg_from_pt_file():
    pt_file_path = pth.join(pth.dirname(__file__), pth.join("data", "sample_fuel_propulsion.yml"))

    ivc = get_indep_var_comp(
        list_inputs(
            FuelCGFromPTFile(power_train_file_path=pt_file_path, number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "fuel_tank_1_fuel_remaining_t", units="kg", val=np.linspace(60.0, 0.0, NB_POINTS_TEST)
    )
    ivc.add_output(
        "fuel_tank_2_fuel_remaining_t", units="kg", val=np.linspace(15.0, 0.0, NB_POINTS_TEST)
    )

    problem = run_system(
        FuelCGFromPTFile(power_train_file_path=pt_file_path, number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("fuel_lever_arm_t_econ") == pytest.approx(
        np.linspace(232.5, 0, NB_POINTS_TEST), rel=1e-6
    )
    assert problem.get_val("fuel_mass_t_econ") == pytest.approx(
        np.linspace(75.0, 0, NB_POINTS_TEST), rel=1e-6
    )

    problem.check_partials(compact_print=True)
