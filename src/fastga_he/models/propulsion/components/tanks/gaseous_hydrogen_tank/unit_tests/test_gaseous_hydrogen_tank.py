# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO


import openmdao.api as om
import pytest
import numpy as np
import os.path as pth

from ..components.sizing_tank_inner_volume import SizingGaseousHydrogenTankInnerVolume
from ..components.sizing_tank_total_hydrogen_mission import (
    SizingGaseousHydrogenTankTotalHydrogenMission,
)
from ..components.sizing_tank_unusable_hydrogen import SizingGaseousHydrogenTankUnusableHydrogen
from ..components.sizing_tank_cg_x import SizingGaseousHydrogenTankCGX
from ..components.sizing_tank_cg_y import SizingGaseousHydrogenTankCGY
from ..components.sizing_tank_length import SizingGaseousHydrogenTankLength
from ..components.sizing_tank_inner_diameter import SizingGaseousHydrogenTankInnerDiameter
from ..components.sizing_tank_outer_diameter import SizingGaseousHydrogenTankOuterDiameter
from ..components.sizing_tank_weight import SizingGaseousHydrogenTankWeight
from ..components.sizing_gravimetric_index import SizingGaseousHydrogenTankGravimetricIndex
from ..components.sizing_tank_drag import SizingGaseousHydrogenTankDrag
from ..components.sizing_tank_wall_thickness import SizingGaseousHydrogenTankWallThickness
from ..components.sizing_tank_overall_length import SizingGaseousHydrogenTankOverallLength
from ..components.sizing_tank_overall_length_fuselage_check import (
    SizingGaseousHydrogenTankOverallLengthFuselageCheck,
)

from ..components.cstr_enforce import ConstraintsGaseousHydrogenTankCapacityEnforce
from ..components.cstr_ensure import ConstraintsGaseousHydrogenTankCapacityEnsure

from ..components.perf_fuel_mission_consumed import PerformancesGaseousHydrogenConsumedMission
from ..components.perf_fuel_remaining import PerformancesGaseousHydrogenRemainingMission

from ..components.sizing_tank import SizingGaseousHydrogenTank
from ..components.perf_gaseous_hydrogen_tank import PerformancesGaseousHydrogenTank

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_tank.xml"
NB_POINTS_TEST = 10


def test_unusable_hydrogen_gas_mission():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingGaseousHydrogenTankUnusableHydrogen(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTankUnusableHydrogen(
            gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:unusable_fuel_mission",
        units="kg",
    ) == pytest.approx(0.03, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_total_hydrogen_gas_mission():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingGaseousHydrogenTankTotalHydrogenMission(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTankTotalHydrogenMission(
            gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:fuel_total_mission",
        units="kg",
    ) == pytest.approx(1.03, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_inner_volume_gaseous_hydrogen_tank():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingGaseousHydrogenTankInnerVolume(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTankInnerVolume(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:inner_volume",
        units="L",
    ) == pytest.approx(1272.265, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_tank_cg_x():
    expected_values = [0.0, 1.73871, 2.8847, 3.96643, 1.73871]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingGaseousHydrogenTankCGX(
                    gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingGaseousHydrogenTankCGX(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", position=option
            ),
            ivc,
        )
        assert problem.get_val(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:CG:x",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_tank_cg_y():
    expected_values = [0.0, 0.0, 1.848, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingGaseousHydrogenTankCGY(
                    gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingGaseousHydrogenTankCGY(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", position=option
            ),
            ivc,
        )
        assert problem.get_val(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:CG:y",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_tank_length():
    ivc = get_indep_var_comp(
        list_inputs(
            SizingGaseousHydrogenTankLength(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTankLength(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:length",
        units="m",
    ) == pytest.approx(1.0, rel=1e-2)

    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_outer_diameter():
    expected_values = [0.97802, 0.97802, 0.543, 0.97802, 0.543]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingGaseousHydrogenTankOuterDiameter(
                    gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingGaseousHydrogenTankOuterDiameter(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", position=option
            ),
            ivc,
        )
        assert problem.get_val(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:outer_diameter",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True, step=1e-7)


def test_tank_overall_length():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingGaseousHydrogenTankOverallLength(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTankOverallLength(
            gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1",
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:overall_length",
        units="m",
    ) == pytest.approx(1.97802, rel=1e-2)
    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_overall_length_fuselage_check():
    # Research independent input value in .xml file
    expected_values = [1.628, -0.4994, 0.0, -0.84646, -0.4994]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        ivc = get_indep_var_comp(
            list_inputs(
                SizingGaseousHydrogenTankOverallLengthFuselageCheck(
                    gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingGaseousHydrogenTankOverallLengthFuselageCheck(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", position=option
            ),
            ivc,
        )
        assert problem.get_val(
            "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:overall_length",
            units="m",
        ) == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True, step=1e-7)


def test_tank_inner_diameter():
    ivc = get_indep_var_comp(
        list_inputs(
            SizingGaseousHydrogenTankInnerDiameter(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTankInnerDiameter(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:inner_diameter",
        units="m",
    ) == pytest.approx(0.97776, rel=1e-2)

    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_wall_thickness():
    ivc = get_indep_var_comp(
        list_inputs(
            SizingGaseousHydrogenTankWallThickness(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTankWallThickness(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:wall_thickness",
        units="m",
    ) == pytest.approx(1.303e-4, rel=1e-2)

    problem.check_partials(compact_print=True, step=1e-7)


def test_gaseous_hydrogen_tank_weight():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingGaseousHydrogenTankWeight(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTankWeight(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:mass",
        units="kg",
    ) == pytest.approx(1.2179, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_gaseous_hydrogen_tank_gravimetric_index():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingGaseousHydrogenTankGravimetricIndex(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:mass",
        val=20.2,
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTankGravimetricIndex(
            gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:gravimetric_index"
    ) == pytest.approx(0.0471, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_gaseous_hydrogen_tank_drag():
    expected_ls_drag = [0.0, 0.0, 0.01057, 0.0, 1.4668e-3]
    expected_cruise_drag = [0.0, 0.0, 0.01057, 0.0, 1.44669e-3]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_ls_drag, expected_cruise_drag
    ):
        # Research independent input value in .xml file
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingGaseousHydrogenTankDrag(
                        gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1",
                        position=option,
                        low_speed_aero=ls_option,
                    )
                ),
                __file__,
                XML_FILE,
            )

            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingGaseousHydrogenTankDrag(
                    gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1",
                    position=option,
                    low_speed_aero=ls_option,
                ),
                ivc,
            )

            if ls_option:
                assert problem.get_val(
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:low_speed:CD0",
                ) == pytest.approx(ls_drag, rel=1e-2)
            else:
                assert problem.get_val(
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:cruise:CD0",
                ) == pytest.approx(cruise_drag, rel=1e-2)

            problem.check_partials(compact_print=True)


def test_sizing_tank():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(SizingGaseousHydrogenTank(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1")),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTank(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"),
        ivc,
    )
    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:mass",
        units="kg",
    ) == pytest.approx(2.476, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:cruise:CD0"
    ) == pytest.approx(0.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:inner_diameter",
        units="m",
    ) == pytest.approx(0.97776, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:overall_length",
        units="m",
    ) == pytest.approx(2.02166, rel=1e-2)

    problem.check_partials(compact_print=True, step=1e-7)


def test_constraints_enforce_tank_capacity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:fuel_total_mission",
        val=10.0,
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsGaseousHydrogenTankCapacityEnforce(
            gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:capacity",
        units="kg",
    ) == pytest.approx(10.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints_ensure_tank_capacity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:fuel_total_mission",
        val=10.0,
        units="kg",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:capacity",
        val=15.0,
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsGaseousHydrogenTankCapacityEnsure(
            gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:capacity",
        units="kg",
    ) == pytest.approx(-5.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_hydrogen_gas_consumed_mission():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_t", val=np.linspace(13.37, 42.0, NB_POINTS_TEST), units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesGaseousHydrogenConsumedMission(
            gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:fuel_consumed_mission",
        units="kg",
    ) == pytest.approx(276.85, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_hydrogen_gas_remaining_mission():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:fuel_consumed_mission",
        units="kg",
        val=140.0,
    )
    ivc.add_output("fuel_consumed_t", val=np.full(NB_POINTS_TEST, 14.0))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesGaseousHydrogenRemainingMission(
            gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert problem.get_val("fuel_remaining_t", units="kg") == pytest.approx(
        np.array([140.0, 126.0, 112.0, 98.0, 84.0, 70.0, 56.0, 42.0, 28.0, 14.0]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_performances_gaseous_hydrogen_tank():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_t", val=np.linspace(13.37, 42.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesGaseousHydrogenTank(
            gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert problem.get_val("fuel_remaining_t", units="kg") == pytest.approx(
        np.array([276.85, 263.48, 246.93, 227.2, 204.28, 178.19, 148.91, 116.46, 80.82, 42.0]),
        rel=1e-2,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:fuel_consumed_mission",
        units="kg",
    ) == pytest.approx(279.62, rel=1e-2)
    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))
