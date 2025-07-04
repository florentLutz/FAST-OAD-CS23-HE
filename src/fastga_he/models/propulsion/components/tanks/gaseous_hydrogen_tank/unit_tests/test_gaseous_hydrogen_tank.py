# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO


import pytest
import numpy as np
import os.path as pth
import openmdao.api as om

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
from ..components.sizing_tank_gravimetric_index import SizingGaseousHydrogenTankGravimetricIndex
from ..components.sizing_tank_drag import SizingGaseousHydrogenTankDrag
from ..components.sizing_tank_wall_thickness import SizingGaseousHydrogenTankWallThickness
from ..components.sizing_tank_length_fuselage_contraints import (
    SizingGaseousHydrogenTankLengthFuselageConstraints,
)

from ..components.cstr_enforce import ConstraintsGaseousHydrogenTankCapacityEnforce
from ..components.cstr_ensure import ConstraintsGaseousHydrogenTankCapacityEnsure

from ..components.perf_fuel_consumed_mission import PerformancesGaseousHydrogenConsumedMission
from ..components.perf_fuel_consumed_main_route import PerformancesGaseousHydrogenConsumedMainRoute
from ..components.perf_fuel_remaining import PerformancesGaseousHydrogenRemainingMission

from ..components.lcc_gaseous_hydrogen_tank_cost import LCCGaseousHydrogenTankCost

from ..components.sizing_tank import SizingGaseousHydrogenTank
from ..components.perf_gaseous_hydrogen_tank import PerformancesGaseousHydrogenTank

from ..constants import POSSIBLE_POSITION, MULTI_TANK_FACTOR

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
    expected_values = [1.73871, 2.8847, 3.96643, 1.73871]

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
    expected_values = [0.0, 1.848, 0.0, 0.0]

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
    ivc.add_output(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:number_of_tank",
        val=1.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTankLength(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:length",
        units="m",
    ) == pytest.approx(1.97802, rel=1e-2)

    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_length_outside_fuselage():
    ivc = get_indep_var_comp(
        list_inputs(
            SizingGaseousHydrogenTankLength(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1")
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:number_of_tank",
        val=2.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingGaseousHydrogenTankLength(
            gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", position="wing_pod"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:length",
        units="m",
    ) == pytest.approx(1.97802, rel=1e-2)

    problem.check_partials(compact_print=True, step=1e-7)


def test_multi_tank_length():
    d_outers = [
        0.97802,
        0.48901,
        0.45390,
        0.40511,
        0.36206,
        0.32601,
        0.32601,
        0.29594,
        0.27069,
    ]
    wall_thickness = 1.3034e-4
    for n_int, _ in MULTI_TANK_FACTOR.items():
        # Research independent input value in .xml file
        n = float(n_int)
        ivc = om.IndepVarComp()
        d_outer = d_outers[n_int - 1]
        ivc.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:outer_diameter",
            val=d_outer,
            units="m",
        )
        d_inner = d_outers[n_int - 1] - 2 * wall_thickness
        ivc.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:inner_diameter",
            val=d_inner,
            units="m",
        )
        ivc.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:inner_volume",
            val=1240.288,
            units="L",
        )
        ivc.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:number_of_tank",
            val=n,
        )
        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingGaseousHydrogenTankLength(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"),
            ivc,
        )
        length = (1.240288 - np.pi * d_inner**3 / 6 * n) / (np.pi * n * d_inner**2 / 4) + d_outer
        assert problem.get_val(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:length",
            units="m",
        ) == pytest.approx(length, rel=1e-2)

        problem.check_partials(compact_print=True, step=1e-7)


def test_tank_outer_diameter():
    expected_values = [0.97802, 0.217, 0.97802, 0.217]

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


def test_mutli_tank_outer_diameter():
    expected_values = [
        0.97802,
        0.48901,
        0.45390,
        0.40511,
        0.36206,
        0.32601,
        0.32601,
        0.29594,
        0.27069,
    ]

    for n_int, _ in MULTI_TANK_FACTOR.items():
        # Research independent input value in .xml file
        n = float(n_int)
        ivc = get_indep_var_comp(
            list_inputs(
                SizingGaseousHydrogenTankOuterDiameter(
                    gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
                )
            ),
            __file__,
            XML_FILE,
        )
        ivc.add_output(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:number_of_tank",
            val=n,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingGaseousHydrogenTankOuterDiameter(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"
            ),
            ivc,
        )

        assert problem.get_val(
            "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:outer_diameter",
            units="m",
        ) == pytest.approx(expected_values[n_int - 1], rel=1e-2)

        problem.check_partials(compact_print=True, step=1e-7)


def test_tank_length_fuselage_constraints():
    # Research independent input value in .xml file
    expected_values = [-0.4994, -0.84646]
    position_in_fuselage = ["in_the_cabin", "in_the_back"]
    for option, expected_value in zip(position_in_fuselage, expected_values):
        ivc = get_indep_var_comp(
            list_inputs(
                SizingGaseousHydrogenTankLengthFuselageConstraints(
                    gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingGaseousHydrogenTankLengthFuselageConstraints(
                gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1", position=option
            ),
            ivc,
        )
        assert problem.get_val(
            "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
            "dimension:length",
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
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
        "dimension:inner_diameter",
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
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
        "dimension:wall_thickness",
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
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
        "gravimetric_index"
    ) == pytest.approx(0.0485, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_gaseous_hydrogen_tank_drag():
    expected_ls_drag = [0.0, 0.01057, 0.0, 0.00233519]
    expected_cruise_drag = [0.0, 0.01057, 0.0, 0.00230317]

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
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
                    "low_speed:CD0",
                ) == pytest.approx(ls_drag, rel=1e-2)
            else:
                assert problem.get_val(
                    "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
                    "cruise:CD0",
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
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:"
        "inner_diameter",
        units="m",
    ) == pytest.approx(0.97776, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:dimension:"
        "length",
        units="m",
    ) == pytest.approx(2.02166, rel=1e-2)

    problem.check_partials(compact_print=True, step=1e-7)


def test_constraints_enforce_tank_capacity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
        "fuel_total_mission",
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
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
        "fuel_total_mission",
        val=10.0,
        units="kg",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:" "capacity",
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
        "constraints:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
        "capacity",
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
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
        "fuel_consumed_mission",
        units="kg",
    ) == pytest.approx(276.85, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_hydrogen_gas_consumed_main_route():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_t", val=np.ones(NB_POINTS_TEST), units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesGaseousHydrogenConsumedMainRoute(
            gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1",
            number_of_points=NB_POINTS_TEST,
            number_of_points_reserve=2,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:fuel_consumed_main_route",
        units="kg",
    ) == pytest.approx(8.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_hydrogen_gas_remaining_mission():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
        "fuel_consumed_mission",
        units="kg",
        val=140.0,
    )
    ivc.add_output("fuel_consumed_t", val=np.full(NB_POINTS_TEST, 14.0), units="kg")

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
    ivc.add_output("fuel_consumed_t", val=np.linspace(13.37, 42.0, NB_POINTS_TEST), units="kg")

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
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:"
        "fuel_consumed_mission",
        units="kg",
    ) == pytest.approx(279.62, rel=1e-2)
    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_cost():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:mass",
        units="kg",
        val=140.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCGaseousHydrogenTankCost(gaseous_hydrogen_tank_id="gaseous_hydrogen_tank_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:purchase_cost",
        units="USD",
    ) == pytest.approx(893.2, rel=1e-2)

    problem.check_partials(compact_print=True)
