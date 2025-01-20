# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO


import openmdao.api as om
import pytest
import numpy as np
import os.path as pth

from ..components.sizing_tank_unusable_hydrogen import SizingCryogenicHydrogenTankUnusableHydrogen
from ..components.sizing_tank_total_hydrogen_mission import (
    SizingCryogenicHydrogenTankTotalHydrogenMission,
)
from ..components.sizing_tank_wall_thickness import SizingCryogenicHydrogenTankWallThickness
from ..components.sizing_tank_cg_x import SizingCryogenicHydrogenTankCGX
from ..components.sizing_tank_cg_y import SizingCryogenicHydrogenTankCGY
from ..components.sizing_tank_length import SizingCryogenicHydrogenTankLength
from ..components.sizing_tank_inner_volume import SizingCryogenicHydrogenTankInnerVolume
from ..components.sizing_tank_inner_diameter import SizingCryogenicHydrogenTankInnerDiameter
from ..components.sizing_tank_weight import SizingCryogenicHydrogenTankWeight
from ..components.sizing_tank_aspect_ratio import SizingCryogenicHydrogenTankAspectRatio
from ..components.sizing_tank_stress_coefficient import SizingCryogenicHydrogenTankStressCoefficinet
from ..components.sizing_gravimetric_index import SizingCryogenicHydrogenTankGravimetricIndex
from ..components.sizing_tank_drag import SizingCryogenicHydrogenTankDrag
from ..components.sizing_tank_outer_diameter import SizingCryogenicHydrogenTankOuterDiameter
from ..components.sizing_tank_wall_diameter import SizingCryogenicHydrogenTankWallDiameter
from ..components.sizing_tank_diameter_update import SizingCryogenicHydrogenTankDiameterUpdate
from ..components.sizing_tank_overall_length import SizingCryogenicHydrogenTankOverallLength
from ..components.sizing_tank_insulation_layer_thermal_resistance import (
    SizingCryogenicHydrogenTankInsulationThermalResistance,
)
from ..components.sizing_tank_wall_thermal_resistance import (
    SizingCryogenicHydrogenTankWallThermalResistance,
)
from ..components.sizing_tank_thermal_resistance import SizingCryogenicHydrogenTankThermalResistance
from ..components.sizing_tank_overall_length_fuselage_check import (
    SizingCryogenicHydrogenTankOverallLengthFuselageCheck,
)

from ..components.cstr_enforce import ConstraintsCryogenicHydrogenTankCapacityEnforce
from ..components.cstr_ensure import ConstraintsCryogenicHydrogenTankCapacityEnsure

from ..components.perf_fuel_mission_consumed import PerformancesLiquidHydrogenConsumedMission
from ..components.perf_fuel_remaining import PerformancesLiquidHydrogenRemainingMission
from ..components.perf_fuel_boil_off import PerformancesHydrogenBoilOffMission
from ..components.perf_exterior_temperature import PerformancesExteriorTemperature
from ..components.perf_rayleigh_number import PerformancesCryogenicHydrogenTankRayleighNumber
from ..components.perf_nusselt_number import PerformancesCryogenicHydrogenTankNusseltNumber
from ..components.perf_tank_skin_temperature import PerformancesLiquidHydrogenTankSkinTemperature
from ..components.perf_air_kinematic_viscosity import PerformancesAirKinematicViscosity
from ..components.perf_air_conductivity import PerformancesAirThermalConductivity
from ..components.perf_tank_heat_radiation import PerformancesCryogenicHydrogenTankRadiation
from ..components.perf_tank_heat_convection import PerformancesCryogenicHydrogenTankConvection
from ..components.perf_tank_heat_conduction import PerformancesCryogenicHydrogenTankConduction
from ..components.perf_tank_temperature import PerformancesLiquidHydrogenTankTemperature
from ..components.perf_total_boil_off_hydrogen import PerformancesHydrogenBoilOffTotal

from ..components.sizing_tank import SizingCryogenicHydrogenTank
from ..components.perf_cryogenic_hydrogen_tank import PerformancesCryogenicHydrogenTank

from ..constants import POSSIBLE_POSITION

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_tank.xml"
NB_POINTS_TEST = 10


def test_unusable_hydrogen_mission():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:overall_hydrogen_boil_off",
        val=3.0,
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankUnusableHydrogen(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:unusable_fuel_mission",
            units="kg",
        )
        == pytest.approx(3.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_total_hydrogen_mission():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankTotalHydrogenMission(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankTotalHydrogenMission(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:fuel_total_mission",
            units="kg",
        )
        == pytest.approx(1.01, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_inner_volume_cryogenic_hydrogen_tank():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankInnerVolume(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankInnerVolume(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:inner_volume",
            units="m**3",
        )
        == pytest.approx(0.014255, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_tank_cg_x():

    expected_values = [0.0, 1.73871, 2.8847, 3.96643, 1.73871]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingCryogenicHydrogenTankCGX(
                    cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingCryogenicHydrogenTankCGX(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", position=option
            ),
            ivc,
        )
        assert (
            problem.get_val(
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:CG:x",
                units="m",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        problem.check_partials(compact_print=True)


def test_tank_cg_y():

    expected_values = [0.0, 1.848, 0.0, 0.0]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingCryogenicHydrogenTankCGY(
                    cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingCryogenicHydrogenTankCGY(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", position=option
            ),
            ivc,
        )
        assert (
            problem.get_val(
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:CG:y",
                units="m",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        problem.check_partials(compact_print=True)


def test_tank_length():

    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankLength(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankLength(cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"),
        ivc,
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:length",
            units="m",
        )
        == pytest.approx(1.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_outer_diameter():

    expected_values = [0.97802, 0.97802, 1.2, 0.97802, 0.97802]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                SizingCryogenicHydrogenTankOuterDiameter(
                    cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingCryogenicHydrogenTankOuterDiameter(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", position=option
            ),
            ivc,
        )
        assert (
            problem.get_val(
                "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:outer_diameter",
                units="m",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        problem.check_partials(compact_print=True, step=1e-7)


def test_tank_adjust_outer_diameter():

    ivc = om.IndepVarComp()
    # Research independent input value in .xml file

    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:length",
        val=-1.0,
        units="m",
    )

    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:diameter",
        val=3.0,
        units="m",
    )
    ivc.add_output("data:geometry:fuselage:maximum_height", val=10.0, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankOuterDiameter(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", position="wing_pod"
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:outer_diameter",
            units="m",
        )
        == pytest.approx(2.3811, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)

    ivc = om.IndepVarComp()
    # Research independent input value in .xml file

    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:length",
        val=1.0,
        units="m",
    )

    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:diameter",
        val=8.0,
        units="m",
    )
    ivc.add_output("data:geometry:fuselage:maximum_height", val=10.0, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankOuterDiameter(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", position="in_the_fuselage"
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:outer_diameter",
            units="m",
        )
        == pytest.approx(7.5, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_wall_diameter():

    ivc = om.IndepVarComp()
    # Research independent input value in .xml file

    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:outer_diameter",
        val=1.0,
        units="m",
    )

    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:insulation_thickness",
        val=0.1,
        units="m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankWallDiameter(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:wall_diameter",
            units="m",
        )
        == pytest.approx(0.8, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_diameter_update():

    ivc = om.IndepVarComp()
    # Research independent input value in .xml file

    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:outer_diameter",
        val=1.0,
        units="m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankDiameterUpdate(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:diameter",
            units="m",
        )
        == pytest.approx(1.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_overall_length():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankOverallLength(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankOverallLength(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1",
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:overall_length",
            units="m",
        )
        == pytest.approx(1.98802, rel=1e-2)
    )
    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_overall_length_fuselage_check():
    # Research independent input value in .xml file
    expected_values = [1.638, -0.4894, 0.0, -0.83646, -0.4894]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):

        ivc = get_indep_var_comp(
            list_inputs(
                SizingCryogenicHydrogenTankOverallLengthFuselageCheck(
                    cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", position=option
                )
            ),
            __file__,
            XML_FILE,
        )

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            SizingCryogenicHydrogenTankOverallLengthFuselageCheck(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", position=option
            ),
            ivc,
        )
        assert (
            problem.get_val(
                "constraints:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:overall_length",
                units="m",
            )
            == pytest.approx(expected_value, rel=1e-2)
        )

        problem.check_partials(compact_print=True, step=1e-7)


def test_tank_aspect_ratio():

    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankAspectRatio(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankAspectRatio(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:aspect_ratio",
        )
        == pytest.approx(2.012, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_stress_coefficient():

    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankStressCoefficinet(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankStressCoefficinet(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:stress_coefficient",
        )
        == pytest.approx(0.7485, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_inner_diameter():

    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankInnerDiameter(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankInnerDiameter(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:inner_diameter",
            units="m",
        )
        == pytest.approx(0.97776, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)


def test_tank_wall_thickness():

    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankWallThickness(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankWallThickness(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:wall_thickness",
            units="m",
        )
        == pytest.approx(0.00513, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)


def test_cryogenic_hydrogen_tank_weight():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankWeight(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankWeight(cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:mass",
            units="kg",
        )
        == pytest.approx(4.88, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_cryogenic_hydrogen_tank_gravimetric_index():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankGravimetricIndex(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:mass",
        val=20.2,
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankGravimetricIndex(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:gravimetric_index"
    ) == pytest.approx(0.04715, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_cryogenic_hydrogen_tank_drag():

    expected_ls_drag = [0.0, 0.01078987, 0.0, 1.4878e-3]
    expected_cruise_drag = [0.0, 0.01078987, 0.0, 1.4674e-3]

    for option, ls_drag, cruise_drag in zip(
        POSSIBLE_POSITION, expected_ls_drag, expected_cruise_drag
    ):
        # Research independent input value in .xml file
        for ls_option in [True, False]:
            ivc = get_indep_var_comp(
                list_inputs(
                    SizingCryogenicHydrogenTankDrag(
                        cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1",
                        position=option,
                        low_speed_aero=ls_option,
                    )
                ),
                __file__,
                XML_FILE,
            )

            # Run problem and check obtained value(s) is/(are) correct
            problem = run_system(
                SizingCryogenicHydrogenTankDrag(
                    cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1",
                    position=option,
                    low_speed_aero=ls_option,
                ),
                ivc,
            )

            if ls_option:
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:low_speed:CD0",
                    )
                    == pytest.approx(ls_drag, rel=1e-2)
                )
            else:
                assert (
                    problem.get_val(
                        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:cruise:CD0",
                    )
                    == pytest.approx(cruise_drag, rel=1e-2)
                )

            problem.check_partials(compact_print=True)


def test_insulation_thermal_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankInsulationThermalResistance(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankInsulationThermalResistance(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:insulation:thermal_resistance",
            units="K/W",
        )
        == pytest.approx(94.765, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_wall_thermal_resistance():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTankWallThermalResistance(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankWallThermalResistance(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:wall_thermal_resistance",
            units="K/W",
        )
        == pytest.approx(2.015e-6, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_thermal_resistance():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:wall_thermal_resistance",
        val=1.0,
        units="K/W",
    )

    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:insulation:thermal_resistance",
        val=1.0,
        units="K/W",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTankThermalResistance(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:thermal_resistance",
            units="K/W",
        )
        == pytest.approx(2.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_sizing_tank():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingCryogenicHydrogenTank(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1",
            )
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:overall_hydrogen_boil_off",
        val=0.1,
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCryogenicHydrogenTank(cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:mass",
            units="kg",
        )
        == pytest.approx(1.05, rel=1e-2)
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:cruise:CD0"
    ) == pytest.approx(0.0, rel=1e-2)
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:inner_diameter",
            units="m",
        )
        == pytest.approx(0.96750504, rel=1e-2)
    )

    problem.check_partials(compact_print=True, step=1e-7)
    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_constraints_enforce_tank_capacity():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:fuel_total_mission",
        val=10.0,
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsCryogenicHydrogenTankCapacityEnforce(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:capacity",
            units="kg",
        )
        == pytest.approx(10.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_constraints_ensure_tank_capacity():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:fuel_total_mission",
        val=10.0,
        units="kg",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:capacity",
        val=15.0,
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ConstraintsCryogenicHydrogenTankCapacityEnsure(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "constraints:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:capacity",
            units="kg",
        )
        == pytest.approx(-5.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_hydrogen_gas_consumed_mission():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fuel_consumed_t", val=np.linspace(13.37, 42.0, NB_POINTS_TEST), units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesLiquidHydrogenConsumedMission(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:fuel_consumed_mission",
            units="kg",
        )
        == pytest.approx(276.85, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_hydrogen_gas_remaining_mission():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:fuel_total_mission",
        units="kg",
        val=140.0,
    )
    ivc.add_output("fuel_consumed_t", units="kg", val=np.full(NB_POINTS_TEST, 14.0))
    ivc.add_output("hydrogen_boil_off_t", units="kg", val=np.full(NB_POINTS_TEST, 1.0))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesLiquidHydrogenRemainingMission(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert problem.get_val("fuel_remaining_t", units="kg") == pytest.approx(
        np.array([140.0, 125.0, 110.0, 95.0, 80.0, 65.0, 50.0, 35.0, 20.0, 5.0]), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_hydrogen_boil_off():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()

    ivc.add_output("heat_conduction", units="J/s", val=np.full(NB_POINTS_TEST, 10.0))
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 1.0))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHydrogenBoilOffMission(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )
    assert problem.get_val("hydrogen_boil_off_t", units="kg") == pytest.approx(
        np.full(NB_POINTS_TEST, 2.239e-5), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_overall_hydrogen_boil_off():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()

    ivc.add_output(
        "hydrogen_boil_off_t",
        val=np.full(NB_POINTS_TEST, 0.3),
        units="kg",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHydrogenBoilOffTotal(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:overall_hydrogen_boil_off",
            units="kg",
        )
        == pytest.approx(3.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_tank_temperature():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesLiquidHydrogenTankTemperature(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesLiquidHydrogenTankTemperature(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:liquid_hydrogen_temperature",
            units="K",
        )
        == pytest.approx(34.69409465, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_air_kinematic_viscosity():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAirKinematicViscosity(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "air_kinematic_viscosity",
            units="m**2/s",
        )
        == pytest.approx(np.full(NB_POINTS_TEST, 1.461e-05), rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_exterior_temperature():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesExteriorTemperature(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "exterior_temperature",
            units="K",
        )
        == pytest.approx(np.full(NB_POINTS_TEST, 288.15), rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_tank_skin_temperature():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:thermal_resistance",
        val=10.0,
        units="K/W",
    )
    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:liquid_hydrogen_temperature",
        val=30.0,
        units="K",
    )
    ivc.add_output("heat_conduction", val=np.full(NB_POINTS_TEST, 10.0), units="W")
    ivc.add_output("exterior_temperature", val=np.full(NB_POINTS_TEST, 290.0), units="K")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesLiquidHydrogenTankSkinTemperature(
            number_of_points=NB_POINTS_TEST, cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "skin_temperature",
            units="K",
        )
        == pytest.approx(np.full(NB_POINTS_TEST, 130.0), rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_air_conductivity():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("exterior_temperature", val=np.full(NB_POINTS_TEST, 0.0), units="degC")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAirThermalConductivity(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "air_thermal_conductivity",
            units="W/m/K",
        )
        == pytest.approx(np.full(NB_POINTS_TEST, 0.024), rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_rayleigh_number():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("exterior_temperature", val=np.full(NB_POINTS_TEST, 300.0), units="K")
    ivc.add_output("skin_temperature", val=np.full(NB_POINTS_TEST, 200.0), units="K")
    ivc.add_output("air_kinematic_viscosity", val=np.full(NB_POINTS_TEST, 1.0), units="m**2/s")
    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:outer_diameter",
        val=0.98802,
        units="m",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCryogenicHydrogenTankRayleighNumber(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "tank_rayleigh_number",
        )
        == pytest.approx(np.full(NB_POINTS_TEST, 2.239), rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_tank_nusselt_number():

    expected_values = [
        np.full(NB_POINTS_TEST, 1.3),
        np.full(NB_POINTS_TEST, 1.3),
        np.full(NB_POINTS_TEST, 1.234),
        np.full(NB_POINTS_TEST, 1.3),
        np.full(NB_POINTS_TEST, 1.234),
    ]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                PerformancesCryogenicHydrogenTankNusseltNumber(
                    cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1",
                    position=option,
                    number_of_points=NB_POINTS_TEST,
                )
            ),
            __file__,
            XML_FILE,
        )

        if option == "wing_pod" or option == "underbelly":
            ivc.add_output("true_airspeed", val=np.full(NB_POINTS_TEST, 100.0), units="m/s")
            ivc.add_output(
                "air_kinematic_viscosity", val=np.full(NB_POINTS_TEST, 1.0), units="m**2/s"
            )
        else:
            ivc.add_output("true_airspeed", val=np.full(NB_POINTS_TEST, 100.0), units="m/s")
            ivc.add_output("exterior_temperature", val=np.full(NB_POINTS_TEST, 2.239))

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            PerformancesCryogenicHydrogenTankNusseltNumber(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1",
                position=option,
                number_of_points=NB_POINTS_TEST,
            ),
            ivc,
        )

        assert problem.get_val("tank_nusselt_number") == pytest.approx(expected_value, rel=1e-2)

        problem.check_partials(compact_print=True)


def test_tank_heat_convection():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:length",
        val=1.0,
        units="m",
    )

    ivc.add_output(
        "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:dimension:outer_diameter",
        val=0.98802,
        units="m",
    )

    ivc.add_output("skin_temperature", val=np.full(NB_POINTS_TEST, 250.0), units="K")
    ivc.add_output("exterior_temperature", val=np.full(NB_POINTS_TEST, 300.0), units="K")
    ivc.add_output("air_thermal_conductivity", val=np.full(NB_POINTS_TEST, 0.024), units="W/m/K")
    ivc.add_output("tank_nusselt_number", val=np.full(NB_POINTS_TEST, 1.0))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCryogenicHydrogenTankConvection(
            number_of_points=NB_POINTS_TEST, cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1"
        ),
        ivc,
    )

    assert (
        problem.get_val(
            "heat_convection",
            units="W",
        )
        == pytest.approx(np.full(NB_POINTS_TEST, 7.494), rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_tank_heat_conduction():

    # Research independent input value in .xml file
    ivc = om.IndepVarComp()

    ivc.add_output(
        "heat_convection",
        val=np.full(NB_POINTS_TEST, 1.0),
        units="W",
    )

    ivc.add_output(
        "heat_radiation",
        val=np.full(NB_POINTS_TEST, 1.0),
        units="W",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCryogenicHydrogenTankConduction(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "heat_conduction",
            units="W",
        )
        == pytest.approx(2.0, rel=1e-2)
    )

    problem.check_partials(compact_print=True)


def test_tank_radiation():

    expected_values = [
        np.full(NB_POINTS_TEST, 68.2266),
        np.full(NB_POINTS_TEST, 68.2266),
        np.full(NB_POINTS_TEST, 98.127),
        np.full(NB_POINTS_TEST, 68.2266),
        np.full(NB_POINTS_TEST, 98.127),
    ]

    for option, expected_value in zip(POSSIBLE_POSITION, expected_values):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(
            list_inputs(
                PerformancesCryogenicHydrogenTankRadiation(
                    cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1",
                    position=option,
                    number_of_points=NB_POINTS_TEST,
                )
            ),
            __file__,
            XML_FILE,
        )

        ivc.add_output("exterior_temperature", val=np.full(NB_POINTS_TEST, 300.0), units="K")
        ivc.add_output("skin_temperature", val=np.full(NB_POINTS_TEST, 200.0), units="K")

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(
            PerformancesCryogenicHydrogenTankRadiation(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1",
                position=option,
                number_of_points=NB_POINTS_TEST,
            ),
            ivc,
        )

        assert problem.get_val("heat_radiation", units="W") == pytest.approx(
            expected_value, rel=1e-2
        )

        problem.check_partials(compact_print=True)


def test_performances_cryogenic_hydrogen_tank():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesCryogenicHydrogenTank(
                cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1",
                number_of_points=NB_POINTS_TEST,
            )
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("fuel_consumed_t", val=np.linspace(13.37, 42.0, NB_POINTS_TEST))
    ivc.add_output("time_step", val=np.full(NB_POINTS_TEST, 200.0), units="s")
    ivc.add_output("true_airspeed", val=np.full(NB_POINTS_TEST, 100.0), units="m/s")

    problem = run_system(
        PerformancesCryogenicHydrogenTank(
            cryogenic_hydrogen_tank_id="cryogenic_hydrogen_tank_1", number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )
    assert (
        problem.get_val(
            "data:propulsion:he_power_train:cryogenic_hydrogen_tank:cryogenic_hydrogen_tank_1:fuel_consumed_mission",
            units="kg",
        )
        == pytest.approx(279.62, rel=1e-2)
    )
    problem.check_partials(compact_print=True, step=1e-7)
    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))
