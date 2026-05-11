# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import pytest
import numpy as np
import openmdao.api as om
import os.path as pth

from ..sizing_heat_sink_length import SizingHeatSinkFinLength
from ..sizing_heat_sink_fin_thickness import SizingHeatSinkFinThickness
from ..sizing_heat_sink_fin_spacing import SizingHeatSinkFinSpacing
from ..sizing_heat_sink_fin_height import SizingHeatSinkFinHeight
from ..sizing_heat_sink_weight import SizingHeatSinkMass
from ..sizing_heat_sink_added_wet_area import SizingHeatSinkWetArea
from ..sizing_finned_heat_sink import SizingFinnedHeatSink

from ..perf_heat_sink_base_temperature import PerformancesHeatSinkBasedTemperature
from ..perf_base_temperature_difference import PerformancesFinBaseTemperatureDifference
from ..perf_air_reynold_number import PerformancesAirReynoldsNumber
from ..perf_air_nusselt_number import PerformancesAirNusseltNumber
from ..perf_air_convection_heat_transfer_coefficient import (
    PerformancesAirConvectionHeatTransferCoefficient,
)
from ..perf_fin_parameter import PerformancesFinParameter
from ..perf_fin_heat_transfer_parameter import PerformancesFinHeatTransferParameter
from ..perf_design_dissipation_power import PerformancesDesignDissipationPower
from ..perf_finned_heat_sink import PerformancesFinnedHeatSink

from tests.testing_utilities import run_system

NB_POINTS_TEST = 10


def test_heat_sink_length():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:dimension:length",
        units="m",
        val=3.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHeatSinkFinLength(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_length",
        units="m",
    ) == pytest.approx(3.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_fin_thickness():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:dimension:height",
        units="m",
        val=0.72,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHeatSinkFinThickness(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_thickness",
        units="m",
    ) == pytest.approx(7.2e-4, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_fin_spacing():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:dimension:height",
        units="m",
        val=0.72,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:dimension:width",
        units="m",
        val=0.72,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_thickness",
        units="m",
        val=7.2e-4,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:number_of_fins",
        units="unitless",
        val=100,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHeatSinkFinSpacing(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_spacing",
        units="m",
    ) == pytest.approx(0.02088, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_fin_height():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:design_dissipation_power",
        units="kW",
        val=300.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_parameter",
        units="m**-1",
        val=13.8,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_heat_transfer_parameter",
        units="W/K",
        val=275.3,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":convection_heat_transfer_coefficient",
        units="W/m**2/K",
        val=100.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:number_of_fins",
        units="unitless",
        val=100,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHeatSinkFinHeight(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_height",
        units="m",
    ) == pytest.approx(0.526, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_weight():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_length",
        units="m",
        val=3.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_thickness",
        units="m",
        val=7.2e-4,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_height",
        units="m",
        val=0.526,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":number_of_fins",
        units="unitless",
        val=100,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHeatSinkMass(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":mass",
        units="kg",
    ) == pytest.approx(306.7, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_added_wet_area():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_length",
        units="m",
        val=3.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_thickness",
        units="m",
        val=7.2e-4,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_height",
        units="m",
        val=0.526,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":number_of_fins",
        units="unitless",
        val=100,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHeatSinkWetArea(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":wet_area",
        units="m**2",
    ) == pytest.approx(315.8, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_finned_heat_sink():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:dimension:length",
        units="m",
        val=3.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:dimension:height",
        units="m",
        val=0.72,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:dimension:width",
        units="m",
        val=0.72,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:"
        "design_dissipation_power",
        units="kW",
        val=300.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_parameter",
        units="m**-1",
        val=13.8,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_heat_transfer_parameter",
        units="W/K",
        val=275.3,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":convection_heat_transfer_coefficient",
        units="W/m**2/K",
        val=100.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:number_of_fins",
        units="unitless",
        val=100,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingFinnedHeatSink(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_length",
        units="m",
    ) == pytest.approx(3.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_thickness",
        units="m",
    ) == pytest.approx(7.2e-4, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_spacing",
        units="m",
    ) == pytest.approx(0.02088, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_height",
        units="m",
    ) == pytest.approx(0.526, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:mass",
        units="kg",
    ) == pytest.approx(306.7, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:wet_area",
        units="m**2",
    ) == pytest.approx(315.8, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_sink_base_temperature():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:operating_temperature",
        units="K",
        val=350.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHeatSinkBasedTemperature(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:base_temperature",
        units="K",
    ) == pytest.approx(340.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_base_temperature_difference():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:base_temperature",
        units="K",
        val=340.0,
    )
    ivc.add_output("exterior_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)

    problem = run_system(
        PerformancesFinBaseTemperatureDifference(
            number_of_points=NB_POINTS_TEST,
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            finned_heat_sink_id="finned_heat_sink_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:base_temperature_difference",
        units="K",
    ) == pytest.approx(40.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_air_reynolds_number():
    ivc = om.IndepVarComp()
    ivc.add_output("true_airspeed", units="m/s", val=93.0, shape=NB_POINTS_TEST)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_spacing",
        units="m",
        val=0.02088,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_length",
        units="m",
        val=3.0,
    )
    ivc.add_output("dynamic_viscosity", val=1.8e-5, units="Pa*s", shape=NB_POINTS_TEST)

    problem = run_system(
        PerformancesAirReynoldsNumber(
            number_of_points=NB_POINTS_TEST,
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            finned_heat_sink_id="finned_heat_sink_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "reynolds_number",
        units="unitless",
    ) == pytest.approx(750.85, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_air_nusselt_number():
    ivc = om.IndepVarComp()
    ivc.add_output("reynolds_number", units="unitless", val=750.85, shape=NB_POINTS_TEST)
    ivc.add_output("prandtl_number", units="unitless", val=0.71, shape=NB_POINTS_TEST)

    problem = run_system(
        PerformancesAirNusseltNumber(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val(
        "nusselt_number",
        units="unitless",
    ) == pytest.approx(17.3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_air_convection_heat_transfer_coefficient():
    ivc = om.IndepVarComp()
    ivc.add_output("nusselt_number", units="unitless", val=17.3)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_spacing",
        units="m",
        val=0.02088,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":air_conduction_coefficient",
        units="W/m/K",
        val=0.026,
    )

    problem = run_system(
        PerformancesAirConvectionHeatTransferCoefficient(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":convection_heat_transfer_coefficient",
        units="W/m**2/K",
    ) == pytest.approx(21.5, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fin_parameter():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_thickness",
        units="m",
        val=7.2e-4,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":convection_heat_transfer_coefficient",
        units="W/m**2/K",
        val=100.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_length",
        units="m",
        val=3.0,
    )

    problem = run_system(
        PerformancesFinParameter(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_parameter",
        units="m**-1",
    ) == pytest.approx(43.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fin_heat_transfer_parameter():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":convection_heat_transfer_coefficient",
        units="W/m**2/K",
        val=100.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_length",
        units="m",
        val=3.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:fin_thickness",
        units="m",
        val=7.2e-4,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:base_temperature_difference",
        units="K",
        val=40.0,
    )

    problem = run_system(
        PerformancesFinHeatTransferParameter(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_heat_transfer_parameter",
        units="W/K",
    ) == pytest.approx(88.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_design_dissipation_power():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:thermal_power_max",
        units="kW",
        val=600.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:power_max",
        units="kW",
        val=300.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:max_compressed_air_heat",
        units="kW",
        val=30.0,
    )

    problem = run_system(
        PerformancesDesignDissipationPower(
            pemfc_stack_bop_id="pemfc_stack_bop_1", finned_heat_sink_id="finned_heat_sink_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":design_dissipation_power",
        units="kW",
    ) == pytest.approx(330.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_perf_finned_heat_sink():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:operating_temperature",
        units="K",
        val=350.0,
    )
    ivc.add_output("altitude", units="m", val=0.0, shape=NB_POINTS_TEST)
    ivc.add_output("exterior_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)
    ivc.add_output("true_airspeed", units="m/s", val=93.0, shape=NB_POINTS_TEST)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_length",
        units="m",
        val=3.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_thickness",
        units="m",
        val=7.2e-4,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_spacing",
        units="m",
        val=0.02088,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_height",
        units="m",
        val=0.526,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":number_of_fins",
        units="unitless",
        val=100,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:thermal_power_max",
        units="kW",
        val=600.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:power_max",
        units="kW",
        val=300.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:max_compressed_air_heat",
        units="kW",
        val=30.0,
    )

    problem = run_system(
        PerformancesFinnedHeatSink(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            finned_heat_sink_id="finned_heat_sink_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:base_temperature",
        units="K",
    ) == pytest.approx(340.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1:base_temperature_difference",
        units="K",
    ) == pytest.approx(40.0, rel=1e-2)
    assert problem.get_val(
        "reynolds_number",
        units="unitless",
    ) == pytest.approx(750.85, rel=1e-2)
    assert problem.get_val(
        "nusselt_number",
        units="unitless",
    ) == pytest.approx(17.3, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":convection_heat_transfer_coefficient",
        units="W/m**2/K",
    ) == pytest.approx(21.9, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_parameter",
        units="m**-1",
    ) == pytest.approx(20.12, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":fin_heat_transfer_parameter",
        units="W/K",
    ) == pytest.approx(41.2, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:finned_heat_sink_1"
        ":design_dissipation_power",
        units="kW",
    ) == pytest.approx(330.0, rel=1e-2)

    problem.check_partials(compact_print=True)
