# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import pytest
import numpy as np
import openmdao.api as om
import os.path as pth

from ..perf_air_heat_capacity import PerformancesAirHeatCapacity
from ..perf_coolant_heat_capacity import PerformancesCoolantHeatCapacity
from ..perf_heat_capacity_ratio import PerformancesHeatCapacityRatio
from ..perf_heat_exchanger_NTU import PerformancesHeatExchangerNTU
from ..perf_heat_exchanger_UA import PerformancesHeatExchangerUA
from ..perf_mean_air_temperature import PerformancesMeanAirTemperature
from ..perf_air_mass_velocity import PerformancesAirMassVelocity
from ..perf_coolant_mass_velocity import PerformancesCoolantMassVelocity
from ..perf_air_reynold_number import PerformancesAirReynoldsNumber
from ..perf_coolant_reynold_number import PerformancesCoolantReynoldsNumber
from ..perf_fanning_friction_factor import PerformancesFanningFrictionFactor
from ..perf_pressure_drop_coefficient import PerformancesPressureDropCoefficient
from ..perf_air_pressure_drop import PerformancesAirPressureDrop
from ..perf_coolant_pressure_drop import PerformancesCoolantPressureDrop
from ..perf_heat_exchanger import PerformancesHeatExchanger

from ..sizing_fin_geometry import SizingHeatExchangerFinGeometry
from ..sizing_fin_geometry_factor import SizingHeatExchangerFinFactor
from ..sizing_fin_hydraulic_diameter import SizingHeatExchangerFinHydraulicDiameter
from ..sizing_heat_exchanger_separating_plate_layer_count import (
    SizingHeatExchangerSeparatingPlateLayerCount,
)
from ..sizing_heat_exchanger_no_flow_length import SizingHeatExchangerNoFlowLength
from ..sizing_heat_exchanger_flow_length import SizingHeatExchangerFlowLength
from ..sizing_total_transfer_area_volume_ratio import SizingTotalTransferAreaVolumeRatio
from ..sizing_free_flow_frontal_area_ratio import SizingFreeFlowFrontalAreaRatio
from ..sizing_heat_exchanger_plate_weight import SizingHeatExchangerPlateWeight
from ..sizing_heat_exchanger_channel_weight import SizingHeatExchangerChannelWeight
from ..sizing_heat_exchanger_coolant_volume import SizingHeatExchangerCoolantVolume
from ..sizing_heat_exchanger import SizingHeatExchanger

from tests.testing_utilities import run_system

NB_POINTS_TEST = 10


def test_air_heat_capacity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_rate",
        units="kg/h",
        val=2600.0,
    )
    ivc.add_output(
        "mean_air_specific_heat_capacity",
        units="J/kg/K",
        val=1005.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAirHeatCapacity(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val("air_heat_capacity", units="W/K") == pytest.approx(725.8, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_coolant_heat_capacity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mass_flow_rate",
        units="kg/s",
        val=4.1,
    )
    ivc.add_output(
        "mean_coolant_specific_heat_capacity",
        units="J/kg/K",
        val=3560.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCoolantHeatCapacity(pemfc_stack_bop_id="pemfc_stack_bop_1"),
        ivc,
    )

    assert problem.get_val("coolant_heat_capacity", units="W/K") == pytest.approx(14596.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_capacity_ratio():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "coolant_heat_capacity",
        units="W/K",
        val=14596.0,
    )
    ivc.add_output(
        "air_heat_capacity",
        units="W/K",
        val=725.8,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHeatCapacityRatio(),
        ivc,
    )

    assert problem.get_val("heat_capacity_ratio", units="unitless") == pytest.approx(
        0.0497, rel=1e-2
    )
    assert problem.get_val("min_heat_capacity", units="W/K") == pytest.approx(725.8, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_exchanger_NTU():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "heat_capacity_ratio",
        units="unitless",
        val=0.0497,
    )

    problem = om.Problem(reports=False)
    model = problem.model
    model.add_subsystem(
        name="ivc",
        subsys=ivc,
        promotes=["*"],
    )
    model.add_subsystem(
        name="ntu",
        subsys=PerformancesHeatExchangerNTU(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        promotes=["*"],
    )
    model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    model.nonlinear_solver.options["iprint"] = 0
    model.nonlinear_solver.options["maxiter"] = 200
    model.nonlinear_solver.options["rtol"] = 1e-5
    model.linear_solver = om.DirectSolver()

    problem.setup()
    problem.run_model()

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:NTU",
        units="unitless",
    ) == pytest.approx(4.218, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_exchanger_UA():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "min_heat_capacity",
        units="W/K",
        val=725.8,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:NTU",
        units="unitless",
        val=4.218,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHeatExchangerUA(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:UA",
        units="W/K",
    ) == pytest.approx(3058.1, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_mean_air_temperature():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("air_outlet_temperature", units="K", val=400.0)
    ivc.add_output("air_inlet_temperature", units="K", val=250.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMeanAirTemperature(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mean_air_temperature",
        units="K",
    ) == pytest.approx(325.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_air_mass_velocity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_rate",
        units="kg/h",
        val=2600.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_length",
        units="m",
        val=0.0907,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:free_flow_frontal_area_ratio",
        units="unitless",
        val=0.48,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAirMassVelocity(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "air_mass_velocity",
        units="kg/m**2/s",
    ) == pytest.approx(23.07, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_coolant_mass_velocity():
    # Research independent input value in .xml file

    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mass_flow_rate",
        units="kg/s",
        val=4.1,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.05,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:free_flow_frontal_area_ratio",
        units="unitless",
        val=0.48,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCoolantMassVelocity(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "coolant_mass_velocity",
        units="kg/m**2/s",
    ) == pytest.approx(237.6, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_air_reynold_number():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "air_mass_velocity",
        units="kg/m**2/s",
        val=23.07,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_hydraulic_diameter",
        units="m",
        val=1.92e-3,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mean_air_dynamic_viscosity",
        units="Pa*s",
        val=1.85e-5,
    )

    problem = run_system(
        PerformancesAirReynoldsNumber(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "air_reynolds_number",
        units="unitless",
    ) == pytest.approx(2394.3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_coolant_reynold_number():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "coolant_mass_velocity",
        units="kg/m**2/s",
        val=237.6,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_hydraulic_diameter",
        units="m",
        val=1.92e-3,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mean_coolant_dynamic_viscosity",
        units="Pa*s",
        val=0.00089,
    )

    problem = run_system(
        PerformancesCoolantReynoldsNumber(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "coolant_reynolds_number",
        units="unitless",
    ) == pytest.approx(512.6, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_fanning_friction_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "air_reynolds_number",
        units="unitless",
        val=87.18,
    )
    ivc.add_output(
        "coolant_reynolds_number",
        units="unitless",
        val=512.6,
    )

    problem = run_system(
        PerformancesFanningFrictionFactor(),
        ivc,
    )

    assert problem.get_val(
        "air_fanning_friction_factor",
        units="unitless",
    ) == pytest.approx(0.29, rel=1e-2)
    assert problem.get_val(
        "coolant_fanning_friction_factor",
        units="unitless",
    ) == pytest.approx(0.087, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pressure_drop_coefficient():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:free_flow_frontal_area_ratio",
        units="unitless",
        val=0.48,
    )

    problem = run_system(
        PerformancesPressureDropCoefficient(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "entrance_pressure_drop_coefficient",
        units="unitless",
    ) == pytest.approx(-0.5096, rel=1e-2)
    assert problem.get_val(
        "exit_pressure_drop_coefficient",
        units="unitless",
    ) == pytest.approx(1.04, rel=1e-2)


def test_air_pressure_drop():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("entrance_pressure_drop_coefficient", units="unitless", val=-0.5096)
    ivc.add_output("exit_pressure_drop_coefficient", units="unitless", val=1.04)
    ivc.add_output(
        "air_mass_velocity",
        units="kg/m**2/s",
        val=23.07,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:free_flow_frontal_area_ratio",
        units="unitless",
        val=0.48,
    )
    ivc.add_output(
        "air_fanning_friction_factor",
        units="unitless",
        val=0.29,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_hydraulic_diameter",
        units="m",
        val=1.92e-3,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_length",
        units="m",
        val=0.0907,
    )
    ivc.add_output("mean_air_density", units="kg/m**3", val=1.2)
    ivc.add_output("air_inlet_density", units="kg/m**3", val=1.19)
    ivc.add_output("air_outlet_density", units="kg/m**3", val=1.21)

    problem = run_system(
        PerformancesAirPressureDrop(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            heat_exchanger_id="heat_exchanger_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_pressure_drop",
        units="Pa",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 12262.23), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_coolant_pressure_drop():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("entrance_pressure_drop_coefficient", units="unitless", val=-0.5096)
    ivc.add_output("exit_pressure_drop_coefficient", units="unitless", val=1.04)
    ivc.add_output(
        "coolant_fanning_friction_factor",
        units="unitless",
        val=0.087,
    )
    ivc.add_output(
        "coolant_mass_velocity",
        units="kg/m**2/s",
        val=237.6,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_hydraulic_diameter",
        units="m",
        val=1.92e-3,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.05,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:free_flow_frontal_area_ratio",
        units="unitless",
        val=0.48,
    )
    ivc.add_output("mean_coolant_density", units="kg/m**3", val=1000.0)
    ivc.add_output("coolant_inlet_density", units="kg/m**3", val=980.0)
    ivc.add_output("coolant_outlet_density", units="kg/m**3", val=1020.0)

    problem = run_system(
        PerformancesCoolantPressureDrop(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_pressure_drop",
        units="Pa",
    ) == pytest.approx(268.52, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_heat_exchanger_performance():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("air_outlet_temperature", units="K", val=300.0)
    ivc.add_output("air_inlet_temperature", units="K", val=250.0)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_rate",
        units="kg/h",
        val=2600.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.05,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:no_flow_length",
        units="m",
        val=0.719,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:free_flow_frontal_area_ratio",
        units="unitless",
        val=0.48,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_length",
        units="m",
        val=0.0907,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mass_flow_rate",
        units="kg/s",
        val=4.1,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mean_temperature",
        units="K",
        val=348.2,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_hydraulic_diameter",
        units="m",
        val=1.92e-3,
    )
    ivc.add_output(
        "air_static_pressure",
        units="Pa",
        val=101325.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:static_pressure",
        units="Pa",
        val=200000.0,
    )
    ivc.add_output("coolant_outlet_temperature", units="K", val=340.0)
    ivc.add_output("coolant_inlet_temperature", units="K", val=350.0)

    problem = run_system(
        PerformancesHeatExchanger(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            heat_exchanger_id="heat_exchanger_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_pressure_drop",
        units="Pa",
    ) == pytest.approx(38804.12, rel=1e-2)

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_sizing_fin_geometry():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_length",
        units="m",
        val=1.02e-4,
    )

    problem = run_system(
        SizingHeatExchangerFinGeometry(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_height",
        units="m",
    ) == pytest.approx(6.25e-3, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_spacing",
        units="m",
    ) == pytest.approx(1.18e-3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_fin_geometry_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_height",
        units="m",
        val=6.25e-3,
    )

    problem = run_system(
        SizingHeatExchangerFinFactor(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_spacing_height_ratio",
        units="unitless ",
    ) == pytest.approx(0.189, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_thickness_length_ratio",
        units="unitless ",
    ) == pytest.approx(0.032, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_thickness_spacing_ratio",
        units="unitless ",
    ) == pytest.approx(0.0864, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_heat_exchanger_layer_count():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:layer_count",
        units="unitless",
        val=50.0,
    )

    problem = run_system(
        SizingHeatExchangerSeparatingPlateLayerCount(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val("air_layer_count", units="unitless") == pytest.approx(50.0, rel=1e-2)
    assert problem.get_val("coolant_layer_count", units="unitless") == pytest.approx(49.0, rel=1e-2)
    assert problem.get_val("separating_plate_count", units="unitless") == pytest.approx(
        98.0, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_sizing_fin_hydraulic_diameter():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_height",
        units="m",
        val=6.25e-3,
    )

    problem = run_system(
        SizingHeatExchangerFinHydraulicDiameter(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_hydraulic_diameter",
        units="m",
    ) == pytest.approx(1.92e-3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_heat_exchanger_no_flow_length():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "air_layer_count",
        units="unitless",
        val=50.0,
    )
    ivc.add_output("coolant_layer_count", units="unitless", val=49.0)
    ivc.add_output("separating_plate_count", units="unitless", val=98.0)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_height",
        units="m",
        val=6.25e-3,
    )

    problem = run_system(
        SizingHeatExchangerNoFlowLength(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:no_flow_length",
        units="m",
    ) == pytest.approx(0.719, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_total_transfer_area_volume_ratio():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:plate_spacing",
        units="m",
        val=6.35e-3,
    )

    problem = run_system(
        SizingTotalTransferAreaVolumeRatio(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:transfer_area_volume_ratio",
        units="1/m",
    ) == pytest.approx(1000.90, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_heat_exchanger_flow_length():
    ivc = om.IndepVarComp()

    # Required UA from upstream performance sizing
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:UA",
        units="W/K",
        val=3058.1,  # from test_heat_exchanger_UA result
    )
    # Fixed geometry
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:no_flow_length",
        units="m",
        val=0.719,  # from test_sizing_heat_exchanger_no_flow_length result
    )
    ivc.add_output(
        "separating_plate_count",
        units="unitless",
        val=98.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:transfer_area_volume_ratio",
        units="1/m",
        val=1000.9,
    )

    # Material properties
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:plate_thickness",
        units="m",
        val=8e-4,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:plate_thermally_conductivity",
        units="W/m/K",
        val=237.0,
    )
    # Fluid properties
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mean_coolant_dynamic_viscosity",
        units="Pa*s",
        val=1.75e-3,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mean_coolant_thermal_conductivity",
        units="W/m/K",
        val=0.27,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":mean_coolant_prandtl_number",
        units="unitless",
        val=10.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mean_air_dynamic_viscosity",
        units="Pa*s",
        val=1.83e-5,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":mean_air_thermal_conductivity",
        units="W/m/K",
        val=0.024,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":mean_air_prandtl_number",
        units="unitless",
        val=0.7,
    )
    # Operating conditions
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mass_flow_rate",
        units="kg/s",
        val=4.1,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_rate",
        units="kg/h",
        val=2600.0,  # convert from kg/h
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":fin_hydraulic_diameter",
        units="m",
        val=1.92e-3,
    )

    # Manual problem setup needed — run_system won't work with SubmodelComp + driver
    problem = om.Problem(reports=False)
    model = problem.model
    model.add_subsystem("ivc", ivc, promotes=["*"])
    model.add_subsystem(
        "flow_length",
        SizingHeatExchangerFlowLength(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        promotes=["*"],
    )

    problem.setup()
    problem.run_driver()  # NOT run_model — the inner SLSQP is a driver

    # Primary check: optimizer converged (UA_difference == 0)
    assert problem.get_val("UA_difference", units="W/K") == pytest.approx(0.0, abs=1e-3)

    # Sanity checks on outputs
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
    ) == pytest.approx(0.327, abs=1e-3)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_length",
        units="m",
    ) == pytest.approx(0.05, abs=1e-3)


def test_sizing_free_flow_frontal_area_ratio():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:fin_hydraulic_diameter",
        units="m",
        val=1.92e-3,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:transfer_area_volume_ratio",
        units="1/m",
        val=1000.9,
    )

    problem = run_system(
        SizingFreeFlowFrontalAreaRatio(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:free_flow_frontal_area_ratio",
        units="unitless",
    ) == pytest.approx(0.48, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_heat_exchanger_plate_weight():
    ivc = om.IndepVarComp()

    ivc.add_output("separating_plate_count", units="unitless", val=98.0)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_length",
        units="m",
        val=0.0907,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.05,
    )

    problem = run_system(
        SizingHeatExchangerPlateWeight(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":plate_mass",
        units="kg",
    ) == pytest.approx(1.13, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_heat_exchanger_channel_weight():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_length",
        units="m",
        val=0.0907,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.05,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:no_flow_length",
        units="m",
        val=0.719,
    )

    problem = run_system(
        SizingHeatExchangerChannelWeight(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:channel_mass",
        units="kg",
    ) == pytest.approx(0.0018, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_heat_exchanger_coolant_volume():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_length",
        units="m",
        val=0.0907,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
        val=0.05,
    )

    problem = run_system(
        SizingHeatExchangerCoolantVolume(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":coolant_volume",
        units="m**3",
    ) == pytest.approx(0.00139, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_heat_exchanger():
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:layer_count",
        units="unitless",
        val=50.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:UA",
        units="W/K",
        val=3058.1,  # from test_heat_exchanger_UA result
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:plate_thermally_conductivity",
        units="W/m/K",
        val=237.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mean_coolant_dynamic_viscosity",
        units="Pa*s",
        val=1.75e-3,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mean_coolant_thermal_conductivity",
        units="W/m/K",
        val=0.27,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":mean_coolant_prandtl_number",
        units="unitless",
        val=10.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mean_air_dynamic_viscosity",
        units="Pa*s",
        val=1.83e-5,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":mean_air_thermal_conductivity",
        units="W/m/K",
        val=0.024,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":mean_air_prandtl_number",
        units="unitless",
        val=0.7,
    )
    # Operating conditions
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mass_flow_rate",
        units="kg/s",
        val=4.1,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_rate",
        units="kg/h",
        val=2600.0,  # convert from kg/h
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mean_coolant_density",
        units="kg/m**3",
        val=1000.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mean_air_density",
        units="kg/m**3",
        val=1.225,
    )

    problem = om.Problem(reports=False)
    model = problem.model
    model.add_subsystem("ivc", ivc, promotes=["*"])
    model.add_subsystem(
        "sizing",
        SizingHeatExchanger(
            pemfc_stack_bop_id="pemfc_stack_bop_1", heat_exchanger_id="heat_exchanger_1"
        ),
        promotes=["*"],
    )

    problem.setup()
    problem.run_model()

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_flow_length",
        units="m",
    ) == pytest.approx(0.328, abs=1e-3)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_flow_length",
        units="m",
    ) == pytest.approx(0.05, abs=1e-3)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:mass",
        units="kg",
    ) == pytest.approx(4.09, rel=1e-2)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))
