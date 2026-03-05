# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import pytest
import numpy as np
import openmdao.api as om

from ..perf_air_heat_capacity import PerformancesAirHeatCapacity
from ..perf_coolant_heat_capacity import PerformancesCoolantHeatCapacity
from ..perf_heat_capacity_ratio import PerformancesHeatCapacityRatio
from ..perf_heat_exchanger_NTU import PerformancesHeatExchangerNTU
from ..perf_heat_exchanger_UA import PerformancesHeatExchangerUA
from ..perf_mean_air_temperature import PerformancesMeanAirTemperature

from ..sizing_fin_geometry import SizingHeatExchangerFinGeometry
from ..sizing_fin_geometry_factor import SizingHeatExchangerFinFactor
from ..sizing_fin_hydraulic_diameter import SizingHeatExchangerFinHydraulicDiameter
from ..sizing_heat_exchanger_separating_plate_layer_count import (
    SizingHeatExchangerSeparatingPlateLayerCount,
)
from ..sizing_heat_exchanger_no_flow_length import SizingHeatExchangerNoFlowLength
from ..sizing_heat_exchanger_flow_length import SizingHeatExchangerFlowLength
from ..sizing_total_transfer_area_volume_ratio import SizingTotalTransferAreaVolumeRatio

from tests.testing_utilities import run_system

NB_POINTS_TEST = 10


def test_air_heat_capacity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:air_flow_rate",
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
        PerformancesAirHeatCapacity(pemfc_stack_bop_id="pemfc_stack_bop_1"),
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
        subsys=PerformancesHeatExchangerNTU(pemfc_stack_bop_id="pemfc_stack_bop_1"),
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
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:NTU",
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
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:NTU",
        units="unitless",
        val=4.218,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHeatExchangerUA(pemfc_stack_bop_id="pemfc_stack_bop_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1" ":heat_exchanger:UA",
        units="W/K",
    ) == pytest.approx(3058.1, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_mean_air_temperature():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:air_temperature_out",
        units="K",
        val=400.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger"
        ":air_temperature_in",
        units="K",
        val=250.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMeanAirTemperature(pemfc_stack_bop_id="pemfc_stack_bop_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1"
        ":heat_exchanger:mean_air_temperature",
        units="K",
    ) == pytest.approx(325.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_fin_geometry():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:fin_length",
        units="m",
        val=1.02e-4,
    )

    problem = run_system(
        SizingHeatExchangerFinGeometry(pemfc_stack_bop_id="pemfc_stack_bop_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:fin_height",
        units="m",
    ) == pytest.approx(6.25e-3, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:fin_spacing",
        units="m",
    ) == pytest.approx(1.18e-3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_fin_geometry_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:fin_height",
        units="m",
        val=6.25e-3,
    )

    problem = run_system(
        SizingHeatExchangerFinFactor(pemfc_stack_bop_id="pemfc_stack_bop_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:fin_spacing_height_ratio",
        units="unitless ",
    ) == pytest.approx(0.189, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:fin_thickness_length_ratio",
        units="unitless ",
    ) == pytest.approx(0.032, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:fin_thickness_spacing_ratio",
        units="unitless ",
    ) == pytest.approx(0.0864, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_heat_exchanger_layer_count():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:layer_count",
        units="unitless",
        val=50.0,
    )

    problem = run_system(
        SizingHeatExchangerSeparatingPlateLayerCount(pemfc_stack_bop_id="pemfc_stack_bop_1"),
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
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:fin_height",
        units="m",
        val=6.25e-3,
    )

    problem = run_system(
        SizingHeatExchangerFinHydraulicDiameter(pemfc_stack_bop_id="pemfc_stack_bop_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:fin_hydraulic_diameter",
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
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1"
        ":heat_exchanger:fin_height",
        units="m",
        val=6.25e-3,
    )

    problem = run_system(
        SizingHeatExchangerNoFlowLength(pemfc_stack_bop_id="pemfc_stack_bop_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:no_flow_length",
        units="m",
    ) == pytest.approx(0.719, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_total_transfer_area_volume_ratio():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:plate_spacing",
        units="m",
        val=6.35e-3,
    )

    problem = run_system(
        SizingTotalTransferAreaVolumeRatio(pemfc_stack_bop_id="pemfc_stack_bop_1"),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:transfer_area_volume_ratio",
        units="1/m",
    ) == pytest.approx(1000.90, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_heat_exchanger_flow_length():
    ivc = om.IndepVarComp()

    # Required UA from upstream performance sizing
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:UA",
        units="W/K",
        val=3058.1,  # from test_heat_exchanger_UA result
    )
    # Fixed geometry
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:no_flow_length",
        units="m",
        val=0.719,  # from test_sizing_heat_exchanger_no_flow_length result
    )
    ivc.add_output(
        "separating_plate_count",
        units="unitless",
        val=98.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:transfer_area_volume_ratio",
        units="1/m",
        val=1000.9,
    )

    # Material properties
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:plate_thickness",
        units="m",
        val=8e-4,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:plate_thermally_conductivity",
        units="W/m/K",
        val=237.0,
    )
    # Fluid properties
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:mean_coolant_dynamic_viscosity",
        units="Pa*s",
        val=1.75e-3,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:mean_coolant_thermal_conductivity",
        units="W/m/K",
        val=0.27,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger"
        ":mean_coolant_prandtl_number",
        units="unitless",
        val=10.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:mean_air_dynamic_viscosity",
        units="Pa*s",
        val=1.83e-5,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger"
        ":mean_air_thermal_conductivity",
        units="W/m/K",
        val=0.024,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger"
        ":mean_air_prandtl_number",
        units="unitless",
        val=1006.0,
    )
    # Operating conditions
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mass_flow_rate",
        units="kg/s",
        val=4.1,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:air_flow_rate",
        units="kg/h",
        val=2600.0,  # convert from kg/h
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger"
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
        SizingHeatExchangerFlowLength(pemfc_stack_bop_id="pemfc_stack_bop_1"),
        promotes=["*"],
    )

    problem.setup()
    problem.run_driver()  # NOT run_model — the inner SLSQP is a driver

    # Primary check: optimizer converged (UA_difference == 0)
    assert problem.get_val("UA_difference", units="W/K") == pytest.approx(0.0, abs=1e-3)

    # Sanity checks on outputs
    assert (
        0.05
        <= problem.get_val(
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:coolant_flow_length",
            units="m",
        )
        <= 0.5
    )
    assert (
        0.05
        <= problem.get_val(
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:air_flow_length",
            units="m",
        )
        <= 0.5
    )
    assert problem.get_val("HEX_volume", units="m**3") > 0.0

    print(problem.get_val(
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:air_flow_length",
            units="m",
        ))
    print(problem.get_val(
            "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger:coolant_flow_length",
            units="m",
        ))
