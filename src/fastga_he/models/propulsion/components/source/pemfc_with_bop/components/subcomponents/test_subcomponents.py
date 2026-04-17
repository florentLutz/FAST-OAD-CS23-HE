# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import pytest
import pathlib
import openmdao.api as om
import numpy as np

from fastga_he.models.cost.unit_tests.test_cost import XML_FILE
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from .fluid_characteristics.fluid_density import FluidDensity
from .fluid_characteristics.fluid_specific_heat_capacity import FluidSpecificHeatCapacity
from .fluid_characteristics.fluid_thermal_conductivity import FluidThermalConductivity
from .fluid_characteristics.fluid_dynamic_viscosity import FluidDynamicViscosity
from .fluid_characteristics.fluid_prandtl_number import FluidPrandtlNumber
from .fluid_characteristics.fluid_enthalpy import FluidEnthalpy
from .fluid_characteristics.fluid_specific_volume import FluidSpecificVolume

from .humidifier.perf_humidifier_air_pressure_drop import (
    PerformancesHumidifierRatingPressureDrop,
)
from .humidifier.perf_humidifier_oxidizer_temperature import (
    PerformancesHumidifierOxidizerTemperature,
)
from .humidifier.perf_humidifier import PerformancesHumidifier
from .humidifier.sizing_humidifier_volume import SizingHumidifierVolume
from .humidifier.sizing_humidifier_weight import SizingHumidifierWeight
from .humidifier.sizing_humidifier import SizingHumidifier

from .coolant_tank.sizing_coolant_tank import SizingCoolantTank

from .pipe.perf_pipe_reynolds_number import PerformancesPipeReynoldsNumber
from .pipe.perf_pipe_darcy_friction_factor import PerformancesPipeDarcyFrictionFactor
from .pipe.perf_pipe_coolant_pressure_drop import PerformancesPipeCoolantPressureDrop
from .pipe.sizing_pipe_wall_thickness import SizingPipeWallThickness
from .pipe.sizing_pipe import SizingPipe
from .pipe.perf_pipe import PerformancesPipe

from .pump.perf_pump_volumetric_flow_rate import PerformancesPumpVolumetricFlowRate
from .pump.perf_pump import PerformancesPump
from .pump.sizing_pump_weight import SizingPumpWeight

from .compressor.perf_compressor_pressure_ratio import PerformancesCompressorPressureRatio
from .compressor.perf_compressor_mean_pressure import PerformancesCompressorMeanPressure
from .compressor.perf_compressor_outlet_temperature import PerformancesCompressorOutletTemperature
from .compressor.perf_compressor_mean_temperature import PerformancesCompressorMeanTemperature
from .compressor.perf_compressor_pressure_target import PerformancesCompressorPressureTarget
from .compressor.perf_compressor_pressure_supply import PerformancesCompressorPressureSupply
from .compressor.perf_compressor_power_required import PerformancesCompressorPowerRequired
from .compressor.perf_compressor_power_rating import PerformancesCompressorPowerRating
from .compressor.perf_compressor import PerformancesCompressor
from .compressor.sizing_compressor_weight import SizingCompressorWeight

from .valve.sizing_valve import SizingValve

from .sizing_pemfc_bop import SizingPEMFCBOP
from .perf_pemfc_bop import PerformancesPEMFCBOP

NB_POINTS_TEST = 10
XML_FILE = "sample_pemfc_stack_with_bop.xml"
RESULTS_FOLDER_PATH = pathlib.Path(__file__).parent / "results"


def test_fluid_density():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidDensity(), ivc)
    assert problem.get_val("fluid_density", units="kg/m**3") == pytest.approx(1.177, rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)
    ivc.add_output("fluid_pressure", units="atm", val=1.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidDensity(number_of_points=NB_POINTS_TEST), ivc)
    assert problem.get_val("fluid_density", units="kg/m**3") == pytest.approx(
        np.full(NB_POINTS_TEST, 1.177), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=200.0, shape=NB_POINTS_TEST)
    ivc.add_output("fluid_pressure", units="atm", val=-1.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidDensity(number_of_points=NB_POINTS_TEST), ivc)
    assert problem.get_val("fluid_density", units="kg/m**3") == pytest.approx(
        np.full(NB_POINTS_TEST, 1.177), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_fluid_specific_heat_capacity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidSpecificHeatCapacity(), ivc)
    assert problem.get_val("fluid_specific_heat_capacity", units="J/kg/K") == pytest.approx(
        1006.4, rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)
    ivc.add_output("fluid_pressure", units="atm", val=1.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidSpecificHeatCapacity(number_of_points=NB_POINTS_TEST), ivc)
    assert problem.get_val("fluid_specific_heat_capacity", units="J/kg/K") == pytest.approx(
        np.full(NB_POINTS_TEST, 1006.4), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)
    ivc.add_output("fluid_pressure", units="atm", val=-1.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidSpecificHeatCapacity(number_of_points=NB_POINTS_TEST), ivc)
    assert problem.get_val("fluid_specific_heat_capacity", units="J/kg/K") == pytest.approx(
        np.full(NB_POINTS_TEST, 1006.4), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_fluid_thermal_conductivity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidThermalConductivity(), ivc)
    assert problem.get_val("fluid_thermal_conductivity", units="W/m/K") == pytest.approx(
        0.0264, rel=1e-2
    )

    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)
    ivc.add_output("fluid_pressure", units="atm", val=1.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidThermalConductivity(number_of_points=NB_POINTS_TEST), ivc)
    assert problem.get_val("fluid_thermal_conductivity", units="W/m/K") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.0264), rel=1e-2
    )


def test_fluid_dynamic_viscosity():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidDynamicViscosity(), ivc)
    assert problem.get_val("fluid_dynamic_viscosity", units="Pa*s") == pytest.approx(
        1.85e-5, rel=1e-2
    )

    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)
    ivc.add_output("fluid_pressure", units="atm", val=1.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidDynamicViscosity(number_of_points=NB_POINTS_TEST), ivc)
    assert problem.get_val("fluid_dynamic_viscosity", units="Pa*s") == pytest.approx(
        np.full(NB_POINTS_TEST, 1.85e-5), rel=1e-2
    )


def test_fluid_prandtl_number():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidPrandtlNumber(), ivc)
    assert problem.get_val("fluid_prandtl_number", units="unitless") == pytest.approx(
        0.71, rel=1e-2
    )

    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)
    ivc.add_output("fluid_pressure", units="atm", val=1.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidPrandtlNumber(number_of_points=NB_POINTS_TEST), ivc)
    assert problem.get_val("fluid_prandtl_number", units="unitless") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.71), rel=1e-2
    )


def test_fluid_enthalpy():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidEnthalpy(), ivc)
    assert problem.get_val("fluid_enthalpy", units="J/kg") == pytest.approx(426297.77, rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)
    ivc.add_output("fluid_pressure", units="atm", val=1.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidEnthalpy(number_of_points=NB_POINTS_TEST), ivc)
    assert problem.get_val("fluid_enthalpy", units="J/kg") == pytest.approx(
        np.full(NB_POINTS_TEST, 426297.77), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_fluid_specific_volume():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0)
    ivc.add_output("fluid_pressure", units="atm", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidSpecificVolume(), ivc)
    assert problem.get_val("fluid_specific_volume", units="m**3/kg") == pytest.approx(
        0.85, rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output("fluid_temperature", units="K", val=300.0, shape=NB_POINTS_TEST)
    ivc.add_output("fluid_pressure", units="atm", val=1.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FluidSpecificVolume(number_of_points=NB_POINTS_TEST), ivc)
    assert problem.get_val("fluid_specific_volume", units="m**3/kg") == pytest.approx(
        np.full(NB_POINTS_TEST, 0.85), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_performances_humidifier_rating_pressure_drop():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "humidifier_air_density",
        units="kg/m**3",
        val=1.177,
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "air_consumption",
        units="kg/h",
        val=2600.0,
        shape=NB_POINTS_TEST,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesHumidifierRatingPressureDrop(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            humidifier_id="humidifier_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:humidifier_1"
        ":air_pressure_drop",
        units="MPa",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 0.0887), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performances_humidifier_oxidizer_temperature():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:oxidizer_temperature",
        units="K",
        val=310.0,
    )

    problem = run_system(
        PerformancesHumidifierOxidizerTemperature(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "oxidizer_temperature",
        units="K",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 310.0), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performances_humidifier():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:operating_temperature",
        units="K",
        val=300.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:operating_pressure",
        units="atm",
        val=1.0,
    )
    ivc.add_output("air_consumption", units="kg/h", val=2600.0, shape=NB_POINTS_TEST)

    problem = run_system(
        PerformancesHumidifier(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            humidifier_id="humidifier_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:humidifier_1"
        ":air_pressure_drop",
        units="MPa",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 0.091), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_humidifier_weight():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:power_rating",
        units="kW",
        val=150.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHumidifierWeight(
            pemfc_stack_bop_id="pemfc_stack_bop_1", humidifier_id="humidifier_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:humidifier_1:mass",
        units="kg",
    ) == pytest.approx(12.44, rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:power_rating",
        units="kW",
        val=1200.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHumidifierWeight(
            pemfc_stack_bop_id="pemfc_stack_bop_1", humidifier_id="humidifier_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:humidifier_1:mass",
        units="kg",
    ) == pytest.approx(97.98, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_humidifier_volume():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:power_rating",
        units="kW",
        val=140.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHumidifierVolume(
            pemfc_stack_bop_id="pemfc_stack_bop_1", humidifier_id="humidifier_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:humidifier_1:volume",
        units="m**3",
    ) == pytest.approx(0.0152, rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:power_rating",
        units="kW",
        val=1200.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHumidifierVolume(
            pemfc_stack_bop_id="pemfc_stack_bop_1", humidifier_id="humidifier_1"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:humidifier_1:volume",
        units="m**3",
    ) == pytest.approx(0.124, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_humidifier():
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:power_rating",
        units="kW",
        val=140.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingHumidifier(pemfc_stack_bop_id="pemfc_stack_bop_1", humidifier_id="humidifier_1"), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:humidifier_1:volume",
        units="m**3",
    ) == pytest.approx(0.0152, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:humidifier_1:mass",
        units="kg",
    ) == pytest.approx(11.69, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_coolant_tank_sizing():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:coolant_volume",
        units="m**3",
        val=0.02,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:coolant_volume",
        units="m**3",
        val=0.005,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCoolantTank(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            coolant_tank_id="coolant_tank_1",
            coolant_component_ids=["pipe_1", "heat_exchanger_1"],
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1"
        ":coolant_tank_1:coolant_volume",
        units="m**3",
    ) == pytest.approx(0.025, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant_tank_1:volume",
        units="m**3",
    ) == pytest.approx(0.0274, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant_tank_1:mass",
        units="kg",
    ) == pytest.approx(6.585, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pipe_wall_thickness():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:static_pressure",
        units="Pa",
        val=200000.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:radius",
        units="m",
        val=0.03,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(SizingPipeWallThickness(pemfc_stack_bop_id="pemfc_stack_bop_1"), ivc)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:wall_thickness",
        units="m",
    ) == pytest.approx(0.000226, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_pipe():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:static_pressure",
        units="Pa",
        val=200000.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mass_flow_rate",
        units="kg/s",
        val=4.1,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mean_temperature",
        units="K",
        val=345.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:number_of_pipes",
        units="unitless",
        val=5,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingPipe(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            coolant_fluid_type="ethylene glycol",
            pipe_id="pipe_1",
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:radius",
        units="m",
    ) == pytest.approx(0.029, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:wall_thickness",
        units="m",
    ) == pytest.approx(0.00022, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:mass",
        units="kg",
    ) == pytest.approx(0.9, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pipe_reynolds_number():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:radius",
        units="m",
        val=0.01,
    )
    ivc.add_output(
        "coolant_density",
        units="kg/m**3",
        val=175.0,
    )
    ivc.add_output(
        "coolant_dynamic_viscosity",
        units="Pa*s",
        val=0.195e-3,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPipeReynoldsNumber(pemfc_stack_bop_id="pemfc_stack_bop_1", pipe_id="pipe_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:reynolds_number",
        units="unitless",
    ) == pytest.approx(26920.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pipe_darcy_friction_factor():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:radius",
        units="m",
        val=0.01,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:reynolds_number",
        units="unitless",
        val=1500.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPipeDarcyFrictionFactor(
            pemfc_stack_bop_id="pemfc_stack_bop_1", pipe_id="pipe_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1"
        ":darcy_friction_factor",
        units="unitless",
    ) == pytest.approx(0.043, rel=1e-2)

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:radius",
        units="m",
        val=0.01,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:reynolds_number",
        units="unitless",
        val=5000.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPipeDarcyFrictionFactor(
            pemfc_stack_bop_id="pemfc_stack_bop_1", pipe_id="pipe_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1"
        ":darcy_friction_factor",
        units="unitless",
    ) == pytest.approx(0.0378, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pipe_coolant_pressure_drop():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "coolant_density",
        units="kg/m**3",
        val=175.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:radius",
        units="m",
        val=0.01,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1"
        ":darcy_friction_factor",
        units="unitless",
        val=0.043,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:number_of_pipes",
        units="unitless",
        val=5,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPipeCoolantPressureDrop(
            pemfc_stack_bop_id="pemfc_stack_bop_1", pipe_id="pipe_1"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1"
        ":coolant_pressure_drop",
        units="Pa",
    ) == pytest.approx(1058.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pipe_performance():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:radius",
        units="m",
        val=0.01,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:static_pressure",
        units="Pa",
        val=200000.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mass_flow_rate",
        units="kg/s",
        val=4.1,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mean_temperature",
        units="K",
        val=345.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1:number_of_pipes",
        units="unitless",
        val=5,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPipe(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            coolant_fluid_type="ethylene glycol",
            pipe_id="pipe_1",
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1"
        ":darcy_friction_factor",
        units="unitless",
    ) == pytest.approx(0.0239, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1"
        ":coolant_pressure_drop",
        units="Pa",
    ) == pytest.approx(3465.83, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_valve_sizing():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mass_flow_rate",
        units="kg/s",
        val=4.1,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingValve(pemfc_stack_bop_id="pemfc_stack_bop_1", valve_id="valve_1"),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:valve_1:mass",
        units="kg",
    ) == pytest.approx(0.924, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:valve_1:volume",
        units="m**3",
    ) == pytest.approx(0.000838, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pump_volumetric_flow_rate():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mass_flow_rate",
        units="kg/s",
        val=4.1,
    )
    ivc.add_output(
        "coolant_density",
        units="kg/m**3",
        val=175.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPumpVolumetricFlowRate(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            pump_id="pump_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pump_1:volumetric_flow_rate",
        units="m**3/s",
    ) == pytest.approx(0.0281, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pump_performance():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mass_flow_rate",
        units="kg/s",
        val=4.1,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:mean_temperature",
        units="K",
        val=345.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:coolant:static_pressure",
        units="Pa",
        val=200000.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pipe_1"
        ":coolant_pressure_drop",
        units="Pa",
        val=1500.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1"
        ":coolant_pressure_drop",
        units="Pa",
        val=1500.0,
    )

    problem = run_system(
        PerformancesPump(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            pump_id="pump_1",
            coolant_fluid_type="ethylene glycol",
            coolant_component_ids=["pipe_1", "heat_exchanger_1"],
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pump_1:volumetric_flow_rate",
        units="m**3/s",
    ) == pytest.approx(0.00477, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pump_1:pressure_compensation",
        units="Pa",
    ) == pytest.approx(103000.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pump_1:power_rating",
        units="W",
    ) == pytest.approx(839.2, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_pump_weight():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pump_1:power_rating",
        units="W",
        val=839.2,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingPumpWeight(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            pump_id="pump_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pump_1:mass",
        units="kg",
    ) == pytest.approx(11.24, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_compressor_performances_pressure_ratio():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("compressor_pressure_supply", units="Pa", val=204587.23, shape=NB_POINTS_TEST)
    ivc.add_output("ambient_pressure", units="Pa", val=101325.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCompressorPressureRatio(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("compressor_pressure_ratio", units="unitless") == pytest.approx(
        np.full(NB_POINTS_TEST, 2.0), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_compressor_performances_mean_pressure():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("compressor_pressure_supply", units="Pa", val=204587.23, shape=NB_POINTS_TEST)
    ivc.add_output("ambient_pressure", units="Pa", val=101325.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCompressorMeanPressure(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("mean_compressor_pressure", units="Pa") == pytest.approx(
        np.full(NB_POINTS_TEST, 151988.0), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_compressor_performances_outlet_temperature():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("exterior_temperature", units="K", val=288.15, shape=NB_POINTS_TEST)
    ivc.add_output("compressor_pressure_ratio", units="unitless", val=2.0, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCompressorOutletTemperature(
            number_of_points=NB_POINTS_TEST,
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            compressor_id="compressor_1",
        ),
        ivc,
    )

    assert problem.get_val("compressor_outlet_temperature", units="K") == pytest.approx(
        np.full(NB_POINTS_TEST, 362.4), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_compressor_performances_mean_temperature():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("exterior_temperature", units="K", val=288.15, shape=NB_POINTS_TEST)
    ivc.add_output("compressor_outlet_temperature", units="K", val=362.4, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCompressorMeanTemperature(
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("mean_compressor_temperature", units="K") == pytest.approx(
        np.full(NB_POINTS_TEST, 325.3), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_compressor_performances_pressure_target():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:operating_pressure",
        units="atm",
        val=1.0,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCompressorPressureTarget(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val("compressor_pressure_target", units="Pa") == pytest.approx(
        101325.0, rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_compressor_performances_pressure_supply():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("compressor_pressure_target", units="Pa", val=101325.0, shape=NB_POINTS_TEST)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:humidifier_1:air_pressure_drop",
        units="Pa",
        val=91000.0,
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_pressure_drop",
        units="Pa",
        val=12262.23,
        shape=NB_POINTS_TEST,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCompressorPressureSupply(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            number_of_points=NB_POINTS_TEST,
            connected_humidifier_id="humidifier_1",
            connected_heat_exchanger_id="heat_exchanger_1",
        ),
        ivc,
    )

    assert problem.get_val("compressor_pressure_supply", units="Pa") == pytest.approx(
        np.full(NB_POINTS_TEST, 204587.23), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_compressor_performances_power_required():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("compressor_pressure_ratio", units="unitless", val=2.0, shape=NB_POINTS_TEST)
    ivc.add_output("exterior_temperature", units="K", val=288.15, shape=NB_POINTS_TEST)
    ivc.add_output(
        "compressed_air_specific_heat_capacity", units="J/kg/K", val=1005.0, shape=NB_POINTS_TEST
    )
    ivc.add_output("air_consumption", units="kg/s", val=0.72, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCompressorPowerRequired(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            compressor_id="compressor_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:compressor_1:power_required",
        units="W",
    ) == pytest.approx(np.full(NB_POINTS_TEST, 53724.14), rel=1e-2)

    problem.check_partials(compact_print=True)


def test_compressor_performances_power_rating():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:compressor_1:power_required",
        units="W",
        val=50123.24,
        shape=NB_POINTS_TEST,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesCompressorPowerRating(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            compressor_id="compressor_1",
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:compressor_1:power_rating",
        units="W",
    ) == pytest.approx(50123.24, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_compressor_performances():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output("mach", units="unitless", val=0.3, shape=NB_POINTS_TEST)
    ivc.add_output("exterior_temperature", units="K", val=288.15, shape=NB_POINTS_TEST)
    ivc.add_output("air_consumption", units="kg/s", val=0.72, shape=NB_POINTS_TEST)
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:operating_pressure",
        units="atm",
        val=1.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:humidifier_1:air_pressure_drop",
        units="Pa",
        val=91000.0,
        shape=NB_POINTS_TEST,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:heat_exchanger_1:air_pressure_drop",
        units="Pa",
        val=12262.23,
        shape=NB_POINTS_TEST,
    )

    problem = run_system(
        PerformancesCompressor(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            compressor_id="compressor_1",
            number_of_points=NB_POINTS_TEST,
            connected_humidifier_id="humidifier_1",
            connected_heat_exchanger_id="heat_exchanger_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:compressor_1:power_rating",
        units="W",
    ) == pytest.approx(54714.5, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_sizing_compressor_weight():
    # Research independent input value in .xml file
    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:compressor_1:power_rating",
        units="W",
        val=54714.5,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingCompressorWeight(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            compressor_id="compressor_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:compressor_1:mass",
        units="kg",
    ) == pytest.approx(7.37, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_bop_sizing():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            SizingPEMFCBOP(
                pemfc_stack_bop_id="pemfc_stack_bop_1",
                coolant_fluid_type="ethylene glycol",
                compressor_id="compressor_1",
                pump_id="pump_1",
                valve_id="valve_1",
                pipe_id="pipe_1",
                humidifier_id="humidifier_1",
                primary_heat_exchanger_id="heat_exchanger_1",
                supplement_heat_exchanger_id="heat_exchanger_2",
                nozzle_id="nozzle_1",
                diffuser_id="diffuser_1",
                air_inlet_id="air_inlet_1",
                coolant_tank_id="coolant_tank_1",
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        SizingPEMFCBOP(
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            coolant_fluid_type="ethylene glycol",
            compressor_id="compressor_1",
            pump_id="pump_1",
            valve_id="valve_1",
            pipe_id="pipe_1",
            humidifier_id="humidifier_1",
            primary_heat_exchanger_id="heat_exchanger_1",
            supplement_heat_exchanger_id="heat_exchanger_2",
            coolant_tank_id="coolant_tank_1",
            nozzle_id="nozzle_1",
            diffuser_id="diffuser_1",
            air_inlet_id="air_inlet_1",
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:bop_mass",
        units="kg",
    ) == pytest.approx(82.92, rel=1e-2)

    problem.output_file_path = RESULTS_FOLDER_PATH / "test_bop_sizing_outputs.xml"

    problem.check_partials(compact_print=True)

    problem.write_outputs()

    n2_path = RESULTS_FOLDER_PATH / "n2_bop_sizing.html"

    om.n2(problem, show_browser=False, outfile=n2_path)


def test_bop_performances():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesPEMFCBOP(
                number_of_points=NB_POINTS_TEST,
                pemfc_stack_bop_id="pemfc_stack_bop_1",
                compressor_id="compressor_1",
                pump_id="pump_1",
                air_inlet_id="air_inlet_1",
                primary_heat_exchanger_id="heat_exchanger_1",
                supplement_heat_exchanger_id="heat_exchanger_2",
                humidifier_id="humidifier_1",
                pipe_id="pipe_1",
                diffuser_id="diffuser_1",
                nozzle_id="nozzle_1",
                coolant_fluid_type="ethylene glycol",
            )
        ),
        __file__,
        "test_bop_perf_inputs.xml",
    )
    ivc.add_output("exterior_temperature", units="K", val=288.15, shape=NB_POINTS_TEST)
    ivc.add_output("altitude", units="m", val=1000.0, shape=NB_POINTS_TEST)
    ivc.add_output("true_airspeed", units="m/s", val=100.0, shape=NB_POINTS_TEST)
    ivc.add_output("density", units="kg/m**3", val=1.112, shape=NB_POINTS_TEST)
    ivc.add_output("air_consumption", units="kg/s", val=0.72, shape=NB_POINTS_TEST)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesPEMFCBOP(
            number_of_points=NB_POINTS_TEST,
            pemfc_stack_bop_id="pemfc_stack_bop_1",
            compressor_id="compressor_1",
            pump_id="pump_1",
            air_inlet_id="air_inlet_1",
            primary_heat_exchanger_id="heat_exchanger_1",
            supplement_heat_exchanger_id="heat_exchanger_2",
            humidifier_id="humidifier_1",
            pipe_id="pipe_1",
            diffuser_id="diffuser_1",
            nozzle_id="nozzle_1",
            coolant_fluid_type="ethylene glycol",
        ),
        ivc,
    )

    # assert problem.get_val(
    #     "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:pump_1"
    #     ":pressure_compensation",
    #     units="kPa",
    # ) == pytest.approx(100.87, rel=1e-2)
    # assert problem.get_val(
    #     "data:propulsion:he_power_train:PEMFC_stack_bop:pemfc_stack_bop_1:compressor_1:power_rating",
    #     units="kW",
    # ) == pytest.approx(85.7, rel=1e-2)

    problem.check_partials(compact_print=True)

    problem.output_file_path = RESULTS_FOLDER_PATH / "test_bop_performance_outputs.xml"

    problem.write_outputs()

    n2_path = RESULTS_FOLDER_PATH / "n2_bop_performance.html"

    om.n2(problem, show_browser=False, outfile=n2_path)
