# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import pytest

import fastoad.api as oad
from stdatm import Atmosphere

from tests.testing_utilities import get_indep_var_comp, list_inputs, run_system
from utils.write_outputs import write_outputs
from utils.filter_residuals import filter_residuals

from .simple_assembly.performances_simple_assembly_splitter import PerformancesAssemblySplitter

from ..assemblers.performances_from_pt_file import PowerTrainPerformancesFromFile
from ..assemblers.sizing_from_pt_file import PowerTrainSizingFromFile
from ..assemblers.delta_from_pt_file import AerodynamicDeltasFromPTFile

from . import outputs

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

XML_FILE = "simple_assembly_splitter.xml"
NB_POINTS_TEST = 10


def test_performances_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_simple_generator.yml")

    ivc = get_indep_var_comp(
        list_inputs(
            PowerTrainPerformancesFromFile(
                power_train_file_path=pt_file_path,
                number_of_points=NB_POINTS_TEST,
                pre_condition_pt=True,
            )
        ),
        __file__,
        XML_FILE,
    )

    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 500))

    problem = run_system(
        PowerTrainPerformancesFromFile(
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
            pre_condition_pt=True,
        ),
        ivc,
    )

    # om.n2(problem)

    _, _, residuals = problem.model.component.get_nonlinear_vectors()

    current_in = problem.get_val("component.dc_dc_converter_1.dc_current_out", units="A")
    voltage_in = problem.get_val("component.dc_dc_converter_1.dc_voltage_out", units="V")

    current_in_2 = problem.get_val("component.rectifier_1.dc_current_out", units="A")
    voltage_in_2 = problem.get_val("component.rectifier_1.dc_voltage_out", units="V")

    assert current_in * voltage_in == pytest.approx(
        current_in_2 * voltage_in_2,
        abs=1,
    )

    torque_generator = problem.get_val("component.generator_1.torque_in", units="N*m")
    omega_generator = (
        problem.get_val("component.generator_1.rpm", units="min**-1") * 2.0 * np.pi / 60.0
    )

    assert torque_generator * omega_generator == pytest.approx(
        np.array(
            [
                108181.60732714,
                108737.3269776,
                109283.1398736,
                109819.02208511,
                110344.94944601,
                110860.89757604,
                111366.841902,
                111862.7576782,
                112348.6200063,
                112824.40385453,
            ]
        ),
        rel=1e-3,
    )

    assert problem.get_val("component.ice_1.fuel_consumed_t", units="kg") == pytest.approx(
        np.array(
            [
                4.09026344,
                4.10237559,
                4.11429648,
                4.1260253,
                4.13756115,
                4.148903,
                4.16004973,
                4.17100016,
                4.18175298,
                4.19230684,
            ]
        ),
        abs=1e-2,
    )
    # Only one input and one output to the system so input and output should be equal
    assert problem.get_val(
        "component.fuel_system_1.fuel_consumed_in_t_1", units="kg"
    ) == pytest.approx(
        problem.get_val("component.ice_1.fuel_consumed_t", units="kg"),
        abs=1e-2,
    )
    assert problem.get_val("component.fuel_tank_1.fuel_remaining_t", units="kg") == pytest.approx(
        np.array(
            [
                41.42453467,
                37.33427122,
                33.23189563,
                29.11759916,
                24.99157385,
                20.85401271,
                16.70510971,
                12.54505998,
                8.37405982,
                4.19230684,
            ]
        ),
        abs=1e-2,
    )


def test_assembly_sizing_from_pt_file():
    oad.RegisterSubmodel.active_models["submodel.propulsion.constraints.pmsm.rpm"] = (
        "fastga_he.submodel.propulsion.constraints.pmsm.rpm.enforce"
    )

    pt_file_path = pth.join(DATA_FOLDER_PATH, "simple_assembly_simple_generator.yml")

    ivc = get_indep_var_comp(
        list_inputs(PowerTrainSizingFromFile(power_train_file_path=pt_file_path)),
        __file__,
        XML_FILE,
    )

    problem = run_system(PowerTrainSizingFromFile(power_train_file_path=pt_file_path), ivc)
    # Run problem and check obtained value(s) is/(are) correct
    problem.run_model()

    assert problem.get_val(
        "data:propulsion:he_power_train:propeller:propeller_1:mass", units="kg"
    ) == pytest.approx(36.35, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:mass", units="kg"
    ) == pytest.approx(16.19, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:inverter:inverter_1:mass", units="kg"
    ) == pytest.approx(19.95, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_bus:dc_bus_1:mass", units="kg"
    ) == pytest.approx(0.96, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_cable_harness:harness_1:mass", units="kg"
    ) == pytest.approx(22.31, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_DC_converter:dc_dc_converter_1:mass", units="kg"
    ) == pytest.approx(266.59, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:mass", units="kg"
    ) == pytest.approx(1500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_1:mass", units="kg"
    ) == pytest.approx(6.40, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_2:mass", units="kg"
    ) == pytest.approx(6.40, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_SSPC:dc_sspc_3:mass", units="kg"
    ) == pytest.approx(6.40, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier_1:mass", units="kg"
    ) == pytest.approx(28.93, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:turbo_generator:generator_1:mass", units="kg"
    ) == pytest.approx(21.63, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:DC_splitter:dc_splitter_1:mass", units="kg"
    ) == pytest.approx(0.623, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:mass", units="kg"
    ) == pytest.approx(302.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:mass", units="kg"
    ) == pytest.approx(0.4124, rel=1e-2)

    assert problem.get_val("data:propulsion:he_power_train:mass", units="kg") == pytest.approx(
        2260.00, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:CG:x", units="m") == pytest.approx(
        2.847, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:low_speed:CD0") == pytest.approx(
        0.00567915, rel=1e-2
    )
    assert problem.get_val("data:propulsion:he_power_train:cruise:CD0") == pytest.approx(
        0.00561219, rel=1e-2
    )

    write_outputs(
        pth.join(outputs.__path__[0], "assembly_sizing_from_pt_file.xml"),
        problem,
    )
