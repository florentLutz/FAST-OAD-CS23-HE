# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os.path as pth
import fastoad.api as oad
import numpy as np
import copy
import openmdao.api as om
import pytest
from stdatm import Atmosphere

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from utils.filter_residuals import filter_residuals

from fastga_he.models.propulsion.assemblers.performances_from_pt_file import (
    PowerTrainPerformancesFromFile,
)
from fastga_he.models.propulsion.assemblers.delta_from_pt_file import AerodynamicDeltasFromPTFile

from ..components.loads.pmsm import PerformancesPMSM
from ..components.propulsor.propeller import PerformancesPropeller
from ..components.connectors.inverter import PerformancesInverter
from ..components.connectors.dc_cable import PerformancesHarness
from ..components.connectors.dc_bus import PerformancesDCBus
from ..components.connectors.dc_dc_converter import PerformancesDCDCConverter
from ..components.source.battery import PerformancesBatteryPack

from ..assemblers.thrust_distributor import ThrustDistributor
from ..assemblers.power_rate import PowerRate


DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")

XML_FILE = "quad_assembly.xml"
NB_POINTS_TEST = 50
COEFF_DIFF = 0.0



class PerformancesAssembly(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["maxiter"] = 200
        self.nonlinear_solver.options["rtol"] = 1e-4
        self.linear_solver = om.DirectSolver()

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        # Propellers
        self.add_subsystem(
            "propeller_1",
            PerformancesPropeller(propeller_id="propeller_1", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*", "thrust", "density"],
        )
        self.add_subsystem(
            "propeller_2",
            PerformancesPropeller(propeller_id="propeller_2", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*", "thrust", "density"],
        )
        self.add_subsystem(
            "propeller_3",
            PerformancesPropeller(propeller_id="propeller_3", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*", "thrust", "density"],
        )
        self.add_subsystem(
            "propeller_4",
            PerformancesPropeller(propeller_id="propeller_4", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*", "thrust", "density"],
        )

        # Motors
        self.add_subsystem(
            "motor_1",
            PerformancesPMSM(motor_id="motor_1", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "motor_2",
            PerformancesPMSM(motor_id="motor_2", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "motor_3",
            PerformancesPMSM(motor_id="motor_3", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "motor_4",
            PerformancesPMSM(motor_id="motor_4", number_of_points=number_of_points),
            promotes=["data:*"],
        )

        # Inverters
        self.add_subsystem(
            "inverter_1",
            PerformancesInverter(inverter_id="inverter_1", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "inverter_2",
            PerformancesInverter(inverter_id="inverter_2", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "inverter_3",
            PerformancesInverter(inverter_id="inverter_3", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "inverter_4",
            PerformancesInverter(inverter_id="inverter_4", number_of_points=number_of_points),
            promotes=["data:*"],
        )

        # DC Buses
        self.add_subsystem(
            "dc_bus_1",
            PerformancesDCBus(
                dc_bus_id="dc_bus_1",
                number_of_points=number_of_points,
                number_of_inputs=2,
                number_of_outputs=1,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_bus_2",
            PerformancesDCBus(
                dc_bus_id="dc_bus_2",
                number_of_points=number_of_points,
                number_of_inputs=1,
                number_of_outputs=2,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_bus_3",
            PerformancesDCBus(
                dc_bus_id="dc_bus_3",
                number_of_points=number_of_points,
                number_of_inputs=1,
                number_of_outputs=2,
            ),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "dc_bus_4",
            PerformancesDCBus(
                dc_bus_id="dc_bus_4",
                number_of_points=number_of_points,
                number_of_inputs=2,
                number_of_outputs=1,
            ),
            promotes=["data:*"],
        )

        # DC lines
        self.add_subsystem(
            "dc_line_1",
            PerformancesHarness(
                harness_id="harness_1",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "exterior_temperature"],
        )
        self.add_subsystem(
            "dc_line_2",
            PerformancesHarness(
                harness_id="harness_2",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "exterior_temperature"],
        )
        self.add_subsystem(
            "dc_line_3",
            PerformancesHarness(
                harness_id="harness_3",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "exterior_temperature"],
        )
        self.add_subsystem(
            "dc_line_4",
            PerformancesHarness(
                harness_id="harness_4",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "exterior_temperature"],
        )
        self.add_subsystem(
            "dc_line_5",
            PerformancesHarness(
                harness_id="harness_5",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "exterior_temperature"],
        )
        self.add_subsystem(
            "dc_line_6",
            PerformancesHarness(
                harness_id="harness_6",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "exterior_temperature"],
        )

        # Source bus
        self.add_subsystem(
            "dc_bus_5",
            PerformancesDCBus(
                dc_bus_id="dc_bus_5",
                number_of_points=number_of_points,
                number_of_inputs=1,
                number_of_outputs=4,
            ),
            promotes=["data:*"],
        )

        # Source converter
        self.add_subsystem(
            "dc_dc_converter_1",
            PerformancesDCDCConverter(
                dc_dc_converter_id="dc_dc_converter_1",
                number_of_points=number_of_points,
            ),
            promotes=["data:*"],
        )

        self.add_subsystem(
            "battery_pack_1",
            PerformancesBatteryPack(
                battery_pack_id="battery_pack_1",
                number_of_points=number_of_points,
            ),
            promotes=["data:*", "time_step"],
        )

        self.connect("propeller_1.rpm", "motor_1.rpm")
        self.connect("propeller_2.rpm", "motor_2.rpm")
        self.connect("propeller_3.rpm", "motor_3.rpm")
        self.connect("propeller_4.rpm", "motor_4.rpm")

        self.connect("propeller_1.shaft_power_in", "motor_1.shaft_power_out")
        self.connect("propeller_2.shaft_power_in", "motor_2.shaft_power_out")
        self.connect("propeller_3.shaft_power_in", "motor_3.shaft_power_out")
        self.connect("propeller_4.shaft_power_in", "motor_4.shaft_power_out")

        self.connect(
            "motor_1.ac_current_rms_in_one_phase", "inverter_1.ac_current_rms_out_one_phase"
        )
        self.connect(
            "motor_2.ac_current_rms_in_one_phase", "inverter_2.ac_current_rms_out_one_phase"
        )
        self.connect(
            "motor_3.ac_current_rms_in_one_phase", "inverter_3.ac_current_rms_out_one_phase"
        )
        self.connect(
            "motor_4.ac_current_rms_in_one_phase", "inverter_4.ac_current_rms_out_one_phase"
        )

        self.connect("motor_1.ac_voltage_peak_in", "inverter_1.ac_voltage_peak_out")
        self.connect("motor_2.ac_voltage_peak_in", "inverter_2.ac_voltage_peak_out")
        self.connect("motor_3.ac_voltage_peak_in", "inverter_3.ac_voltage_peak_out")
        self.connect("motor_4.ac_voltage_peak_in", "inverter_4.ac_voltage_peak_out")

        self.connect("motor_1.ac_voltage_rms_in", "inverter_1.ac_voltage_rms_out")
        self.connect("motor_2.ac_voltage_rms_in", "inverter_2.ac_voltage_rms_out")
        self.connect("motor_3.ac_voltage_rms_in", "inverter_3.ac_voltage_rms_out")
        self.connect("motor_4.ac_voltage_rms_in", "inverter_4.ac_voltage_rms_out")

        self.connect("dc_bus_1.dc_voltage", "inverter_1.dc_voltage_in")
        self.connect("dc_bus_2.dc_voltage", "inverter_2.dc_voltage_in")
        self.connect("dc_bus_3.dc_voltage", "inverter_3.dc_voltage_in")
        self.connect("dc_bus_4.dc_voltage", "inverter_4.dc_voltage_in")

        self.connect("inverter_1.dc_current_in", "dc_bus_1.dc_current_out_1")
        self.connect("inverter_2.dc_current_in", "dc_bus_2.dc_current_out_1")
        self.connect("inverter_3.dc_current_in", "dc_bus_3.dc_current_out_1")
        self.connect("inverter_4.dc_current_in", "dc_bus_4.dc_current_out_1")

        # DC bus 1
        self.connect("dc_bus_1.dc_voltage", "dc_line_1.dc_voltage_out")
        self.connect("dc_bus_1.dc_voltage", "dc_line_5.dc_voltage_out")
        self.connect("dc_line_1.dc_current", "dc_bus_1.dc_current_in_1")
        self.connect("dc_line_5.dc_current", "dc_bus_1.dc_current_in_2")

        # DC bus 2
        self.connect("dc_bus_2.dc_voltage", "dc_line_2.dc_voltage_out")
        self.connect("dc_bus_2.dc_voltage", "dc_line_5.dc_voltage_in")
        self.connect("dc_line_2.dc_current", "dc_bus_2.dc_current_in_1")
        self.connect("dc_line_5.dc_current", "dc_bus_2.dc_current_out_2")

        # DC bus 3
        self.connect("dc_bus_3.dc_voltage", "dc_line_3.dc_voltage_out")
        self.connect("dc_bus_3.dc_voltage", "dc_line_6.dc_voltage_in")
        self.connect("dc_line_3.dc_current", "dc_bus_3.dc_current_in_1")
        self.connect("dc_line_6.dc_current", "dc_bus_3.dc_current_out_2")

        # DC bus 4
        self.connect("dc_bus_4.dc_voltage", "dc_line_4.dc_voltage_out")
        self.connect("dc_bus_4.dc_voltage", "dc_line_6.dc_voltage_out")
        self.connect("dc_line_4.dc_current", "dc_bus_4.dc_current_in_1")
        self.connect("dc_line_6.dc_current", "dc_bus_4.dc_current_in_2")

        self.connect("dc_bus_5.dc_voltage", "dc_line_1.dc_voltage_in")
        self.connect("dc_bus_5.dc_voltage", "dc_line_2.dc_voltage_in")
        self.connect("dc_bus_5.dc_voltage", "dc_line_3.dc_voltage_in")
        self.connect("dc_bus_5.dc_voltage", "dc_line_4.dc_voltage_in")

        self.connect("dc_line_1.dc_current", "dc_bus_5.dc_current_out_1")
        self.connect("dc_line_2.dc_current", "dc_bus_5.dc_current_out_2")
        self.connect("dc_line_3.dc_current", "dc_bus_5.dc_current_out_3")
        self.connect("dc_line_4.dc_current", "dc_bus_5.dc_current_out_4")

        self.connect("dc_dc_converter_1.dc_current_out", "dc_bus_5.dc_current_in_1")
        self.connect("dc_bus_5.dc_voltage", "dc_dc_converter_1.dc_voltage_out")

        self.connect("battery_pack_1.voltage_out", "dc_dc_converter_1.dc_voltage_in")
        self.connect("dc_dc_converter_1.dc_current_in", "battery_pack_1.dc_current_out")


def test_assembly():

    ivc = get_indep_var_comp(
        list_inputs(PerformancesAssembly(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(500, 550, NB_POINTS_TEST), units="N")
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 50))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAssembly(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    print("\n=========== Propulsive power ===========")
    print(
        problem.get_val("true_airspeed", units="m/s")[0]
        * problem.get_val("component.propeller_1.thrust", units="N")[0]
    )
    print(
        problem.get_val("true_airspeed", units="m/s")[0]
        * problem.get_val("component.propeller_2.thrust", units="N")[0]
    )
    print(
        problem.get_val("true_airspeed", units="m/s")[0]
        * problem.get_val("component.propeller_3.thrust", units="N")[0]
    )
    print(
        problem.get_val("true_airspeed", units="m/s")[0]
        * problem.get_val("component.propeller_4.thrust", units="N")[0]
    )

    print("\n=========== Shaft power ===========")
    print(problem.get_val("component.propeller_1.shaft_power_in", units="W")[0])
    print(problem.get_val("component.propeller_2.shaft_power_in", units="W")[0])
    print(problem.get_val("component.propeller_3.shaft_power_in", units="W")[0])
    print(problem.get_val("component.propeller_4.shaft_power_in", units="W")[0])

    print("\n=========== AC power ===========")
    print(
        problem.get_val("component.motor_1.ac_current_rms_in", units="A")[0]
        * problem.get_val("component.motor_1.ac_voltage_rms_in", units="V")[0]
    )
    print(
        problem.get_val("component.motor_2.ac_current_rms_in", units="A")[0]
        * problem.get_val("component.motor_2.ac_voltage_rms_in", units="V")[0]
    )
    print(
        problem.get_val("component.motor_3.ac_current_rms_in", units="A")[0]
        * problem.get_val("component.motor_3.ac_voltage_rms_in", units="V")[0]
    )
    print(
        problem.get_val("component.motor_4.ac_current_rms_in", units="A")[0]
        * problem.get_val("component.motor_4.ac_voltage_rms_in", units="V")[0]
    )

    print("\n=========== DC power before inverter ===========")
    print(problem.get_val("component.inverter_1.dc_current_in", units="A")[0])
    print(
        problem.get_val("component.inverter_1.dc_current_in", units="A")[0]
        * problem.get_val("component.inverter_1.dc_voltage_in", units="V")[0]
    )
    print(problem.get_val("component.inverter_2.dc_current_in", units="A")[0])
    print(
        problem.get_val("component.inverter_2.dc_current_in", units="A")[0]
        * problem.get_val("component.inverter_2.dc_voltage_in", units="V")[0]
    )
    print(problem.get_val("component.inverter_3.dc_current_in", units="A")[0])
    print(
        problem.get_val("component.inverter_3.dc_current_in", units="A")[0]
        * problem.get_val("component.inverter_3.dc_voltage_in", units="V")[0]
    )
    print(problem.get_val("component.inverter_4.dc_current_in", units="A")[0])
    print(
        problem.get_val("component.inverter_4.dc_current_in", units="A")[0]
        * problem.get_val("component.inverter_4.dc_voltage_in", units="V")[0]
    )

    print("\n=========== DC currents in cables ===========")
    print(problem.get_val("component.dc_line_1.dc_current", units="A")[0])
    print(problem.get_val("component.dc_line_2.dc_current", units="A")[0])
    print(problem.get_val("component.dc_line_3.dc_current", units="A")[0])
    print(problem.get_val("component.dc_line_4.dc_current", units="A")[0])
    print(problem.get_val("component.dc_line_5.dc_current", units="A")[0])
    print(problem.get_val("component.dc_line_6.dc_current", units="A")[0])

    print("\n=========== DC power before bus/end of cable ===========")
    print(
        problem.get_val("component.dc_line_1.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_1.dc_voltage_out", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_2.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_2.dc_voltage_out", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_3.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_3.dc_voltage_out", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_4.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_4.dc_voltage_out", units="V")[0]
    )

    print("\n=========== DC power start of cable ===========")
    print(
        problem.get_val("component.dc_line_1.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_1.dc_voltage_in", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_2.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_2.dc_voltage_in", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_3.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_3.dc_voltage_in", units="V")[0]
    )
    print(
        problem.get_val("component.dc_line_4.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_4.dc_voltage_in", units="V")[0]
    )

    print("\n=========== DC power after converter ===========")
    print(
        problem.get_val("component.dc_dc_converter_1.dc_current_out", units="A")[0]
        * problem.get_val("component.dc_dc_converter_1.dc_voltage_out", units="V")[0]
    )

    print("\n=========== DC power before converter ===========")
    print(
        problem.get_val("component.dc_dc_converter_1.dc_current_in", units="A")[0]
        * problem.get_val("component.dc_dc_converter_1.dc_voltage_in", units="V")[0]
    )
    # Result is not really accurate since we used a ICE propeller coupled to a small PMSM not
    # sized for the demand, though it shows that the assembly works just fine

    assert problem.get_val(
        "component.dc_dc_converter_1.dc_current_in", units="A"
    ) * problem.get_val("component.dc_dc_converter_1.dc_voltage_in", units="V") == pytest.approx(
        np.array(
            [
                303709.5,
                304887.2,
                306068.2,
                307252.3,
                308439.5,
                309629.7,
                310822.9,
                312019.0,
                313218.0,
                314419.8,
                315624.4,
                316831.7,
                318041.7,
                319254.4,
                320469.7,
                321687.5,
                322907.9,
                324130.9,
                325356.3,
                326584.3,
                327814.7,
                329047.6,
                330283.1,
                331521.0,
                332761.4,
                334004.3,
                335249.8,
                336497.8,
                337748.4,
                339001.7,
                340257.6,
                341516.2,
                342777.5,
                344041.7,
                345308.7,
                346578.6,
                347851.5,
                349127.4,
                350406.4,
                351688.5,
                352973.9,
                354262.6,
                355554.6,
                356850.1,
                358149.1,
                359451.6,
                360757.8,
                362067.6,
                363381.1,
                364698.4,
            ]
        ),
        abs=1,
    )

    # om.n2(problem)


def test_assembly_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "quad_assembly.yml")

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
    ivc.add_output("thrust", val=np.linspace(500, 550, NB_POINTS_TEST) * 4.0, units="N")
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 50))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PowerTrainPerformancesFromFile(
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
            pre_condition_pt=True,
        ),
        ivc,
    )

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    assert problem.get_val("non_consumable_energy_t_econ", units="W*h") == pytest.approx(
        np.array(
            [
                4218.2,
                4234.6,
                4251.0,
                4267.4,
                4283.9,
                4300.5,
                4317.0,
                4333.7,
                4350.3,
                4367.0,
                4383.7,
                4400.5,
                4417.3,
                4434.1,
                4451.0,
                4467.9,
                4484.9,
                4501.9,
                4518.9,
                4536.0,
                4553.0,
                4570.2,
                4587.3,
                4604.5,
                4621.7,
                4639.0,
                4656.3,
                4673.6,
                4691.0,
                4708.4,
                4725.9,
                4743.3,
                4760.9,
                4778.4,
                4796.0,
                4813.7,
                4831.3,
                4849.1,
                4866.8,
                4884.6,
                4902.5,
                4920.4,
                4938.3,
                4956.3,
                4974.4,
                4992.5,
                5010.6,
                5028.8,
                5047.0,
                5065.3,
            ]
        ),
        abs=1e-1,
    )
    assert problem.get_val("fuel_consumed_t_econ", units="kg") == pytest.approx(
        np.zeros(NB_POINTS_TEST),
        abs=1,
    )

    # om.n2(problem)


def test_assembly_no_cross_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "quad_assembly_no_cross.yml")

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
    ivc.add_output("thrust", val=np.linspace(500, 550, NB_POINTS_TEST) * 4.0, units="N")
    ivc.add_output(
        "exterior_temperature",
        units="degK",
        val=Atmosphere(altitude, altitude_in_feet=False).temperature,
    )
    ivc.add_output("time_step", units="s", val=np.full(NB_POINTS_TEST, 50))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PowerTrainPerformancesFromFile(
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
            pre_condition_pt=True,
        ),
        ivc,
    )

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    assert problem.get_val("non_consumable_energy_t_econ", units="W*h") == pytest.approx(
        np.array(
            [
                4218.2,
                4234.6,
                4251.0,
                4267.4,
                4283.9,
                4300.5,
                4317.0,
                4333.7,
                4350.3,
                4367.0,
                4383.7,
                4400.5,
                4417.3,
                4434.1,
                4451.0,
                4467.9,
                4484.9,
                4501.9,
                4518.9,
                4536.0,
                4553.0,
                4570.2,
                4587.3,
                4604.5,
                4621.7,
                4639.0,
                4656.3,
                4673.6,
                4691.0,
                4708.4,
                4725.9,
                4743.3,
                4760.9,
                4778.4,
                4796.0,
                4813.7,
                4831.3,
                4849.1,
                4866.8,
                4884.6,
                4902.5,
                4920.4,
                4938.3,
                4956.3,
                4974.4,
                4992.5,
                5010.6,
                5028.8,
                5047.0,
                5065.3,
            ]
        ),
        abs=1e-1,
    )
    assert problem.get_val("fuel_consumed_t_econ", units="kg") == pytest.approx(
        np.zeros(NB_POINTS_TEST),
        abs=1,
    )

    # om.n2(problem)


def test_thrust_distributor():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "quad_assembly.yml")

    ivc = get_indep_var_comp(
        list_inputs(
            ThrustDistributor(power_train_file_path=pt_file_path, number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("thrust", val=np.linspace(500, 550, NB_POINTS_TEST) * 4.0, units="N")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        ThrustDistributor(power_train_file_path=pt_file_path, number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("propeller_1_thrust", units="N") == pytest.approx(
        np.linspace(500, 550, NB_POINTS_TEST), rel=1e-2
    )
    assert problem.get_val("propeller_2_thrust", units="N") == pytest.approx(
        np.linspace(500, 550, NB_POINTS_TEST), rel=1e-2
    )
    assert problem.get_val("propeller_3_thrust", units="N") == pytest.approx(
        np.linspace(500, 550, NB_POINTS_TEST), rel=1e-2
    )
    assert problem.get_val("propeller_4_thrust", units="N") == pytest.approx(
        np.linspace(500, 550, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_power_rate():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "quad_assembly.yml")

    ivc = get_indep_var_comp(
        list_inputs(PowerRate(power_train_file_path=pt_file_path, number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "motor_1_shaft_power_out", val=np.linspace(32.5, 65.0, NB_POINTS_TEST), units="kW"
    )
    ivc.add_output(
        "motor_2_shaft_power_out", val=np.linspace(35.0, 70.0, NB_POINTS_TEST), units="kW"
    )
    ivc.add_output(
        "motor_3_shaft_power_out", val=np.linspace(37.5, 75.0, NB_POINTS_TEST), units="kW"
    )
    ivc.add_output(
        "motor_4_shaft_power_out", val=np.linspace(40.0, 80.0, NB_POINTS_TEST), units="kW"
    )
    ivc.add_output("engine_setting_econ", val=np.ones(NB_POINTS_TEST))
    ivc.add_output("altitude_econ", val=np.ones(NB_POINTS_TEST), units="ft")
    ivc.add_output("density_econ", val=np.ones(NB_POINTS_TEST), units="kg/m**3")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PowerRate(power_train_file_path=pt_file_path, number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("thrust_rate_t_econ") == pytest.approx(
        np.linspace(0.5, 1.0, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_slipstream_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "quad_assembly.yml")

    ivc = get_indep_var_comp(
        list_inputs(
            AerodynamicDeltasFromPTFile(
                power_train_file_path=pt_file_path,
                number_of_points=NB_POINTS_TEST,
            )
        ),
        __file__,
        XML_FILE,
    )

    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
    ivc.add_output("density", val=Atmosphere(altitude).density, units="kg/m**3")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("alpha", val=np.linspace(5.0, 10.0, NB_POINTS_TEST), units="deg")
    ivc.add_output("thrust", val=np.linspace(500, 550, NB_POINTS_TEST) * 4.0, units="N")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        AerodynamicDeltasFromPTFile(
            power_train_file_path=pt_file_path,
            number_of_points=NB_POINTS_TEST,
        ),
        ivc,
    )

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    assert problem.get_val("propeller_1.delta_Cl") * 1e6 == pytest.approx(
        np.array(
            [
                528.6,
                534.0,
                539.3,
                544.6,
                549.9,
                555.1,
                560.4,
                565.6,
                570.7,
                575.9,
                581.0,
                586.0,
                591.1,
                596.1,
                601.1,
                606.1,
                611.1,
                616.0,
                620.9,
                625.8,
                630.6,
                635.4,
                640.2,
                645.0,
                649.7,
                654.4,
                659.1,
                663.8,
                668.4,
                673.1,
                677.7,
                682.2,
                686.8,
                691.3,
                695.8,
                700.2,
                704.7,
                709.1,
                713.5,
                717.9,
                722.2,
                726.6,
                730.9,
                735.1,
                739.4,
                743.6,
                747.8,
                752.0,
                756.2,
                760.3,
            ]
        ),
        rel=1e-3,
    )
    assert problem.get_val("propeller_1.delta_Cd") * 1e9 == pytest.approx(
        np.array(
            [
                34.462,
                34.305,
                34.150,
                33.996,
                33.842,
                33.690,
                33.539,
                33.388,
                33.239,
                33.091,
                32.944,
                32.797,
                32.652,
                32.508,
                32.364,
                32.222,
                32.080,
                31.940,
                31.800,
                31.661,
                31.523,
                31.386,
                31.250,
                31.115,
                30.981,
                30.847,
                30.715,
                30.583,
                30.452,
                30.322,
                30.193,
                30.064,
                29.937,
                29.810,
                29.684,
                29.559,
                29.434,
                29.311,
                29.188,
                29.066,
                28.945,
                28.824,
                28.704,
                28.585,
                28.467,
                28.349,
                28.232,
                28.116,
                28.001,
                27.886,
            ]
        ),
        rel=1e-3,
    )
    assert problem.get_val("propeller_1.delta_Cm") * 1e9 == pytest.approx(
        np.array(
            [
                -182.17,
                -181.35,
                -180.53,
                -179.71,
                -178.90,
                -178.09,
                -177.29,
                -176.50,
                -175.71,
                -174.93,
                -174.15,
                -173.38,
                -172.61,
                -171.84,
                -171.09,
                -170.33,
                -169.58,
                -168.84,
                -168.10,
                -167.37,
                -166.64,
                -165.92,
                -165.20,
                -164.48,
                -163.77,
                -163.07,
                -162.37,
                -161.67,
                -160.98,
                -160.29,
                -159.61,
                -158.93,
                -158.25,
                -157.58,
                -156.92,
                -156.26,
                -155.60,
                -154.94,
                -154.30,
                -153.65,
                -153.01,
                -152.37,
                -151.74,
                -151.11,
                -150.48,
                -149.86,
                -149.24,
                -148.63,
                -148.02,
                -147.41,
            ]
        ),
        rel=1e-3,
    )
    assert problem.get_val("propeller_1.delta_Cl") == pytest.approx(
        problem.get_val("propeller_4.delta_Cl"),
        rel=1e-6,
    )
    assert problem.get_val("propeller_2.delta_Cl") == pytest.approx(
        problem.get_val("propeller_3.delta_Cl"),
        rel=1e-6,
    )
    assert all(problem.get_val("propeller_1.delta_Cl") != problem.get_val("propeller_2.delta_Cl"))

    assert problem.get_val("delta_Cl") == pytest.approx(
        problem.get_val("propeller_1.delta_Cl")
        + problem.get_val("propeller_2.delta_Cl")
        + problem.get_val("propeller_3.delta_Cl")
        + problem.get_val("propeller_4.delta_Cl"),
        rel=1e-6,
    )
    assert problem.get_val("delta_Cm") == pytest.approx(
        problem.get_val("propeller_1.delta_Cm")
        + problem.get_val("propeller_2.delta_Cm")
        + problem.get_val("propeller_3.delta_Cm")
        + problem.get_val("propeller_4.delta_Cm"),
        rel=1e-6,
    )
    delta_Cdi = (
        np.array(
            [
                116.44,
                119.1,
                121.79,
                124.49,
                127.22,
                129.98,
                132.75,
                135.55,
                138.37,
                141.21,
                144.08,
                146.96,
                149.87,
                152.8,
                155.75,
                158.72,
                161.71,
                164.72,
                167.75,
                170.8,
                173.87,
                176.96,
                180.07,
                183.2,
                186.35,
                189.52,
                192.7,
                195.91,
                199.13,
                202.37,
                205.63,
                208.91,
                212.2,
                215.52,
                218.85,
                222.19,
                225.56,
                228.94,
                232.33,
                235.75,
                239.18,
                242.62,
                246.09,
                249.56,
                253.06,
                256.57,
                260.09,
                263.63,
                267.19,
                270.75,
            ]
        )
        * 1e-6
    )
    assert problem.get_val("delta_Cdi") == pytest.approx(
        delta_Cdi,
        rel=1e-3,
    )
    assert problem.get_val("delta_Cd") == pytest.approx(
        problem.get_val("propeller_1.delta_Cd")
        + problem.get_val("propeller_2.delta_Cd")
        + problem.get_val("propeller_3.delta_Cd")
        + problem.get_val("propeller_4.delta_Cd")
        + delta_Cdi,
        rel=1e-3,
    )

    # om.n2(problem)
