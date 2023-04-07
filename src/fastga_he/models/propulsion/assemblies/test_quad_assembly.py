# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os.path as pth

import numpy as np
import openmdao.api as om
import pytest
from stdatm import Atmosphere

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from utils.filter_residuals import filter_residuals

from fastga_he.models.propulsion.assemblers.performances_from_pt_file import (
    PowerTrainPerformancesFromFile,
)

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
            promotes=["true_airspeed", "altitude", "data:*", "thrust"],
        )
        self.add_subsystem(
            "propeller_2",
            PerformancesPropeller(propeller_id="propeller_2", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*", "thrust"],
        )
        self.add_subsystem(
            "propeller_3",
            PerformancesPropeller(propeller_id="propeller_3", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*", "thrust"],
        )
        self.add_subsystem(
            "propeller_4",
            PerformancesPropeller(propeller_id="propeller_4", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*", "thrust"],
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
                296985.0,
                298067.0,
                299151.0,
                300238.0,
                301328.0,
                302421.0,
                303517.0,
                304615.0,
                305716.0,
                306820.0,
                307927.0,
                309036.0,
                310148.0,
                311262.0,
                312379.0,
                313499.0,
                314621.0,
                315746.0,
                316873.0,
                318003.0,
                319135.0,
                320270.0,
                321408.0,
                322548.0,
                323691.0,
                324837.0,
                325985.0,
                327136.0,
                328290.0,
                329446.0,
                330606.0,
                331768.0,
                332934.0,
                334103.0,
                335274.0,
                336449.0,
                337628.0,
                338809.0,
                339994.0,
                341183.0,
                342375.0,
                343570.0,
                344769.0,
                345972.0,
                347179.0,
                348390.0,
                349605.0,
                350824.0,
                352047.0,
                353273.0,
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
                pre_condition_voltage=True,
            )
        ),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
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
            pre_condition_voltage=True,
        ),
        ivc,
    )

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    assert problem.get_val("non_consumable_energy_t_econ", units="W*h") == pytest.approx(
        np.array(
            [
                4124.7,
                4139.8,
                4154.9,
                4170.0,
                4185.1,
                4200.3,
                4215.6,
                4230.8,
                4246.1,
                4261.4,
                4276.8,
                4292.2,
                4307.7,
                4323.1,
                4338.6,
                4354.2,
                4369.8,
                4385.4,
                4401.1,
                4416.8,
                4432.5,
                4448.3,
                4464.1,
                4480.0,
                4495.9,
                4511.9,
                4527.9,
                4543.9,
                4560.0,
                4576.1,
                4592.2,
                4608.4,
                4624.7,
                4640.9,
                4657.3,
                4673.6,
                4690.1,
                4706.5,
                4723.0,
                4739.6,
                4756.1,
                4772.8,
                4789.5,
                4806.2,
                4823.0,
                4839.8,
                4856.6,
                4873.5,
                4890.5,
                4907.5,
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
                pre_condition_voltage=True,
            )
        ),
        __file__,
        XML_FILE,
    )
    altitude = np.full(NB_POINTS_TEST, 0.0)
    ivc.add_output("altitude", val=altitude, units="m")
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
            pre_condition_voltage=True,
        ),
        ivc,
    )

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    assert problem.get_val("non_consumable_energy_t_econ", units="W*h") == pytest.approx(
        np.array(
            [
                4124.7,
                4139.8,
                4154.9,
                4170.0,
                4185.1,
                4200.3,
                4215.6,
                4230.8,
                4246.1,
                4261.4,
                4276.8,
                4292.2,
                4307.7,
                4323.1,
                4338.6,
                4354.2,
                4369.8,
                4385.4,
                4401.1,
                4416.8,
                4432.5,
                4448.3,
                4464.1,
                4480.0,
                4495.9,
                4511.9,
                4527.9,
                4543.9,
                4560.0,
                4576.1,
                4592.2,
                4608.4,
                4624.7,
                4640.9,
                4657.3,
                4673.6,
                4690.1,
                4706.5,
                4723.0,
                4739.6,
                4756.1,
                4772.8,
                4789.5,
                4806.2,
                4823.0,
                4839.8,
                4856.6,
                4873.5,
                4890.5,
                4907.5,
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
    ivc.add_output("engine_setting_econ", val=np.ones(NB_POINTS_TEST), units="N")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PowerRate(power_train_file_path=pt_file_path, number_of_points=NB_POINTS_TEST),
        ivc,
    )

    assert problem.get_val("thrust_rate_t_econ") == pytest.approx(
        np.linspace(0.5, 1.0, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)
