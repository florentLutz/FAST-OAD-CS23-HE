# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import time

import numpy as np
import openmdao.api as om
import pytest
from stdatm import Atmosphere

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

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

XML_FILE = "quad_assembly.xml"
NB_POINTS_TEST = 25
COEFF_DIFF = 0.0


class PerformancesAssembly(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Solvers setup
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["maxiter"] = 200
        self.nonlinear_solver.options["rtol"] = 1e-5
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
                297007.0,
                299218.0,
                301440.0,
                303673.0,
                305917.0,
                308173.0,
                310440.0,
                312719.0,
                315009.0,
                317310.0,
                319623.0,
                321947.0,
                324283.0,
                326630.0,
                328989.0,
                331360.0,
                333742.0,
                336135.0,
                338541.0,
                340958.0,
                343386.0,
                345827.0,
                348279.0,
                350743.0,
                353218.0,
            ]
        ),
        abs=1,
    )

    # om.n2(problem)


def test_assembly_from_pt_file():

    pt_file_path = "D:/fl.lutz/FAST/FAST-OAD/FAST-OAD-CS23-HE/src/fastga_he/models/propulsion/assemblies/data/quad_assembly.yml"

    ivc = get_indep_var_comp(
        list_inputs(
            PowerTrainPerformancesFromFile(
                power_train_file_path=pt_file_path, number_of_points=NB_POINTS_TEST
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
            power_train_file_path=pt_file_path, number_of_points=NB_POINTS_TEST
        ),
        ivc,
    )

    assert problem.get_val(
        "component.dc_dc_converter_1.dc_current_in", units="A"
    ) * problem.get_val("component.dc_dc_converter_1.dc_voltage_in", units="V") == pytest.approx(
        np.array(
            [
                297007.0,
                299218.0,
                301440.0,
                303673.0,
                305918.0,
                308173.0,
                310441.0,
                312719.0,
                315009.0,
                317310.0,
                319623.0,
                321947.0,
                324283.0,
                326630.0,
                328989.0,
                331359.0,
                333741.0,
                336135.0,
                338540.0,
                340957.0,
                343385.0,
                345825.0,
                348277.0,
                350741.0,
                353216.0,
            ]
        ),
        abs=1,
    )

    # om.n2(problem)


def test_thrust_distributor():

    pt_file_path = "D:/fl.lutz/FAST/FAST-OAD/FAST-OAD-CS23-HE/src/fastga_he/models/propulsion/assemblies/data/quad_assembly.yml"

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
