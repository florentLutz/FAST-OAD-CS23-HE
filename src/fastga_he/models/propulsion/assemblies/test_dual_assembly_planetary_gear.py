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
from fastga_he.gui.power_train_network_viewer import power_train_network_viewer

from ..components.loads.pmsm import PerformancesPMSM
from ..components.connectors.planetary_gear import PerformancesPlanetaryGear
from ..components.propulsor.propeller import PerformancesPropeller
from ..components.connectors.inverter import PerformancesInverter
from ..components.connectors.dc_cable import PerformancesHarness
from ..components.connectors.dc_bus import PerformancesDCBus
from ..components.connectors.dc_dc_converter import PerformancesDCDCConverter
from ..components.source.battery import PerformancesBatteryPack

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
OUT_FOLDER_PATH = pth.join(pth.dirname(__file__), "outputs")

XML_FILE = "dual_assembly.xml"
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
        self.options.declare(
            "gear_mode",
            default="percent_split",
            desc="Mode of the planetary gear, should be either percent_split or power_share",
            values=["percent_split", "power_share"],
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        gear_mode = self.options["gear_mode"]

        # Propellers
        self.add_subsystem(
            "propeller_1",
            PerformancesPropeller(propeller_id="propeller_1", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "data:*", "thrust", "density"],
        )

        self.add_subsystem(
            "planetary_gear_1",
            PerformancesPlanetaryGear(
                planetary_gear_id="planetary_gear_1",
                number_of_points=number_of_points,
                gear_mode=gear_mode,
            ),
            promotes=["data:*"],
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

        # DC Buses
        self.add_subsystem(
            "dc_bus_1",
            PerformancesDCBus(
                dc_bus_id="dc_bus_1",
                number_of_points=number_of_points,
                number_of_inputs=1,
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

        # Source bus
        self.add_subsystem(
            "dc_bus_5",
            PerformancesDCBus(
                dc_bus_id="dc_bus_5",
                number_of_points=number_of_points,
                number_of_inputs=1,
                number_of_outputs=2,
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

        self.connect("propeller_1.rpm", "planetary_gear_1.rpm_out")
        self.connect("propeller_1.shaft_power_in", "planetary_gear_1.shaft_power_out")

        self.connect("planetary_gear_1.rpm_in_1", "motor_1.rpm")
        self.connect("planetary_gear_1.rpm_in_2", "motor_2.rpm")

        self.connect("planetary_gear_1.shaft_power_in_1", "motor_1.shaft_power_out")
        self.connect("planetary_gear_1.shaft_power_in_2", "motor_2.shaft_power_out")

        self.connect(
            "motor_1.ac_current_rms_in_one_phase", "inverter_1.ac_current_rms_out_one_phase"
        )
        self.connect(
            "motor_2.ac_current_rms_in_one_phase", "inverter_2.ac_current_rms_out_one_phase"
        )

        self.connect("motor_1.ac_voltage_peak_in", "inverter_1.ac_voltage_peak_out")
        self.connect("motor_2.ac_voltage_peak_in", "inverter_2.ac_voltage_peak_out")

        self.connect("motor_1.ac_voltage_rms_in", "inverter_1.ac_voltage_rms_out")
        self.connect("motor_2.ac_voltage_rms_in", "inverter_2.ac_voltage_rms_out")

        self.connect("dc_bus_1.dc_voltage", "inverter_1.dc_voltage_in")
        self.connect("dc_bus_2.dc_voltage", "inverter_2.dc_voltage_in")

        self.connect("inverter_1.dc_current_in", "dc_bus_1.dc_current_out_1")
        self.connect("inverter_2.dc_current_in", "dc_bus_2.dc_current_out_1")

        # DC bus 1
        self.connect("dc_bus_1.dc_voltage", "dc_line_1.dc_voltage_out")
        self.connect("dc_line_1.dc_current", "dc_bus_1.dc_current_in_1")

        # DC bus 2
        self.connect("dc_bus_2.dc_voltage", "dc_line_2.dc_voltage_out")
        self.connect("dc_line_2.dc_current", "dc_bus_2.dc_current_in_1")

        self.connect("dc_bus_5.dc_voltage", "dc_line_1.dc_voltage_in")
        self.connect("dc_bus_5.dc_voltage", "dc_line_2.dc_voltage_in")

        self.connect("dc_line_1.dc_current", "dc_bus_5.dc_current_out_1")
        self.connect("dc_line_2.dc_current", "dc_bus_5.dc_current_out_2")

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

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # Result is not really accurate since we used a ICE propeller coupled to a small PMSM not
    # sized for the demand, though it shows that the assembly works just fine

    assert problem.get_val(
        "component.dc_dc_converter_1.dc_current_in", units="A"
    ) * problem.get_val("component.dc_dc_converter_1.dc_voltage_in", units="V") == pytest.approx(
        np.array(
            [
                77500.5,
                77778.3,
                78056.7,
                78335.8,
                78615.4,
                78895.7,
                79176.6,
                79458.0,
                79740.1,
                80022.8,
                80306.1,
                80590.0,
                80874.5,
                81159.6,
                81445.3,
                81731.6,
                82018.6,
                82306.1,
                82594.3,
                82883.1,
                83172.5,
                83462.4,
                83753.1,
                84044.3,
                84336.1,
                84628.6,
                84921.6,
                85215.3,
                85509.6,
                85804.5,
                86100.1,
                86396.2,
                86693.0,
                86990.4,
                87288.4,
                87587.0,
                87886.3,
                88186.1,
                88486.6,
                88787.7,
                89089.5,
                89391.8,
                89694.8,
                89998.4,
                90302.7,
                90607.5,
                90913.0,
                91219.1,
                91525.9,
                91833.2,
            ]
        ),
        abs=1,
    )

    # Based on selected planetary gear mode, the power in each branch should be the same,
    # we'll check it in the dc lines om.n2(problem)

    assert problem.get_val("component.dc_line_1.dc_current", units="A")[0] * problem.get_val(
        "component.dc_line_1.dc_voltage_out", units="V"
    )[0] == pytest.approx(
        problem.get_val("component.dc_line_2.dc_current", units="A")[0]
        * problem.get_val("component.dc_line_2.dc_voltage_out", units="V")[0],
        rel=1e-5,
    )


def test_assembly_power_share():
    system = PerformancesAssembly(number_of_points=NB_POINTS_TEST, gear_mode="power_share")

    ivc = get_indep_var_comp(
        list_inputs(system),
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
        PerformancesAssembly(number_of_points=NB_POINTS_TEST, gear_mode="power_share"),
        ivc,
    )

    # Should be a bit different from the previous case since we force the power in different path
    # so each branch will have different efficiency
    assert problem.get_val(
        "component.dc_dc_converter_1.dc_current_in", units="A"
    ) * problem.get_val("component.dc_dc_converter_1.dc_voltage_in", units="V") == pytest.approx(
        np.array(
            [
                77735.9,
                78018.1,
                78300.9,
                78584.4,
                78868.6,
                79153.4,
                79438.8,
                79724.9,
                80011.7,
                80299.2,
                80587.2,
                80876.0,
                81165.4,
                81455.5,
                81746.2,
                82037.6,
                82329.6,
                82622.4,
                82915.7,
                83209.8,
                83504.5,
                83799.9,
                84095.9,
                84392.6,
                84690.0,
                84988.1,
                85286.8,
                85586.2,
                85886.2,
                86187.0,
                86488.4,
                86790.4,
                87093.2,
                87396.6,
                87700.7,
                88005.5,
                88311.0,
                88617.1,
                88924.0,
                89231.5,
                89539.6,
                89848.5,
                90158.1,
                90468.3,
                90779.2,
                91090.8,
                91403.1,
                91716.1,
                92029.8,
                92344.1,
            ]
        ),
        abs=1,
    )

    # Based on selected planetary gear mode, the power in the primary branch, at the input of the
    # PMSM should be exactly equal to 20 kW.

    assert problem.get_val("component.motor_1.shaft_power_out", units="kW")[0] == pytest.approx(
        np.full(NB_POINTS_TEST, 20.0), rel=1e-3
    )


def test_assembly_from_pt_file():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "dual_assembly.yml")
    network_file_path = pth.join(OUT_FOLDER_PATH, "dual_assembly.html")

    power_train_network_viewer(pt_file_path, network_file_path)

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
    ivc.add_output("thrust", val=np.linspace(500, 550, NB_POINTS_TEST), units="N")
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

    assert problem.get_val(
        "component.dc_dc_converter_1.dc_current_in", units="A"
    ) * problem.get_val("component.dc_dc_converter_1.dc_voltage_in", units="V") == pytest.approx(
        np.array(
            [
                77735.9,
                78018.1,
                78300.9,
                78584.4,
                78868.6,
                79153.4,
                79438.8,
                79724.9,
                80011.7,
                80299.2,
                80587.2,
                80876.0,
                81165.4,
                81455.5,
                81746.2,
                82037.6,
                82329.6,
                82622.4,
                82915.7,
                83209.8,
                83504.5,
                83799.9,
                84095.9,
                84392.6,
                84690.0,
                84988.1,
                85286.8,
                85586.2,
                85886.2,
                86187.0,
                86488.4,
                86790.4,
                87093.2,
                87396.6,
                87700.7,
                88005.5,
                88311.0,
                88617.1,
                88924.0,
                89231.5,
                89539.6,
                89848.5,
                90158.1,
                90468.3,
                90779.2,
                91090.8,
                91403.1,
                91716.1,
                92029.8,
                92344.1,
            ]
        ),
        abs=1,
    )

    # Based on selected planetary gear mode, the power in the primary branch, at the input of the
    # PMSM should be exactly equal to 20 kW.

    assert problem.get_val("component.motor_1.shaft_power_out", units="kW")[0] == pytest.approx(
        np.full(NB_POINTS_TEST, 20.0), rel=1e-3
    )
