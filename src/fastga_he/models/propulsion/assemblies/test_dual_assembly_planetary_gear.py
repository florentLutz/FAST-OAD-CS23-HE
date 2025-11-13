# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

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
from fastga_he.gui.power_train_network_viewer_hv import power_train_network_viewer_hv


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
                77205.4,
                77495.2,
                77785.7,
                78076.9,
                78368.8,
                78661.4,
                78954.7,
                79248.7,
                79543.3,
                79838.6,
                80134.6,
                80431.2,
                80728.6,
                81026.5,
                81325.2,
                81624.4,
                81924.4,
                82224.7,
                82525.8,
                82827.4,
                83129.7,
                83432.7,
                83736.2,
                84040.4,
                84345.3,
                84650.7,
                84956.8,
                85263.5,
                85570.8,
                85878.7,
                86187.2,
                86496.3,
                86806.0,
                87116.3,
                87427.2,
                87738.8,
                88050.9,
                88363.6,
                88676.8,
                88990.7,
                89305.2,
                89620.2,
                89935.8,
                90252.0,
                90568.8,
                90886.2,
                91204.1,
                91522.6,
                91841.7,
                92161.4,
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
        rel=1e-4,
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
                77451.6,
                77746.0,
                78041.1,
                78337.0,
                78633.6,
                78931.0,
                79229.0,
                79527.9,
                79827.4,
                80127.7,
                80428.7,
                80730.4,
                81032.9,
                81336.0,
                81639.9,
                81944.5,
                82249.5,
                82555.3,
                82861.8,
                83169.0,
                83476.8,
                83785.4,
                84094.6,
                84404.6,
                84715.2,
                85026.5,
                85338.5,
                85651.1,
                85964.4,
                86278.4,
                86593.1,
                86908.4,
                87224.3,
                87541.0,
                87858.3,
                88176.2,
                88494.8,
                88814.0,
                89133.9,
                89454.5,
                89775.7,
                90097.5,
                90420.0,
                90743.1,
                91066.8,
                91391.2,
                91716.3,
                92042.0,
                92368.3,
                92695.2,
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

    power_train_network_viewer_hv(pt_file_path, network_file_path)

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
                77451.6,
                77746.0,
                78041.1,
                78337.0,
                78633.6,
                78931.0,
                79229.0,
                79527.9,
                79827.4,
                80127.7,
                80428.7,
                80730.4,
                81032.9,
                81336.0,
                81639.9,
                81944.5,
                82249.5,
                82555.3,
                82861.8,
                83169.0,
                83476.8,
                83785.4,
                84094.6,
                84404.6,
                84715.2,
                85026.5,
                85338.5,
                85651.1,
                85964.4,
                86278.4,
                86593.1,
                86908.4,
                87224.3,
                87541.0,
                87858.3,
                88176.2,
                88494.8,
                88814.0,
                89133.9,
                89454.5,
                89775.7,
                90097.5,
                90420.0,
                90743.1,
                91066.8,
                91391.2,
                91716.3,
                92042.0,
                92368.3,
                92695.2,
            ]
        ),
        abs=1,
    )

    # Based on selected planetary gear mode, the power in the primary branch, at the input of the
    # PMSM should be exactly equal to 20 kW.

    assert problem.get_val("component.motor_1.shaft_power_out", units="kW")[0] == pytest.approx(
        np.full(NB_POINTS_TEST, 20.0), rel=1e-3
    )
