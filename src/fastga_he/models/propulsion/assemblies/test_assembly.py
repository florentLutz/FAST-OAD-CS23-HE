# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..components.loads.pmsm import PerformancePMSM, SizingPMSM
from ..components.propulsor.propeller import PerformancesPropeller, SizingPropeller
from ..components.connectors.inverter import PerformancesInverter, SizingInverter

XML_FILE = "simple_assembly.xml"
NB_POINTS_TEST = 10


class PerformancesAssembly(om.Group):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]

        ivc = om.IndepVarComp()
        ivc.add_output("rpm", units="min**-1", val=np.full(number_of_points, 4000))

        ivc2 = om.IndepVarComp()
        ivc2.add_output("dc_voltage", units="V", val=np.full(number_of_points, 850.0))

        ivc3 = om.IndepVarComp()
        ivc3.add_output("switching_frequency", units="Hz", val=np.full(number_of_points, 12000.0))

        ivc4 = om.IndepVarComp()
        ivc4.add_output("heat_sink_temperature", units="degK", val=np.full(NB_POINTS_TEST, 288.15))

        self.add_subsystem(
            "propeller_rot_speed",
            ivc,
            promotes=[],
        )
        self.add_subsystem(
            "dc_bus",
            ivc2,
            promotes=[],
        )
        self.add_subsystem(
            "control_inverter",
            ivc3,
            promotes=[],
        )
        self.add_subsystem(
            "inverter_heat_sink",
            ivc4,
            promotes=[],
        )

        self.add_subsystem(
            "propeller_1",
            PerformancesPropeller(propeller_id="propeller_1", number_of_points=number_of_points),
            promotes=["true_airspeed", "altitude", "thrust", "data:*"],
        )
        self.add_subsystem(
            "motor_1",
            PerformancePMSM(motor_id="motor_1", number_of_points=number_of_points),
            promotes=["data:*"],
        )
        self.add_subsystem(
            "inverter_1",
            PerformancesInverter(inverter_id="inverter_1", number_of_points=number_of_points),
            promotes=["dc_current", "data:*"],
        )

        self.connect("propeller_rot_speed.rpm", ["propeller_1.rpm", "motor_1.rpm"])
        self.connect("dc_bus.dc_voltage", "inverter_1.dc_voltage")
        self.connect("control_inverter.switching_frequency", "inverter_1.switching_frequency")
        self.connect("inverter_heat_sink.heat_sink_temperature", "inverter_1.heat_sink_temperature")
        self.connect("propeller_1.shaft_power", "motor_1.shaft_power")
        self.connect("motor_1.rms_current_one_phase", "inverter_1.current")
        self.connect("motor_1.peak_voltage", "inverter_1.peak_ac_voltage")
        self.connect("motor_1.rms_voltage", "inverter_1.rms_voltage")


def test_assembly():

    ivc = get_indep_var_comp(
        list_inputs(PerformancesAssembly(number_of_points=NB_POINTS_TEST)),
        __file__,
        XML_FILE,
    )
    ivc.add_output("altitude", val=np.full(NB_POINTS_TEST, 0.0), units="m")
    ivc.add_output("true_airspeed", val=np.linspace(81.8, 90.5, NB_POINTS_TEST), units="m/s")
    ivc.add_output("thrust", val=np.linspace(1550, 1450, NB_POINTS_TEST), units="N")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAssembly(number_of_points=NB_POINTS_TEST),
        ivc,
    )

    _, _, residuals = problem.model.get_nonlinear_vectors()

    print(problem.get_val("true_airspeed", units="m/s") * problem.get_val("thrust", units="N"))
    print(problem.get_val("component.propeller_1.shaft_power", units="W"))
    print(
        problem.get_val("component.motor_1.rms_current", units="A")
        * problem.get_val("component.motor_1.rms_voltage", units="V")
    )
    print(
        problem.get_val("dc_current", units="A")
        * problem.get_val("component.dc_bus.dc_voltage", units="V")
    )
    # Result is not really accurate since we used a ICE propeller coupled to a small PMSM not
    # sized for the demand, though it shows that the assembly works just fine
