# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import pytest

import openmdao.api as om

from ..components.perf_mission_rpm import PerformancesRPMMission
from ..components.perf_current_rms_3_phases import PerformancesCurrentRMS3Phases
from ..components.perf_torque import PerformancesTorque
from ..components.perf_losses import PerformancesLosses
from ..components.perf_shaft_power_in import PerformancesShaftPowerIn
from ..components.perf_efficiency import PerformancesEfficiency
from ..components.perf_active_power import PerformancesActivePower
from ..components.perf_apparent_power import PerformancesApparentPower
from ..components.perf_voltage_rms import PerformancesVoltageRMS
from ..components.perf_voltage_peak import PerformancesVoltagePeak
from ..components.perf_maximum import PerformancesMaximum

from ..components.perf_generator import PerformancesGenerator

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

XML_FILE = "sample_generator.xml"
NB_POINTS_TEST = 10


def test_rpm_mission():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesRPMMission(generator_id="generator_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesRPMMission(generator_id="generator_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("rpm", units="min**-1") == pytest.approx(
        np.full(NB_POINTS_TEST, 2500.0), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_rms_current_3_phases():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "ac_current_rms_out_one_phase", val=np.linspace(125, 145, NB_POINTS_TEST), units="A"
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesCurrentRMS3Phases(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_current_rms_out", units="A") == pytest.approx(
        np.linspace(375.0, 435.0, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_torque():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesTorque(generator_id="generator_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("ac_current_rms_out", val=np.linspace(375.0, 435.0, NB_POINTS_TEST), units="A")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesTorque(generator_id="generator_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("torque_in", units="N*m") == pytest.approx(
        np.linspace(300.0, 348.0, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_losses():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesLosses(generator_id="generator_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500.0), units="min**-1")
    ivc.add_output("torque_in", val=np.linspace(300.0, 348.0, NB_POINTS_TEST), units="N*m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesLosses(generator_id="generator_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("power_losses", units="W") == pytest.approx(
        np.array([3700.3, 3781.0, 3863.2, 3946.7, 4031.7, 4118.1, 4205.9, 4295.2, 4385.8, 4477.9]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_shaft_power_in():

    ivc = om.IndepVarComp()
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500.0), units="min**-1")
    ivc.add_output("torque_in", val=np.linspace(300.0, 348.0, NB_POINTS_TEST), units="N*m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesShaftPowerIn(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        np.array([78.5, 79.9, 81.3, 82.7, 84.1, 85.5, 86.9, 88.3, 89.7, 91.1]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_efficiency():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_in",
        val=np.array([78.5, 79.9, 81.3, 82.7, 84.1, 85.5, 86.9, 88.3, 89.7, 91.1]),
        units="kW",
    )
    ivc.add_output(
        "power_losses",
        val=np.array(
            [3700.3, 3781.0, 3863.2, 3946.7, 4031.7, 4118.1, 4205.9, 4295.2, 4385.8, 4477.9]
        ),
        units="W",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesEfficiency(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("efficiency") == pytest.approx(
        np.array([0.953, 0.953, 0.952, 0.952, 0.952, 0.952, 0.952, 0.951, 0.951, 0.951]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_active_power():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "shaft_power_in",
        val=np.array([78.5, 79.9, 81.3, 82.7, 84.1, 85.5, 86.9, 88.3, 89.7, 91.1]),
        units="kW",
    )
    ivc.add_output(
        "efficiency",
        val=np.array([0.953, 0.953, 0.952, 0.952, 0.952, 0.952, 0.952, 0.951, 0.951, 0.951]),
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesActivePower(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("active_power", units="kW") == pytest.approx(
        np.array([74.8, 76.1, 77.4, 78.7, 80.1, 81.4, 82.7, 84.0, 85.3, 86.6]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_apparent_power():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "active_power",
        val=np.array([74.8, 76.1, 77.4, 78.7, 80.1, 81.4, 82.7, 84.0, 85.3, 86.6]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesApparentPower(generator_id="generator_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("apparent_power", units="kW") == pytest.approx(
        np.array([74.8, 76.1, 77.4, 78.7, 80.1, 81.4, 82.7, 84.0, 85.3, 86.6]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_voltage_rms():

    ivc = om.IndepVarComp()
    ivc.add_output("ac_current_rms_out", val=np.linspace(375.0, 435.0, NB_POINTS_TEST), units="A")
    ivc.add_output(
        "apparent_power",
        val=np.array([74.8, 76.1, 77.4, 78.7, 80.1, 81.4, 82.7, 84.0, 85.3, 86.6]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesVoltageRMS(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_voltage_rms_out", units="V") == pytest.approx(
        np.array([199.47, 199.39, 199.31, 199.24, 199.42, 199.35, 199.28, 199.21, 199.14, 199.08]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_voltage_peak():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "ac_voltage_rms_out",
        val=np.array(
            [199.47, 199.39, 199.31, 199.24, 199.42, 199.35, 199.28, 199.21, 199.14, 199.08]
        ),
        units="V",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PerformancesVoltagePeak(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("ac_voltage_peak_out", units="V") == pytest.approx(
        np.array([244.3, 244.2, 244.1, 244.02, 244.24, 244.15, 244.07, 243.98, 243.9, 243.82]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)


def test_maximum():

    ivc = om.IndepVarComp()

    ivc.add_output(
        "ac_voltage_peak_out",
        val=np.array([244.3, 244.2, 244.1, 244.02, 244.24, 244.15, 244.07, 243.98, 243.9, 243.82]),
        units="V",
    )
    ivc.add_output(
        "ac_current_rms_out_one_phase", val=np.linspace(125, 145, NB_POINTS_TEST), units="A"
    )
    ivc.add_output("rpm", val=np.full(NB_POINTS_TEST, 2500.0), units="min**-1")
    ivc.add_output("torque_in", val=np.linspace(300.0, 348.0, NB_POINTS_TEST), units="N*m")
    ivc.add_output(
        "power_losses",
        val=np.array(
            [3700.3, 3781.0, 3863.2, 3946.7, 4031.7, 4118.1, 4205.9, 4295.2, 4385.8, 4477.9]
        ),
        units="W",
    )
    ivc.add_output(
        "shaft_power_in",
        val=np.array([78.5, 79.9, 81.3, 82.7, 84.1, 85.5, 86.9, 88.3, 89.7, 91.1]),
        units="kW",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesMaximum(generator_id="generator_1", number_of_points=NB_POINTS_TEST), ivc
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:current_ac_max", units="A"
    ) == pytest.approx(145.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:voltage_ac_max", units="V"
    ) == pytest.approx(244.3, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:torque_max", units="N*m"
    ) == pytest.approx(348.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:rpm_max", units="min**-1"
    ) == pytest.approx(2500.0, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:losses_max", units="W"
    ) == pytest.approx(4477.9, rel=1e-2)
    assert problem.get_val(
        "data:propulsion:he_power_train:generator:generator_1:shaft_power_max", units="W"
    ) == pytest.approx(91.1e3, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performances():

    ivc = get_indep_var_comp(
        list_inputs(
            PerformancesGenerator(generator_id="generator_1", number_of_points=NB_POINTS_TEST)
        ),
        __file__,
        XML_FILE,
    )
    ivc.add_output(
        "ac_current_rms_out_one_phase", val=np.linspace(125, 145, NB_POINTS_TEST), units="A"
    )

    problem = run_system(
        PerformancesGenerator(generator_id="generator_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("ac_voltage_peak_out", units="V") == pytest.approx(
        np.array([244.3, 244.2, 244.1, 244.02, 244.24, 244.15, 244.07, 243.98, 243.9, 243.82]),
        rel=1e-2,
    )
    assert problem.get_val("shaft_power_in", units="kW") == pytest.approx(
        np.array([78.5, 79.9, 81.3, 82.7, 84.1, 85.5, 86.9, 88.3, 89.7, 91.1]),
        rel=1e-2,
    )
    assert problem.get_val("efficiency") == pytest.approx(
        np.array([0.953, 0.953, 0.952, 0.952, 0.952, 0.952, 0.952, 0.951, 0.951, 0.951]),
        rel=1e-2,
    )

    problem.check_partials(compact_print=True)
