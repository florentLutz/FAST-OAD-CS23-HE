# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import pytest
import numpy as np

from ..components.perf_power_in import PerformancesPowerIn
from ..components.perf_current_in import PerformancesCurrentIn
from ..components.perf_maximum import PerformancesMaximum

from ..components.perf_aux_load import PerformancesAuxLoad

from ..components.cstr_enforce import ConstraintsPowerEnforce
from ..components.cstr_ensure import ConstraintsPowerEnsure

from ..components.cstr_aux_load import ConstraintsAuxLoad

from tests.testing_utilities import run_system, get_indep_var_comp

XML_FILE = "ref_aux_load.xml"
NB_POINTS_TEST = 10


def test_power_in_mission():

    ivc = om.IndepVarComp()
    ivc.add_output(
        "data:propulsion:he_power_train:aux_load:aux_load_1:power_in_mission",
        val=15.0,
        units="kW",
    )

    problem = run_system(
        PerformancesPowerIn(aux_load_id="aux_load_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val("power_in", units="kW") == pytest.approx(
        np.full(NB_POINTS_TEST, 15.0), rel=1e-2
    )

    problem.check_partials(compact_print=True)

    ivc2 = om.IndepVarComp()
    ivc2.add_output(
        "data:propulsion:he_power_train:aux_load:aux_load_1:power_in_mission",
        val=np.linspace(8.0, 12.0, NB_POINTS_TEST),
        units="kW",
    )

    problem2 = run_system(
        PerformancesPowerIn(aux_load_id="aux_load_1", number_of_points=NB_POINTS_TEST), ivc2
    )

    assert problem2.get_val("power_in", units="kW") == pytest.approx(
        np.linspace(8.0, 12.0, NB_POINTS_TEST), rel=1e-2
    )

    problem2.check_partials(compact_print=True)


def test_current_in():

    ivc = om.IndepVarComp()
    ivc.add_output("power_in", units="kW", val=np.linspace(8.0, 12.0, NB_POINTS_TEST))
    ivc.add_output("dc_voltage_in", units="V", val=np.full(NB_POINTS_TEST, 400.0))

    problem = run_system(PerformancesCurrentIn(number_of_points=NB_POINTS_TEST), ivc)

    assert problem.get_val("dc_current_in", units="A") == pytest.approx(
        np.linspace(20.0, 30.0, NB_POINTS_TEST), rel=1e-2
    )

    problem.check_partials(compact_print=True)


def test_maximum():

    ivc = om.IndepVarComp()
    ivc.add_output("power_in", units="kW", val=np.linspace(8.0, 12.0, NB_POINTS_TEST))

    problem = run_system(
        PerformancesMaximum(aux_load_id="aux_load_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:aux_load:aux_load_1:power_max", units="kW"
    ) == pytest.approx(12.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_performances():

    input_list = ["data:propulsion:he_power_train:aux_load:aux_load_1:power_in_mission"]
    ivc = get_indep_var_comp(input_list, __file__, XML_FILE)
    ivc.add_output("dc_voltage_in", units="V", val=np.full(NB_POINTS_TEST, 400.0))

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        PerformancesAuxLoad(aux_load_id="aux_load_1", number_of_points=NB_POINTS_TEST), ivc
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:aux_load:aux_load_1:power_max", units="kW"
    ) == pytest.approx(10.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_enforce():

    input_list = ["data:propulsion:he_power_train:aux_load:aux_load_1:power_max"]
    ivc = get_indep_var_comp(input_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsPowerEnforce(aux_load_id="aux_load_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:aux_load:aux_load_1:power_rating", units="kW"
    ) == pytest.approx(10.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraint_ensure():

    input_list = [
        "data:propulsion:he_power_train:aux_load:aux_load_1:power_max",
        "data:propulsion:he_power_train:aux_load:aux_load_1:power_rating",
    ]
    ivc = get_indep_var_comp(input_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsPowerEnsure(aux_load_id="aux_load_1"), ivc)

    assert problem.get_val(
        "constraints:propulsion:he_power_train:aux_load:aux_load_1:power_rating", units="kW"
    ) == pytest.approx(-1.0, rel=1e-2)

    problem.check_partials(compact_print=True)


def test_constraints():

    input_list = ["data:propulsion:he_power_train:aux_load:aux_load_1:power_max"]
    ivc = get_indep_var_comp(input_list, __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ConstraintsAuxLoad(aux_load_id="aux_load_1"), ivc)

    assert problem.get_val(
        "data:propulsion:he_power_train:aux_load:aux_load_1:power_rating", units="kW"
    ) == pytest.approx(10.0, rel=1e-2)

    problem.check_partials(compact_print=True)
