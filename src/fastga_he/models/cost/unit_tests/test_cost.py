# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os
import pathlib
import pytest

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from ..lcc_engineering_man_hours import LCCEngineeringManHours
from ..lcc_tooling_man_hours import LCCToolingManHours
from ..lcc_manufacturing_man_hours import LCCManufacturingManHours
from ..lcc_tooling_cost_per_unit import LCCToolingCost
from ..lcc_engineering_cost_per_unit import LCCEngineeringCost
from ..lcc_dev_suppoet_cost import LCCDevSupportCost
from ..lcc_manufacturing_cost_per_unit import LCCManufacturingCost


XML_FILE = "data.xml"
DATA_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "data"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_engineering_human_hours():
    input_list = [
        "data:weight:airframe:mass",
        "data:TLAR:v_cruise",
        "data:cost:airframe:num_aircraft_5years",
        "data:cost:airframe:flap_factor",
        "data:cost:airframe:composite_fraction",
        "data:cost:airframe:pressurization_factor",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCEngineeringManHours(),
        ivc,
    )

    assert problem.get_val("data:cost:airframe:engineering_man_hours", units="h") == pytest.approx(
        72.679, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_tooling_human_hours():
    input_list = [
        "data:weight:airframe:mass",
        "data:TLAR:v_cruise",
        "data:cost:airframe:num_aircraft_5years",
        "data:cost:airframe:taper_factor",
        "data:cost:airframe:flap_factor",
        "data:cost:airframe:composite_fraction",
        "data:cost:airframe:pressurization_factor",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCToolingManHours(),
        ivc,
    )

    assert problem.get_val("data:cost:airframe:tooling_man_hours", units="h") == pytest.approx(
        92.625, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_manufacturing_human_hours():
    input_list = [
        "data:weight:airframe:mass",
        "data:TLAR:v_cruise",
        "data:cost:airframe:num_aircraft_5years",
        "data:cost:airframe:flap_factor",
        "data:cost:airframe:composite_fraction",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCManufacturingManHours(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:airframe:manufacturing_man_hours", units="h"
    ) == pytest.approx(688.025, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_engineering_cost():
    input_list = [
        "data:cost:airframe:engineering_man_hours",
        "data:cost:airframe:engineering_cost_per_hour",
        "data:cost:cpi_2012",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCEngineeringCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:airframe:engineering_cost_per_unit", units="USD"
    ) == pytest.approx(26998.679, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_development_support_cost():
    input_list = [
        "data:weight:airframe:mass",
        "data:TLAR:v_cruise",
        "data:cost:airframe:prototype_number",
        "data:cost:airframe:num_aircraft_5years",
        "data:cost:airframe:flap_factor",
        "data:cost:airframe:composite_fraction",
        "data:cost:airframe:pressurization_factor",
        "data:cost:cpi_2012",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCDevSupportCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:airframe:dev_support_cost", units="USD"
    ) == pytest.approx(57318.152, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_tooling_cost():
    input_list = [
        "data:cost:airframe:tooling_man_hours",
        "data:cost:airframe:tooling_cost_per_hour",
        "data:cost:cpi_2012",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCToolingCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:airframe:tooling_cost_per_unit", units="USD"
    ) == pytest.approx(22832.745, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_manufacturing_cost():
    input_list = [
        "data:cost:airframe:manufacturing_man_hours",
        "data:cost:airframe:manufacturing_cost_per_hour",
        "data:cost:cpi_2012",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCManufacturingCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:airframe:manufacturing_cost_per_unit", units="USD"
    ) == pytest.approx(147385.351, rel=1e-3)

    problem.check_partials(compact_print=True)
