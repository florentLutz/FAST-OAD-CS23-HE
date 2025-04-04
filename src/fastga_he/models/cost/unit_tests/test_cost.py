# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os
import pathlib
import pytest
import os.path as pth
import openmdao.api as om

from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from ..lcc_engineering_man_hours import LCCEngineeringManHours
from ..lcc_sale_price import LCCSalePrice
from ..lcc_tooling_man_hours import LCCToolingManHours
from ..lcc_manufacturing_man_hours import LCCManufacturingManHours
from ..lcc_flight_test_cost import LCCFlightTestCost
from ..lcc_quality_control_cost import LCCQualityControlCost
from ..lcc_tooling_cost import LCCToolingCost
from ..lcc_engineering_cost import LCCEngineeringCost
from ..lcc_dev_suppoet_cost import LCCDevSupportCost
from ..lcc_material_cost import LCCMaterialCost
from ..lcc_avionics_cost import LCCAvionicsCost
from ..lcc_manufacturing_cost import LCCManufacturingCost
from ..lcc_certification_cost import LCCCertificationCost
from ..lcc_production_cost import LCCProductionCost
from ..lcc_production_cost_sum import LCCSumProductionCost
from ..lcc_landing_gear_cost_reduction import LCCLandingGearCostReduction


XML_FILE = "data.xml"
DATA_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "data"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_engineering_human_hours():
    input_list = [
        "data:weight:airframe:mass",
        "data:TLAR:v_cruise",
        "data:cost:num_aircraft_5years",
        "data:cost:composite_fraction",
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

    assert problem.get_val("data:cost:engineering_man_hours", units="h") == pytest.approx(
        72.679, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_tooling_human_hours():
    input_list = [
        "data:weight:airframe:mass",
        "data:TLAR:v_cruise",
        "data:cost:num_aircraft_5years",
        "data:cost:taper_factor",
        "data:cost:composite_fraction",
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

    assert problem.get_val("data:cost:tooling_man_hours", units="h") == pytest.approx(
        87.993, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_manufacturing_human_hours():
    input_list = [
        "data:weight:airframe:mass",
        "data:TLAR:v_cruise",
        "data:cost:num_aircraft_5years",
        "data:cost:composite_fraction",
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

    assert problem.get_val("data:cost:manufacturing_man_hours", units="h") == pytest.approx(
        688.025, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_engineering_cost():
    input_list = [
        "data:cost:engineering_man_hours",
        "data:cost:engineering_cost_per_hour",
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

    assert problem.get_val("data:cost:engineering_cost_per_unit", units="USD") == pytest.approx(
        26998.679, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_development_support_cost():
    input_list = [
        "data:weight:airframe:mass",
        "data:TLAR:v_cruise",
        "data:cost:prototype_number",
        "data:cost:num_aircraft_5years",
        "data:cost:composite_fraction",
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

    assert problem.get_val("data:cost:dev_support_cost_per_unit", units="USD") == pytest.approx(
        764.242, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_tooling_cost():
    input_list = [
        "data:cost:tooling_man_hours",
        "data:cost:tooling_cost_per_hour",
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

    assert problem.get_val("data:cost:tooling_cost_per_unit", units="USD") == pytest.approx(
        21690.978, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_manufacturing_cost():
    input_list = [
        "data:cost:manufacturing_man_hours",
        "data:cost:manufacturing_cost_per_hour",
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

    assert problem.get_val("data:cost:manufacturing_cost_per_unit", units="USD") == pytest.approx(
        147385.351, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_quality_control_cost():
    input_list = [
        "data:cost:manufacturing_cost_per_unit",
        "data:cost:composite_fraction",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCQualityControlCost(),
        ivc,
    )

    assert problem.get_val("data:cost:quality_control_cost_per_unit", units="USD") == pytest.approx(
        19160.096, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_flight_test_cost():
    input_list = [
        "data:weight:airframe:mass",
        "data:TLAR:v_cruise",
        "data:cost:prototype_number",
        "data:cost:num_aircraft_5years",
        "data:cost:cpi_2012",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCFlightTestCost(),
        ivc,
    )

    assert problem.get_val("data:cost:flight_test_cost_per_unit", units="USD") == pytest.approx(
        93.105, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_material_cost():
    input_list = [
        "data:weight:airframe:mass",
        "data:TLAR:v_cruise",
        "data:cost:num_aircraft_5years",
        "data:cost:cpi_2012",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCMaterialCost(),
        ivc,
    )

    assert problem.get_val("data:cost:material_cost_per_unit", units="USD") == pytest.approx(
        8735.09, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_avionics_cost():
    input_list = [
        "data:cost:cpi_2012",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCAvionicsCost(),
        ivc,
    )

    assert problem.get_val("data:cost:avionics_cost_per_unit", units="USD") == pytest.approx(
        21000.0, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_certification_cost():
    input_list = [
        "data:cost:engineering_cost_per_unit",
        "data:cost:dev_support_cost_per_unit",
        "data:cost:flight_test_cost_per_unit",
        "data:cost:tooling_cost_per_unit",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCCertificationCost(),
        ivc,
    )

    assert problem.get_val("data:cost:certification_cost_per_unit", units="USD") == pytest.approx(
        49547.05, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_landing_gear_cost_reduction():
    input_list = [
        "data:cost:fixed_landing_gear",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCLandingGearCostReduction(),
        ivc,
    )

    assert problem.get_val("data:cost:landing_gear_cost_reduction", units="USD") == pytest.approx(
        0.0, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_cost_sum():
    components_type = ["propeller", "turboshaft", "fuel_tank"]
    components_name = ["propeller_1", "turboshaft_1", "fuel_tank_1"]

    ivc = get_indep_var_comp(
        list_inputs(
            LCCSumProductionCost(
                cost_components_type=components_type, cost_components_name=components_name
            )
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output(
        "data:propulsion:he_power_train:propeller:propeller_1:cost_per_unit",
        units="USD",
        val=4403.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:cost_per_unit",
        units="USD",
        val=3.0e5,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:cost_per_unit",
        units="USD",
        val=1000.0,
    )

    problem = run_system(
        LCCSumProductionCost(
            cost_components_type=components_type, cost_components_name=components_name
        ),
        ivc,
    )

    assert problem.get_val("data:cost:production_cost_per_unit", units="USD") == pytest.approx(
        600777.59, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_sale_price():
    input_list = [
        "data:cost:production_cost_per_unit",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCSalePrice(),
        ivc,
    )

    assert problem.get_val("data:cost:sale_price_per_unit", units="USD") == pytest.approx(
        412758.77, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_production_cost():
    ivc = get_indep_var_comp(
        list_inputs(
            LCCProductionCost(power_train_file_path=DATA_FOLDER_PATH / "fuel_propulsion.yml")
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCProductionCost(power_train_file_path=DATA_FOLDER_PATH / "fuel_propulsion.yml"),
        ivc,
    )

    assert problem.get_val("data:cost:engineering_cost_per_unit", units="USD") == pytest.approx(
        26998.679, rel=1e-3
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:ICE:ice_1:cost_per_unit", units="USD"
    ) == pytest.approx(70956.47, rel=1e-3)

    assert problem.get_val("data:cost:production_cost_per_unit", units="USD") == pytest.approx(
        371854.75, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))
