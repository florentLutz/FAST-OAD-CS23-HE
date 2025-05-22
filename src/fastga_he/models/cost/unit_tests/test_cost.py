# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os
import pathlib
import pytest
import os.path as pth
import openmdao.api as om
import fastoad.api as oad

from ..lcc import LCC
from tests.testing_utilities import run_system, get_indep_var_comp, list_inputs

from ..lcc_engineering_man_hours import LCCEngineeringManHours
from ..lcc_msp import LCCMSP
from ..lcc_tooling_man_hours import LCCToolingManHours
from ..lcc_manufacturing_man_hours import LCCManufacturingManHours
from ..lcc_flight_test_cost import LCCFlightTestCost
from ..lcc_quality_control_cost import LCCQualityControlCost
from ..lcc_tooling_cost import LCCToolingCost
from ..lcc_engineering_cost import LCCEngineeringCost
from ..lcc_dev_support_cost import LCCDevSupportCost
from ..lcc_material_cost import LCCMaterialCost
from ..lcc_avionics_cost import LCCAvionicsCost
from ..lcc_manufacturing_cost import LCCManufacturingCost
from ..lcc_certification_cost import LCCCertificationCost
from ..lcc_production_cost import LCCProductionCost
from ..lcc_production_cost_sum import LCCSumProductionCost
from ..lcc_annual_insurance_cost import LCCAnnualInsuranceCost
from ..lcc_landing_gear_cost_reduction import LCCLandingGearCostReduction
from ..lcc_delivery_cost import LCCDeliveryCost
from ..lcc_deliveray_duration_ratio import LCCDeliveryDurationRatio
from ..lcc_landing_cost import LCCLandingCost
from ..lcc_daily_parking_cost import LCCDailyParkingCost
from ..lcc_annual_crew_cost import LCCAnnualCrewCost
from ..lcc_annual_airport_cost import LCCAnnualAirportCost
from ..lcc_annual_loan_cost import LCCAnnualLoanCost
from ..lcc_annual_depreciation import LCCAnnualDepreciation
from ..lcc_maintenance_cost import LCCMaintenanceCost
from ..lcc_maintenance_miscellaneous_cost import LCCMaintenanceMiscellaneousCost
from ..lcc_fuel_cost import LCCFuelCost
from ..lcc_electricity_cost import LCCElectricityCost
from ..lcc_annual_energy_cost import LCCAnnualEnergyCost
from ..lcc_operational_cost_sum import LCCSumOperationalCost
from ..lcc_operational_cost import LCCOperationalCost
from ..lcc_learning_curve_factor import LCCLearningCurveFactor
from ..lcc_learning_curve_discount import LCCLearningCurveDiscount

from ..constants import SERVICE_COST_CERTIFICATION


XML_FILE = "tbm_900_inputs.xml"
DATA_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "data"

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


def test_engineering_man_hours_5_years():
    input_list = [
        "data:weight:airframe:mass",
        "data:cost:v_cruise_design",
        "data:cost:production:number_aircraft_5_years",
        "data:cost:production:composite_fraction",
        "data:geometry:flap_type",
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

    assert problem.get_val(
        "data:cost:production:engineering_man_hours_5_years", units="h"
    ) == pytest.approx(2102.74, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_tooling_man_hours_5_years():
    input_list = [
        "data:weight:airframe:mass",
        "data:cost:v_cruise_design",
        "data:cost:production:number_aircraft_5_years",
        "data:cost:production:composite_fraction",
        "data:geometry:flap_type",
        "data:geometry:wing:taper_ratio",
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

    assert problem.get_val(
        "data:cost:production:tooling_man_hours_5_years", units="h"
    ) == pytest.approx(1143.41, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_manufacturing_man_hours_5_years():
    input_list = [
        "data:weight:airframe:mass",
        "data:cost:v_cruise_design",
        "data:cost:production:number_aircraft_5_years",
        "data:cost:production:composite_fraction",
        "data:geometry:flap_type",
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
        "data:cost:production:manufacturing_man_hours_5_years", units="h"
    ) == pytest.approx(5134.6, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_engineering_cost():
    input_list = [
        "data:cost:production:engineering_cost_per_hour",
        "data:cost:cpi_2012",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )
    ivc.add_output("data:cost:production:engineering_man_hours_5_years", units="h", val=2102.74)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCEngineeringCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:production:engineering_cost_per_unit", units="USD"
    ) == pytest.approx(781122.525, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_development_support_cost():
    input_list = [
        "data:weight:airframe:mass",
        "data:cost:v_cruise_design",
        "data:cost:prototype_number",
        "data:cost:production:number_aircraft_5_years",
        "data:cost:production:composite_fraction",
        "data:cost:cpi_2012",
        "data:geometry:flap_type",
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
        "data:cost:production:dev_support_cost_per_unit", units="USD"
    ) == pytest.approx(39760.8, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_tooling_cost():
    input_list = [
        "data:cost:production:tooling_cost_per_hour",
        "data:cost:cpi_2012",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )
    ivc.add_output("data:cost:production:tooling_man_hours_5_years", units="h", val=1143.41)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCToolingCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:production:tooling_cost_per_unit", units="USD"
    ) == pytest.approx(281858.99, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_manufacturing_cost():
    input_list = [
        "data:cost:production:manufacturing_cost_per_hour",
        "data:cost:cpi_2012",
    ]

    ivc = get_indep_var_comp(
        input_list,
        __file__,
        XML_FILE,
    )
    ivc.add_output("data:cost:production:manufacturing_man_hours_5_years", units="h", val=5134.6)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCManufacturingCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:production:manufacturing_cost_per_unit", units="USD"
    ) == pytest.approx(1099908.9, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_quality_control_cost():
    ivc = om.IndepVarComp()
    ivc.add_output("data:cost:production:manufacturing_cost_per_unit", units="USD", val=1099908.9)
    ivc.add_output("data:cost:production:composite_fraction", val=0.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCQualityControlCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:production:quality_control_cost_per_unit", units="USD"
    ) == pytest.approx(142988.157, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_flight_test_cost():
    input_list = [
        "data:weight:airframe:mass",
        "data:cost:v_cruise_design",
        "data:cost:prototype_number",
        "data:cost:production:number_aircraft_5_years",
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

    assert problem.get_val(
        "data:cost:production:flight_test_cost_per_unit", units="USD"
    ) == pytest.approx(4138.9, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_material_cost():
    input_list = [
        "data:weight:airframe:mass",
        "data:cost:v_cruise_design",
        "data:cost:production:number_aircraft_5_years",
        "data:cost:cpi_2012",
        "data:geometry:flap_type",
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

    assert problem.get_val(
        "data:cost:production:material_cost_per_unit", units="USD"
    ) == pytest.approx(66638.15, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_avionics_cost():
    ivc = om.IndepVarComp()
    ivc.add_output("data:cost:cpi_2012", val=1.4)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCAvionicsCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:production:avionics_cost_per_unit", units="USD"
    ) == pytest.approx(21000.0, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_certification_cost():
    input_list = [
        "data:cost:production:engineering_cost_per_unit",
        "data:cost:production:dev_support_cost_per_unit",
        "data:cost:production:flight_test_cost_per_unit",
        "data:cost:production:tooling_cost_per_unit",
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

    assert problem.get_val(
        "data:cost:production:certification_cost_per_unit", units="USD"
    ) == pytest.approx(1106881.2, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_landing_gear_cost_reduction():
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:landing_gear:type", val=0.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCLandingGearCostReduction(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:production:landing_gear_cost_reduction", units="USD"
    ) == pytest.approx(-7500.0, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_learning_curve_factor():
    ivc = om.IndepVarComp()
    ivc.add_output("data:cost:production:learning_curve_percentage", units="percent", val=85.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCLearningCurveFactor(),
        ivc,
    )

    assert problem.get_val("data:cost:production:learning_curve_factor") == pytest.approx(
        0.757, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_learning_curve_discount():
    ivc = om.IndepVarComp()
    ivc.add_output("data:cost:production:learning_curve_factor", val=0.757)
    ivc.add_output("data:cost:production:similar_aircraft_made", val=1500.0)
    ivc.add_output("data:cost:production:number_aircraft_5_years", val=50.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCLearningCurveDiscount(),
        ivc,
    )

    assert problem.get_val("data:cost:production:maturity_discount") == pytest.approx(
        0.4376, rel=1e-3
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
        "data:propulsion:he_power_train:propeller:propeller_1:purchase_cost",
        units="USD",
        val=4403.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:purchase_cost",
        units="USD",
        val=3.0e5,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:fuel_tank:fuel_tank_1:purchase_cost",
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
        3849700.62, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_aircraft_MSP():
    ivc = om.IndepVarComp()
    ivc.add_output("data:cost:production_cost_per_unit", units="USD", val=3849700.62)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCMSP(),
        ivc,
    )

    assert problem.get_val("data:cost:msp_per_unit", units="USD") == pytest.approx(
        4273167.688, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_delivery_duration_ratio():
    ivc = om.IndepVarComp()
    ivc.add_output("data:mission:sizing:duration", val=5.0, units="h")
    ivc.add_output("data:cost:production:delivery_duration", val=3.0, units="h")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCDeliveryDurationRatio(),
        ivc,
    )

    assert problem.get_val("data:cost:production:delivery_duration_ratio") == pytest.approx(
        0.6, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_delivery_cost():
    ivc = om.IndepVarComp()
    ivc.add_output("data:weight:aircraft:OWE", val=426.58, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCDeliveryCost(delivery_method="train"),
        ivc,
    )

    assert problem.get_val("data:cost:delivery_cost_per_unit", units="USD") == pytest.approx(
        581.21, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    ivc = om.IndepVarComp()
    ivc.add_output("data:cost:production:delivery_duration_ratio", val=1.0)
    ivc.add_output("data:cost:electricity_cost", units="USD", val=8.204)
    ivc.add_output("data:cost:fuel_cost", units="USD", val=91.56)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCDeliveryCost(),
        ivc,
    )

    assert problem.get_val("data:cost:delivery_cost_per_unit", units="USD") == pytest.approx(
        99.764, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_production_cost_hydrogen():
    ivc = get_indep_var_comp(
        list_inputs(
            LCCProductionCost(
                power_train_file_path=DATA_FOLDER_PATH / "propulsion_pemfc.yml",
                delivery_method="train",
            )
        ),
        __file__,
        "data_pemfc.xml",
    )
    ivc.add_output("data:weight:aircraft:OWE", val=426.58, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCProductionCost(
            power_train_file_path=DATA_FOLDER_PATH / "propulsion_pemfc.yml", delivery_method="train"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:gaseous_hydrogen_tank:gaseous_hydrogen_tank_1:purchase_cost",
        units="USD",
    ) == pytest.approx(104.25, rel=1e-3)

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack:pemfc_stack_1:purchase_cost", units="USD"
    ) == pytest.approx(13639.32, rel=1e-3)

    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:purchase_cost", units="USD"
    ) == pytest.approx(60485.41, rel=1e-3)

    assert problem.get_val("data:cost:production_cost_per_unit", units="USD") == pytest.approx(
        385693.87, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_production_cost_hybrid_tbm_900():
    ivc = get_indep_var_comp(
        list_inputs(
            LCCProductionCost(
                power_train_file_path=DATA_FOLDER_PATH / "turbo_electric_propulsion.yml",
                delivery_method="train",
            )
        ),
        __file__,
        "input_ecopulse.xml",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCProductionCost(
            power_train_file_path=DATA_FOLDER_PATH / "turbo_electric_propulsion.yml",
            delivery_method="train",
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:purchase_cost", units="USD"
    ) == pytest.approx(449215.82, rel=1e-3)
    assert problem.get_val(
        "data:propulsion:he_power_train:gearbox:gearbox_1:purchase_cost", units="USD"
    ) == pytest.approx(1959.02, rel=1e-3)
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:purchase_cost", units="USD"
    ) == pytest.approx(15762.91, rel=1e-3)
    assert problem.get_val(
        "data:propulsion:he_power_train:turbo_generator:turbo_generator:purchase_cost", units="USD"
    ) == pytest.approx(221207.71, rel=1e-3)

    assert problem.get_val("data:cost:production_cost_per_unit", units="USD") == pytest.approx(
        4452653.5, rel=1e-3
    )

    assert problem.get_val("data:cost:msp_per_unit", units="USD") == pytest.approx(
        4942445.38, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_production_cost_tbm_900():
    ivc = get_indep_var_comp(
        list_inputs(
            LCCProductionCost(
                power_train_file_path=DATA_FOLDER_PATH / "turboshaft_propulsion_tbm_900.yml",
                delivery_method="train",
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCProductionCost(
            power_train_file_path=DATA_FOLDER_PATH / "turboshaft_propulsion_tbm_900.yml",
            delivery_method="train",
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:purchase_cost", units="USD"
    ) == pytest.approx(449215.82, rel=1e-3)

    assert problem.get_val("data:cost:production_cost_per_unit", units="USD") == pytest.approx(
        4062984.46, rel=1e-3
    )

    assert problem.get_val("data:cost:msp_per_unit", units="USD") == pytest.approx(
        4509912.75, rel=1e-3
    )

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_annual_insurance_cost():
    ivc = om.IndepVarComp()

    ivc.add_output("data:cost:msp_per_unit", units="USD", val=412758.77)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCAnnualInsuranceCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:operation:annual_insurance_cost", units="USD/yr"
    ) == pytest.approx(4627.59, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_landing_cost():
    expect_mtow = [1.0, 2.0, 3.0, 7.0]
    expect_cost = [41.59, 53.8, 72.3, 66.53]
    for mtow, cost in zip(expect_mtow, expect_cost):
        ivc = om.IndepVarComp()
        ivc.add_output("data:weight:aircraft:MTOW", units="t", val=mtow)

        problem = run_system(
            LCCLandingCost(),
            ivc,
        )

        assert problem.get_val("data:cost:operation:landing_cost", units="USD") == pytest.approx(
            cost, rel=1e-3
        )

        problem.check_partials(compact_print=True)


def test_daily_parking_cost():
    expect_mtow = [1.0, 2.0, 3.0, 7.0]
    expect_cost = [2.132, 4.186, 7.2, 82.6]
    for mtow, cost in zip(expect_mtow, expect_cost):
        ivc = om.IndepVarComp()
        ivc.add_output("data:weight:aircraft:MTOW", units="t", val=mtow)

        problem = run_system(
            LCCDailyParkingCost(),
            ivc,
        )

        assert problem.get_val(
            "data:cost:operation:daily_parking_cost", units="USD/d"
        ) == pytest.approx(cost, rel=1e-3)

        problem.check_partials(compact_print=True)


def test_annual_crew_cost():
    ivc = om.IndepVarComp()

    ivc.add_output("data:cost:operation:number_of_pilot", val=1.0)
    ivc.add_output("data:cost:operation:number_of_cabin_crew", val=1.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCAnnualCrewCost(),
        ivc,
    )

    assert problem.get_val("data:cost:operation:annual_crew_cost", units="USD/yr") == pytest.approx(
        138756.7, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_annual_airport_cost():
    ivc = om.IndepVarComp()

    ivc.add_output("data:cost:operation:daily_parking_cost", units="USD/d", val=10.0)
    ivc.add_output("data:cost:operation:landing_cost", units="USD", val=50.0)
    ivc.add_output("data:TLAR:flight_per_year", units="1/yr", val=20.0)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCAnnualAirportCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:operation:annual_airport_cost", units="USD/yr"
    ) == pytest.approx(4650.0, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_annual_loan_cost():
    ivc = om.IndepVarComp()

    ivc.add_output("data:cost:operation:loan_principal", units="USD", val=1.0e6)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCAnnualLoanCost(),
        ivc,
    )

    assert problem.get_val("data:cost:operation:annual_loan_cost", units="USD/yr") == pytest.approx(
        109794.62, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_annual_depreciation_cost():
    ivc = om.IndepVarComp()

    ivc.add_output("data:cost:msp_per_unit", units="USD", val=412758.77)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCAnnualDepreciation(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:operation:annual_depreciation_cost", units="USD/yr"
    ) == pytest.approx(18574.15, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_annual_maintenance_cost():
    ivc = get_indep_var_comp(
        list_inputs(LCCMaintenanceCost()),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCMaintenanceCost(),
        ivc,
    )

    assert problem.get_val("data:cost:operation:maintenance_cost", units="USD/yr") == pytest.approx(
        77369.07, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_annual_maintenance_miscellaneous_cost():
    ivc = get_indep_var_comp(
        list_inputs(LCCMaintenanceMiscellaneousCost()),
        __file__,
        XML_FILE,
    )
    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCMaintenanceMiscellaneousCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:operation:miscellaneous_cost", units="USD/yr"
    ) == pytest.approx(21926.83, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_fuel_cost():
    tank_types = ["fuel_tank", "fuel_tank"]
    tank_names = ["fuel_tank_1", "fuel_tank_2"]
    fuel_types = ["avgas", "diesel"]

    ivc = get_indep_var_comp(
        list_inputs(
            LCCFuelCost(tank_types=tank_types, tank_names=tank_names, fuel_types=fuel_types)
        ),
        __file__,
        XML_FILE,
    )

    problem = run_system(
        LCCFuelCost(tank_types=tank_types, tank_names=tank_names, fuel_types=fuel_types),
        ivc,
    )

    assert problem.get_val("data:cost:fuel_cost", units="USD") == pytest.approx(2257.1, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_electricity_cost():
    components_type = ["battery_pack", "battery_pack"]
    components_name = ["battery_pack_1", "battery_pack_2"]

    ivc = om.IndepVarComp()

    ivc.add_output(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1" ":energy_consumed_mission",
        units="W*h",
        val=12525.12,
    )

    ivc.add_output(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2" ":energy_consumed_mission",
        units="W*h",
        val=12525.12,
    )

    problem = run_system(
        LCCElectricityCost(
            electricity_components_type=components_type, electricity_components_name=components_name
        ),
        ivc,
    )

    assert problem.get_val("data:cost:electricity_cost", units="USD") == pytest.approx(
        16.408, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_energy_cost():
    ivc = om.IndepVarComp()
    ivc.add_output("data:cost:electricity_cost", units="USD", val=8.204)
    ivc.add_output("data:cost:fuel_cost", units="USD", val=91.56)
    ivc.add_output("data:TLAR:flight_per_year", val=10.0)

    problem = run_system(
        LCCAnnualEnergyCost(),
        ivc,
    )

    assert problem.get_val(
        "data:cost:operation:annual_electricity_cost", units="USD/yr"
    ) == pytest.approx(82.04, rel=1e-3)

    assert problem.get_val("data:cost:operation:annual_fuel_cost", units="USD/yr") == pytest.approx(
        915.6, rel=1e-3
    )

    problem.check_partials(compact_print=True)


def test_operational_sum():
    components_type = ["propeller", "turboshaft"]
    components_name = ["propeller_1", "turboshaft_1"]

    ivc = get_indep_var_comp(
        list_inputs(
            LCCSumOperationalCost(
                cost_components_type=components_type, cost_components_name=components_name
            )
        ),
        __file__,
        XML_FILE,
    )

    ivc.add_output(
        "data:propulsion:he_power_train:propeller:propeller_1:operational_cost",
        units="USD/yr",
        val=4403.0,
    )
    ivc.add_output(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:operational_cost",
        units="USD/yr",
        val=3.0e5,
    )
    ivc.add_output(
        "data:cost:operation:annual_fuel_cost",
        units="USD/yr",
        val=1000.0,
    )
    ivc.add_output(
        "data:cost:operation:annual_electricity_cost",
        units="USD/yr",
        val=0.0,
    )

    problem = run_system(
        LCCSumOperationalCost(
            cost_components_type=components_type, cost_components_name=components_name
        ),
        ivc,
    )

    assert problem.get_val(
        "data:cost:operation:annual_cost_per_unit", units="USD/yr"
    ) == pytest.approx(569682.25, rel=1e-3)

    problem.check_partials(compact_print=True)


def test_operational_cost_hydrogen():
    ivc = get_indep_var_comp(
        list_inputs(
            LCCOperationalCost(power_train_file_path=DATA_FOLDER_PATH / "propulsion_pemfc.yml")
        ),
        __file__,
        "data_pemfc.xml",
    )
    ivc.add_output("data:weight:aircraft:OWE", val=426.58, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCOperationalCost(power_train_file_path=DATA_FOLDER_PATH / "propulsion_pemfc.yml"),
        ivc,
    )
    assert problem.get_val(
        "data:cost:operation:annual_fuel_cost",
        units="USD/yr",
    ) == pytest.approx(9171.9, rel=1e-3)
    assert problem.get_val(
        "data:cost:operation:annual_electricity_cost",
        units="USD/yr",
    ) == pytest.approx(0.0, rel=1e-3)

    assert problem.get_val(
        "data:propulsion:he_power_train:PEMFC_stack:pemfc_stack_1:operational_cost", units="USD/yr"
    ) == pytest.approx(309.0, rel=1e-3)

    assert problem.get_val(
        "data:cost:operation:annual_cost_per_unit", units="USD/yr"
    ) == pytest.approx(31543.8, rel=1e-3)

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_operational_cost_tbm_900():
    ivc = get_indep_var_comp(
        list_inputs(
            LCCOperationalCost(
                power_train_file_path=DATA_FOLDER_PATH / "turboshaft_propulsion_tbm_900.yml"
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCOperationalCost(
            power_train_file_path=DATA_FOLDER_PATH / "turboshaft_propulsion_tbm_900.yml"
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:operational_cost", units="USD/yr"
    ) == pytest.approx(33177.257, rel=1e-3)

    assert problem.get_val("data:cost:operation:annual_fuel_cost", units="USD/yr") == pytest.approx(
        156817.93, rel=1e-3
    )

    assert problem.get_val("data:cost:operation:maintenance_cost", units="USD/yr") == pytest.approx(
        77369.07, rel=1e-3
    )

    assert problem.get_val(
        "data:cost:operation:annual_cost_per_unit", units="USD/yr"
    ) == pytest.approx(457782.31, rel=1e-3)

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_operational_cost_hybrid_tbm_900():
    ivc = get_indep_var_comp(
        list_inputs(
            LCCOperationalCost(
                power_train_file_path=DATA_FOLDER_PATH / "turbo_electric_propulsion.yml"
            )
        ),
        __file__,
        "input_ecopulse.xml",
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCCOperationalCost(
            power_train_file_path=DATA_FOLDER_PATH / "turbo_electric_propulsion.yml"
        ),
        ivc,
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:turbo_generator:turbo_generator:operational_cost",
        units="USD/yr",
    ) == pytest.approx(50116.82, rel=1e-3)

    assert problem.get_val("data:cost:operation:maintenance_cost", units="USD/yr") == pytest.approx(
        93059.0, rel=1e-3
    )

    assert problem.get_val(
        "data:propulsion:he_power_train:rectifier:rectifier:operational_cost",
        units="USD/yr",
    ) == pytest.approx(166.5, rel=1e-3)

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_cost_tbm_900():
    ivc = get_indep_var_comp(
        list_inputs(
            LCC(
                power_train_file_path=DATA_FOLDER_PATH / "turboshaft_propulsion_tbm_900.yml",
                delivery_method="train",
            )
        ),
        __file__,
        XML_FILE,
    )

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCC(
            power_train_file_path=DATA_FOLDER_PATH / "turboshaft_propulsion_tbm_900.yml",
            delivery_method="train",
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:turboshaft:turboshaft_1:purchase_cost", units="USD"
    ) == pytest.approx(449215.82, rel=1e-3)

    assert problem.get_val("data:cost:production_cost_per_unit", units="USD") == pytest.approx(
        4062984.46, rel=1e-3
    )

    assert problem.get_val("data:cost:msp_per_unit", units="USD") == pytest.approx(
        4509912.75, rel=1e-3
    )

    assert problem.get_val(
        "data:cost:operation:annual_cost_per_unit", units="USD/yr"
    ) == pytest.approx(460976.68, rel=1e-3)

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))


def test_cost_pipistrel():
    ivc = get_indep_var_comp(
        list_inputs(
            LCC(
                power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
                delivery_method="train",
                learning_curve=True,
            )
        ),
        __file__,
        "pipistrel_source.xml",
    )

    oad.RegisterSubmodel.active_models[SERVICE_COST_CERTIFICATION] = None

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(
        LCC(
            power_train_file_path=DATA_FOLDER_PATH / "pipistrel_assembly.yml",
            delivery_method="train",
            learning_curve=True,
        ),
        ivc,
    )
    assert problem.get_val(
        "data:propulsion:he_power_train:PMSM:motor_1:purchase_cost", units="USD"
    ) == pytest.approx(3688.43, rel=1e-3)

    assert problem.get_val("data:cost:production_cost_per_unit", units="USD") == pytest.approx(
        237762.43, rel=1e-3
    )

    assert problem.get_val("data:cost:msp_per_unit", units="USD") == pytest.approx(
        263916.3, rel=1e-3
    )

    assert problem.get_val(
        "data:cost:operation:annual_cost_per_unit", units="USD/yr"
    ) == pytest.approx(35973.7, rel=1e-3)

    problem.check_partials(compact_print=True)

    om.n2(problem, show_browser=False, outfile=pth.join(pth.dirname(__file__), "n2.html"))
