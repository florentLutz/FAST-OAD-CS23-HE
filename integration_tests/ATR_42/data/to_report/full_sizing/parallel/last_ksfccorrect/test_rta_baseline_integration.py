import os
import shutil
from shutil import rmtree
import logging
import os.path as pth
import numpy as np
import openmdao.api as om
import pandas as pd
import pytest
from fastoad.module_management._plugins import FastoadLoader
from numpy.testing import assert_allclose
import fastoad.api as oad
from fastoad import api
from fastoad.io import VariableIO
from fastoad.io.configuration.configuration import FASTOADProblemConfigurator
from fastga_he.gui.power_train_network_viewer import power_train_network_viewer


from utils.filter_residuals import filter_residuals

FastoadLoader()


DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
CONFIGURATION_FILE = "oad_process_float_performance_from_he.yml"
MISSION_FILE = "sizing_mission_R.yml"
SOURCE_FILE = "inputs_ATR42_hybrid.xml"
RESULTS_FOLDER = "problem_folder"


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_pipistrel_network_viewer():
    pt_file_path = pth.join(DATA_FOLDER_PATH, "pt_parallel.yml")
    network_file_path = pth.join(RESULTS_FOLDER_PATH, "ATR42_assembly_hybridPT.html")

    if not os.path.exists(network_file_path):
        power_train_network_viewer(pt_file_path, network_file_path)


def residuals_analyzer(recorder_path, solver):
    cr = om.CaseReader(recorder_path)

    solver_cases = cr.get_cases(solver)

    # Get only the last 10 cases (or all if less than 10)
    last_10_cases = solver_cases[-3:]

    variable_dict = {name: 0.0 for name in last_10_cases[-1].residuals}

    for case in last_10_cases:
        for residual in case.residuals:
            variable_dict[residual] = np.sum(np.abs(case.residuals[residual]))

    sorted_variable_dict = dict(sorted(variable_dict.items(), key=lambda x: x[1], reverse=True))

    return sorted_variable_dict


def test_non_regression_mission(cleanup):
    run_non_regression_test(
        CONFIGURATION_FILE,
        SOURCE_FILE,
        RESULTS_FOLDER,
        check_only_mtow=False,
        tolerance=1.0e-2,
    )


def run_non_regression_test(
    conf_file,
    legacy_result_file,
    result_dir,
    check_only_mtow=True,
    tolerance=5.0e-2,
):
    results_folder_path = pth.join(RESULTS_FOLDER_PATH, result_dir)
    configuration_file_path = pth.join(results_folder_path, conf_file)

    # Copy of configuration file and generation of problem instance ------------------
    api.generate_configuration_file(
        configuration_file_path, distribution_name="rta"
    )  # just ensure folders are created...
    shutil.copy(pth.join(DATA_FOLDER_PATH, conf_file), configuration_file_path)
    shutil.copy(
        pth.join(DATA_FOLDER_PATH, MISSION_FILE),
        pth.join(results_folder_path, MISSION_FILE),
    )
    configurator = FASTOADProblemConfigurator(configuration_file_path)

    # Generation and reading of inputs ----------------------------------------
    ref_inputs = pth.join(DATA_FOLDER_PATH, legacy_result_file)
    configurator.write_needed_inputs(ref_inputs)
    problem = configurator.get_problem()
    problem.read_inputs()
    problem.setup()

    # Run model ---------------------------------------------------------------
    problem.run_model()

    problem.write_outputs()

    try:
        problem.model.performance.flight_points.to_csv(
            pth.join(results_folder_path, "flight_points.csv"),
            sep="\t",
            decimal=",",
        )
    except AttributeError:
        pass
    om.view_connections(
        problem,
        outfile=pth.join(results_folder_path, "connections.html"),
        show_browser=False,
    )

    # Check that weight-performances loop correctly converged
    assert_allclose(
        problem["data:weight:aircraft:OWE"],
        problem["data:weight:airframe:mass"]
        + problem["data:weight:propulsion:mass"]
        + problem["data:weight:systems:mass"]
        + problem["data:weight:furniture:mass"]
        + problem["data:weight:operational:mass"],
        atol=1,
    )
    assert_allclose(
        problem["data:weight:aircraft:MZFW"],
        problem["data:weight:aircraft:OWE"] + problem["data:weight:aircraft:max_payload"],
        atol=1,
    )

    ref_var_list = VariableIO(
        pth.join(DATA_FOLDER_PATH, legacy_result_file),
    ).read()

    row_list = []
    for ref_var in ref_var_list:
        try:
            value = problem.get_val(ref_var.name, units=ref_var.units)[0]
        except KeyError:
            continue
        row_list.append(
            {
                "name": ref_var.name,
                "units": ref_var.units,
                "ref_value": ref_var.value[0],
                "value": value,
            }
        )

    df = pd.DataFrame(row_list)
    df["rel_delta"] = (df.value - df.ref_value) / df.ref_value
    df.loc[(df.ref_value == 0) & (abs(df.value) <= 1e-10), "rel_delta"] = 0.0
    df["abs_rel_delta"] = np.abs(df.rel_delta)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 120)
    print(df.sort_values(by=["abs_rel_delta"]))


"""
    if check_only_mtow:
        assert np.all(df.abs_rel_delta.loc[df.name == "data:weight:aircraft:MTOW"] < tolerance)
    else:
        assert np.all(df.abs_rel_delta < tolerance)
"""


def test_sizing_atr_42_retrofit():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = SOURCE_FILE
    process_file_name = CONFIGURATION_FILE

    configurator = api.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name))
    # print(oad.RegisterSubmodel.active_models["service.mass.propulsion"])

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_ATR42.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # problem.model_options["*"] = {
    #     "cg_names": [
    #             "data:weight:airframe:wing:CG:x",
    #             "data:weight:airframe:fuselage:CG:x",
    #             "data:weight:airframe:horizontal_tail:CG:x",
    #             "data:weight:airframe:vertical_tail:CG:x",
    #             "data:weight:airframe:flight_controls:CG:x",
    #             "data:weight:airframe:landing_gear:main:CG:x",
    #             "data:weight:airframe:landing_gear:front:CG:x",
    #             "data:weight:airframe:pylon:CG:x",
    #             "data:weight:airframe:paint:CG:x",
    #             "data:propulsion:he_power_train:CG:x",
    #             "data:weight:systems:power:auxiliary_power_unit:CG:x",
    #             "data:weight:systems:power:electric_systems:CG:x",
    #             "data:weight:systems:power:hydraulic_systems:CG:x",
    #             "data:weight:systems:life_support:insulation:CG:x",
    #             "data:weight:systems:life_support:air_conditioning:CG:x",
    #             "data:weight:systems:life_support:de-icing:CG:x",
    #             "data:weight:systems:life_support:cabin_lighting:CG:x",
    #             "data:weight:systems:life_support:seats_crew_accommodation:CG:x",
    #             "data:weight:systems:life_support:oxygen:CG:x",
    #             "data:weight:systems:life_support:safety_equipment:CG:x",
    #             "data:weight:systems:navigation:CG:x",
    #             "data:weight:systems:transmission:CG:x",
    #             "data:weight:systems:operational:radar:CG:x",
    #             "data:weight:systems:operational:cargo_hold:CG:x",
    #             "data:weight:systems:flight_kit:CG:x",
    #             "data:weight:furniture:passenger_seats:CG:x",
    #             "data:weight:furniture:food_water:CG:x",
    #             "data:weight:furniture:security_kit:CG:x",
    #             "data:weight:furniture:toilets:CG:x",
    #         ],
    #     "mass_names":  [
    #             "data:weight:airframe:wing:mass",
    #             "data:weight:airframe:fuselage:mass",
    #             "data:weight:airframe:horizontal_tail:mass",
    #             "data:weight:airframe:vertical_tail:mass",
    #             "data:weight:airframe:flight_controls:mass",
    #             "data:weight:airframe:landing_gear:main:mass",
    #             "data:weight:airframe:landing_gear:front:mass",
    #             "data:weight:airframe:pylon:mass",
    #             "data:weight:airframe:paint:mass",
    #             "data:propulsion:he_power_train:mass",
    #             "data:weight:systems:power:auxiliary_power_unit:mass",
    #             "data:weight:systems:power:electric_systems:mass",
    #             "data:weight:systems:power:hydraulic_systems:mass",
    #             "data:weight:systems:life_support:insulation:mass",
    #             "data:weight:systems:life_support:air_conditioning:mass",
    #             "data:weight:systems:life_support:de-icing:mass",
    #             "data:weight:systems:life_support:cabin_lighting:mass",
    #             "data:weight:systems:life_support:seats_crew_accommodation:mass",
    #             "data:weight:systems:life_support:oxygen:mass",
    #             "data:weight:systems:life_support:safety_equipment:mass",
    #             "data:weight:systems:navigation:mass",
    #             "data:weight:systems:transmission:mass",
    #             "data:weight:systems:operational:radar:mass",
    #             "data:weight:systems:operational:cargo_hold:mass",
    #             "data:weight:systems:flight_kit:mass",
    #             "data:weight:furniture:passenger_seats:mass",
    #             "data:weight:furniture:food_water:mass",
    #             "data:weight:furniture:security_kit:mass",
    #             "data:weight:furniture:toilets:mass",
    #         ],
    # }

    # problem.model_options["*"] = {
    #     "cell_capacity_ref": 5.0,
    #     "cell_weight_ref": 45.0e-3,
    # }

    # Change battery pack characteristics so that they match those of a high power,
    # lower capacity cell like the Samsung INR18650-25R, we also take the weight fraction of the
    # Pipistrel battery. Assumes same polarization curve
    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 45.0e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    # problem.setup()

    # problem.set_val(
    #     "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
    #     val=8.0,
    #     units="h**-1",
    # )
    # problem.set_val(
    #     "data:propulsion:he_power_train:battery_pack:battery_pack_2:cell:c_rate_caliber",
    #     val=8.0,
    #     units="h**-1",
    # )

    problem.setup()
    model = problem.model

    # Give good initial guess on a few key value to reduce the time it takes to converge

    # problem.set_val("data:geometry:wing:MAC:at25percent:x", units="m", val=10.0)
    # problem.set_val("data:geometry:wing:area", units="m**2", val=54.0)
    # problem.set_val("data:geometry:horizontal_tail:area", units="m**2", val=11.0)
    # problem.set_val("data:geometry:vertical_tail:area", units="m**2", val=11.0)

    # problem.set_val("data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=100.0)
    # problem.set_val("data:propulsion:he_power_train:battery_pack:battery_pack_2:number_modules", val=100.0)

    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=18000.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=10000.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=13000.0)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=14000.0)

    problem.set_val("data:weight:aircraft_empty:mass", units="kg", val=10855.697082198)
    problem.set_val("data:weight:aircraft_empty:CG:x", units="m", val=10.114459353773869)

    # recorder = om.SqliteRecorder(pth.join(DATA_FOLDER_PATH, "cases.sql"))
    # solver = model.aircraft_sizing.performances.solve_equilibrium.compute_dep_equilibrium.nonlinear_solver
    # solver.add_recorder(recorder)
    # solver.recording_options["record_solver_residuals"] = True
    # solver.recording_options["record_outputs"] = True

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # sorted_variable_residuals = residuals_analyzer(recorder, problem)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    # Construct the file path
    file_path = os.path.join(RESULTS_FOLDER_PATH, "cases.csv")

    # Open the file for writing
    # with open(file_path, "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #
    #     # Write the header
    #     writer.writerow(["Variable name", "Residuals"])
    #
    #     # Write the sum of residuals for each iteration
    #     for name, sum_res in sorted_variable_residuals.items():
    #         writer.writerow([name, sum_res])

    # cr = om.CaseReader(recorder)

    problem.write_outputs()


def test_sizing_atr_42_fullsizing():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = SOURCE_FILE
    process_file_name = CONFIGURATION_FILE

    configurator = api.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name))
    # print(oad.RegisterSubmodel.active_models["service.mass.propulsion"])

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_ATR42.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    # problem.model_options["*"] = {
    #     "cg_names": [
    #             "data:weight:airframe:wing:CG:x",
    #             "data:weight:airframe:fuselage:CG:x",
    #             "data:weight:airframe:horizontal_tail:CG:x",
    #             "data:weight:airframe:vertical_tail:CG:x",
    #             "data:weight:airframe:flight_controls:CG:x",
    #             "data:weight:airframe:landing_gear:main:CG:x",
    #             "data:weight:airframe:landing_gear:front:CG:x",
    #             "data:weight:airframe:pylon:CG:x",
    #             "data:weight:airframe:paint:CG:x",
    #             "data:propulsion:he_power_train:CG:x",
    #             "data:weight:systems:power:auxiliary_power_unit:CG:x",
    #             "data:weight:systems:power:electric_systems:CG:x",
    #             "data:weight:systems:power:hydraulic_systems:CG:x",
    #             "data:weight:systems:life_support:insulation:CG:x",
    #             "data:weight:systems:life_support:air_conditioning:CG:x",
    #             "data:weight:systems:life_support:de-icing:CG:x",
    #             "data:weight:systems:life_support:cabin_lighting:CG:x",
    #             "data:weight:systems:life_support:seats_crew_accommodation:CG:x",
    #             "data:weight:systems:life_support:oxygen:CG:x",
    #             "data:weight:systems:life_support:safety_equipment:CG:x",
    #             "data:weight:systems:navigation:CG:x",
    #             "data:weight:systems:transmission:CG:x",
    #             "data:weight:systems:operational:radar:CG:x",
    #             "data:weight:systems:operational:cargo_hold:CG:x",
    #             "data:weight:systems:flight_kit:CG:x",
    #             "data:weight:furniture:passenger_seats:CG:x",
    #             "data:weight:furniture:food_water:CG:x",
    #             "data:weight:furniture:security_kit:CG:x",
    #             "data:weight:furniture:toilets:CG:x",
    #         ],
    #     "mass_names":  [
    #             "data:weight:airframe:wing:mass",
    #             "data:weight:airframe:fuselage:mass",
    #             "data:weight:airframe:horizontal_tail:mass",
    #             "data:weight:airframe:vertical_tail:mass",
    #             "data:weight:airframe:flight_controls:mass",
    #             "data:weight:airframe:landing_gear:main:mass",
    #             "data:weight:airframe:landing_gear:front:mass",
    #             "data:weight:airframe:pylon:mass",
    #             "data:weight:airframe:paint:mass",
    #             "data:propulsion:he_power_train:mass",
    #             "data:weight:systems:power:auxiliary_power_unit:mass",
    #             "data:weight:systems:power:electric_systems:mass",
    #             "data:weight:systems:power:hydraulic_systems:mass",
    #             "data:weight:systems:life_support:insulation:mass",
    #             "data:weight:systems:life_support:air_conditioning:mass",
    #             "data:weight:systems:life_support:de-icing:mass",
    #             "data:weight:systems:life_support:cabin_lighting:mass",
    #             "data:weight:systems:life_support:seats_crew_accommodation:mass",
    #             "data:weight:systems:life_support:oxygen:mass",
    #             "data:weight:systems:life_support:safety_equipment:mass",
    #             "data:weight:systems:navigation:mass",
    #             "data:weight:systems:transmission:mass",
    #             "data:weight:systems:operational:radar:mass",
    #             "data:weight:systems:operational:cargo_hold:mass",
    #             "data:weight:systems:flight_kit:mass",
    #             "data:weight:furniture:passenger_seats:mass",
    #             "data:weight:furniture:food_water:mass",
    #             "data:weight:furniture:security_kit:mass",
    #             "data:weight:furniture:toilets:mass",
    #         ],
    # }

    # problem.model_options["*"] = {
    #     "cell_capacity_ref": 5.0,
    #     "cell_weight_ref": 45.0e-3,
    # }

    problem.model_options["*"] = {
        "cell_capacity_ref": 2.5,
        "cell_weight_ref": 30e-3,
        "reference_curve_current": [500, 5000, 10000, 15000, 20000],
        "reference_curve_relative_capacity": [1.0, 0.97, 1.0, 0.97, 0.95],
    }

    problem.setup()
    model = problem.model

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:cell:c_rate_caliber",
        val=8.0,
        units="h**-1",
    )

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:cell:c_rate_caliber",
        val=8.0,
        units="h**-1",
    )
    # Give good initial guess on a few key value to reduce the time it takes to converge

    problem.set_val("data:geometry:wing:MAC:at25percent:x", units="m", val=10.0)
    problem.set_val("data:geometry:wing:area", units="m**2", val=53.3)
    problem.set_val("data:geometry:horizontal_tail:area", units="m**2", val=10.645)
    problem.set_val("data:geometry:vertical_tail:area", units="m**2", val=11.0)

    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_1:number_modules", val=3100.0
    )
    problem.set_val(
        "data:propulsion:he_power_train:battery_pack:battery_pack_2:number_modules", val=3100.0
    )

    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=18000.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=11659.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=17159.5)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=18189.10294470838)

    problem.set_val("data:weight:aircraft_empty:mass", units="kg", val=11414.2)
    problem.set_val("data:weight:aircraft_empty:CG:x", units="m", val=10.514757)

    problem.set_val(
        "subgroup.performances.solve_equilibrium.update_mass.mass",
        units="kg",
        val=np.linspace(18000, 16000, 90),
    )

    datafile = oad.DataFile(
        "C:/Users/a.carotenuto/Documents/GitHub/FAST-OAD-CS23-HE/integration_tests/ATR_42/data/to_report/retrofit/parallel/ATR42_retrofit_outputs.xml"
    )

    list_of_variables_to_set = [
        "data:weight:airframe:wing:mass",
        "data:weight:airframe:fuselage:mass",
        "data:weight:airframe:horizontal_tail:mass",
        "data:weight:airframe:vertical_tail:mass",
        "data:weight:airframe:landing_gear:main:mass",
        "data:weight:airframe:landing_gear:front:mass",
        "data:propulsion:he_power_train:mass",
        "data:weight:systems:auxiliary_power_unit:mass",
        "data:weight:systems:electric_systems:electric_generation:mass",
        "data:weight:systems:electric_systems:electric_common_installation:mass",
        "data:weight:systems:hydraulic_systems:mass",
        "data:weight:systems:fire_protection:mass",
        "data:weight:systems:flight_furnishing:mass",
        "data:weight:systems:automatic_flight_system:mass",
        "data:weight:systems:communications:mass",
        "data:weight:systems:ECS:mass",
        "data:weight:systems:de-icing:mass",
        "data:weight:systems:navigation:mass",
        "data:weight:systems:flight_controls:mass",
        "data:weight:furniture:furnishing:mass",
        "data:weight:furniture:water:mass",
        "data:weight:furniture:interior_integration:mass",
        "data:weight:furniture:insulation:mass",
        "data:weight:furniture:cabin_lighting:mass",
        "data:weight:furniture:seats_crew_accommodation:mass",
        "data:weight:furniture:oxygen:mass",
        "data:weight:operational:items:passenger_seats:mass",
        "data:weight:operational:items:unusable_fuel:mass",
        "data:weight:operational:items:documents_toolkit:mass",
        "data:weight:operational:items:galley_structure:mass",
        "data:weight:operational:equipment:others:mass",
    ]

    for names in list_of_variables_to_set:
        problem.set_val(names, datafile[names].value, units=datafile[names].units)

    # recorder = om.SqliteRecorder(pth.join(DATA_FOLDER_PATH, "cases.sql"))
    # solver = model.aircraft_sizing.performances.solve_equilibrium.compute_dep_equilibrium.nonlinear_solver
    # solver.add_recorder(recorder)
    # solver.recording_options["record_solver_residuals"] = True
    # solver.recording_options["record_outputs"] = True

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # sorted_variable_residuals = residuals_analyzer(recorder, problem)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    # Construct the file path
    file_path = os.path.join(RESULTS_FOLDER_PATH, "cases.csv")

    # Open the file for writing
    # with open(file_path, "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #
    #     # Write the header
    #     writer.writerow(["Variable name", "Residuals"])
    #
    #     # Write the sum of residuals for each iteration
    #     for name, sum_res in sorted_variable_residuals.items():
    #         writer.writerow([name, sum_res])

    # cr = om.CaseReader(recorder)

    problem.write_outputs()


def test_sizing_atr_42_turboshaft():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = SOURCE_FILE
    process_file_name = CONFIGURATION_FILE

    configurator = api.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))
    problem = configurator.get_problem()

    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name))
    # print(oad.RegisterSubmodel.active_models["service.mass.propulsion"])

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    n2_path = pth.join(RESULTS_FOLDER_PATH, "n2_ATR42.html")
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.model_options["*"] = {
        "cell_capacity_ref": 5.0,
        "cell_weight_ref": 45.0e-3,
    }

    problem.setup()
    model = problem.model

    # Give good initial guess on a few key value to reduce the time it takes to converge

    problem.set_val("data:geometry:wing:MAC:at25percent:x", units="m", val=10.0)
    problem.set_val("data:geometry:wing:area", units="m**2", val=53.3)
    problem.set_val("data:geometry:horizontal_tail:area", units="m**2", val=10.645)
    problem.set_val("data:geometry:vertical_tail:area", units="m**2", val=11.0)

    problem.set_val("data:weight:aircraft:MTOW", units="kg", val=18000.0)
    problem.set_val("data:weight:aircraft:OWE", units="kg", val=11659.0)
    problem.set_val("data:weight:aircraft:MZFW", units="kg", val=17159.5)
    problem.set_val("data:weight:aircraft:MLW", units="kg", val=18189.10294470838)

    problem.set_val("data:weight:aircraft_empty:mass", units="kg", val=11414.2)
    problem.set_val("data:weight:aircraft_empty:CG:x", units="m", val=10.514757)

    problem.set_val(
        "subgroup.performances.solve_equilibrium.update_mass.mass",
        units="kg",
        val=np.linspace(18000, 16000, 90),
    )

    # recorder = om.SqliteRecorder(pth.join(DATA_FOLDER_PATH, "cases.sql"))
    # solver = model.aircraft_sizing.performances.solve_equilibrium.compute_dep_equilibrium.nonlinear_solver
    # solver.add_recorder(recorder)
    # solver.recording_options["record_solver_residuals"] = True
    # solver.recording_options["record_outputs"] = True

    om.n2(problem, show_browser=False, outfile=n2_path)

    problem.run_model()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)

    # sorted_variable_residuals = residuals_analyzer(recorder, problem)

    # Create the folder if it doesn't exist
    os.makedirs(RESULTS_FOLDER_PATH, exist_ok=True)

    # Construct the file path
    file_path = os.path.join(RESULTS_FOLDER_PATH, "cases.csv")

    problem.write_outputs()


def test_mission_vector_atr_42():
    # Define used files depending on options
    xml_file_name = "atr42_retrofit.xml"
    process_file_name = "atr42_retrofit.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)

    # Create problems with inputs
    problem = configurator.get_problem()
    problem.write_needed_inputs(ref_inputs)
    problem.read_inputs()

    problem.setup()

    # problem.set_val("data:weight:aircraft:MTOW", units="kg", val=15000.0)
    # problem.set_val("data:weight:aircraft:OWE", units="kg", val=10000.0)
    problem.set_val("data:weight:aircraft_empty:mass", units="kg", val=10855.697082198)
    problem.set_val("data:weight:aircraft_empty:CG:x", units="m", val=10.114459353773869)
    # problem.set_val("data:weight:aircraft:MZFW", units="kg", val=13000.0)
    # problem.set_val("data:weight:aircraft:MLW", units="kg", val=14000.0)

    # om.n2(problem, show_browser=True)

    problem.run_model()
    problem.write_outputs()

    _, _, residuals = problem.model.get_nonlinear_vectors()
    residuals = filter_residuals(residuals)
