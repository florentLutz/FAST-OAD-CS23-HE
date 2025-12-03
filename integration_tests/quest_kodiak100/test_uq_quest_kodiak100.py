#  This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
#  Electric Aircraft.
#  Copyright (C) 2025 ISAE-SUPAERO

import logging
import pathlib
import pytest

import numpy as np
import fastoad.api as oad
from uqpce import PCE

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"


def test_sizing_kodiak_100():
    """Test the overall aircraft design process with wing positioning."""
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
    logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

    # Define used files depending on options
    xml_file_name = "input_kodiak100.xml"
    process_file_name = "full_sizing_kodiak100.yml"

    configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)

    samp_count = 19
    aleat_cnt = 500000
    epist_cnt = 1

    pce = PCE(
        order=2,
        verbose=True,
        outputs=True,
        plot=False,
        aleat_samp_size=aleat_cnt,
        epist_samp_size=epist_cnt,

    )

    # Add normal variables for all the parameters we are uncertain of.
    # For the fuel consumption, we have already recalibrated the model for the Kodiak turbine so
    # we introduce uncertainties around that recalibrated value
    pce.add_variable(
        distribution="normal",
        mean=1.05,
        stdev=0.038,
        name="settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_sfc",
    )
    pce.add_variable(
        distribution="normal",
        mean=1.0,
        stdev=0.0882,
        name="settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_weight",
    )
    pce.add_variable(
        distribution="normal",
        mean=1.0,
        stdev=0.0694,
        name="settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_length",
    )
    pce.add_variable(
        distribution="normal",
        mean=1.0,
        stdev=0.0521,
        name="settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_diameter",
    )

    # Generate samples that correspond to the input distributions
    x_t = np.array(pce.sample(count=samp_count))
    mtow_t = np.array([])
    owe_t = np.array([])
    fuel_t = np.array([])
    wing_area_t = np.array([])

    for x in x_t:

        problem = configurator.get_problem()

        # Create inputs
        ref_inputs = DATA_FOLDER_PATH / xml_file_name

        problem.write_needed_inputs(ref_inputs)
        problem.read_inputs()
        problem.setup()

        problem.set_val("settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_sfc", val=x[0])
        problem.set_val("settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_weight", val=x[1])
        problem.set_val("settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_length", val=x[2])
        problem.set_val("settings:propulsion:he_power_train:turboshaft:turboshaft_1:k_diameter", val=x[3])

        problem.run_model()

        mtow_t = np.append(mtow_t, problem.get_val("data:weight:aircraft:MTOW", units="kg"))
        owe_t = np.append(owe_t, problem.get_val("data:weight:aircraft:OWE", units="kg"))
        fuel_t = np.append(fuel_t, problem.get_val("data:mission:sizing:fuel", units="kg"))
        wing_area_t = np.append(wing_area_t, problem.get_val("data:geometry:wing:area", units="m**2"))

    pce.fit(x_t, mtow_t)  # Fit the PCE model
    pce.check_variables(x_t)  # Check if the samples correspond to the distributions
    pce.sobols()  # Calculate the Sobol indices
    cil, cih = pce.confidence_interval()  # Calculate the confidence interval

    print(f"For MTOW: CI low is {cil}, CI high is {cih}")

    # The output file will be stored under output folder
    pce.write_outputs()

    pce.fit(x_t, owe_t)  # Fit the PCE model
    pce.check_variables(x_t)  # Check if the samples correspond to the distributions
    pce.sobols()  # Calculate the Sobol indices
    cil, cih = pce.confidence_interval()  # Calculate the confidence interval

    print(f"For OWE: CI low is {cil}, CI high is {cih}")

    # The output file will be stored under output folder
    pce.write_outputs()

    pce.fit(x_t, fuel_t)  # Fit the PCE model
    pce.check_variables(x_t)  # Check if the samples correspond to the distributions
    pce.sobols()  # Calculate the Sobol indices
    cil, cih = pce.confidence_interval()  # Calculate the confidence interval

    print(f"For fuel: CI low is {cil}, CI high is {cih}")

    # The output file will be stored under output folder
    pce.write_outputs()

    pce.fit(x_t, wing_area_t)  # Fit the PCE model
    pce.check_variables(x_t)  # Check if the samples correspond to the distributions
    pce.sobols()  # Calculate the Sobol indices
    cil, cih = pce.confidence_interval()  # Calculate the confidence interval

    print(f"For wing area: CI low is {cil}, CI high is {cih}")

    # The output file will be stored under output folder
    pce.write_outputs()