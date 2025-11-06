# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import pathlib
import logging
import numpy as np

import fastoad.api as oad

from utils.filter_residuals import filter_residuals
from uqpce.pce.pce import PCE

DATA_FOLDER_PATH = pathlib.Path(__file__).parent / "data"

def test_sizing_sr22_uqpce():
    samp_count = 15
    aleat_cnt = 500000
    epist_cnt = 1

    pce = PCE(
        order=2, verbose=True, outputs=True, plot=False, aleat_samp_size=aleat_cnt,
        epist_samp_size=epist_cnt
    )

    # Define input distribution
    # unit conversion is not automated
    unit_list = ["NM", "kn", "kg"]
    variable_name = ['data:TLAR:range', 'data:TLAR:v_approach', 'data:TLAR:luggage_mass_design']
    mean_list = [850, 80, 55]
    stdev_list = [30, 4, 4]

    # Add normal variables
    for mean, std, name in zip(mean_list,stdev_list,variable_name):
        pce.add_variable(distribution='normal', mean=mean, stdev=std, name=name)


    # Generate samples that correspond to the input distributions
    Xt = np.array(pce.sample(count=samp_count))


    Yt = np.array([])
    for x in Xt:
        """Test the overall aircraft design process with wing positioning under VLM method."""
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("fastoad.module_management._bundle_loader").disabled = True
        logging.getLogger("fastoad.openmdao.variables.variable").disabled = True

        # Define used files depending on options
        xml_file_name = "input_sr22.xml"
        process_file_name = "full_sizing_fuel.yml"

        configurator = oad.FASTOADProblemConfigurator(DATA_FOLDER_PATH / process_file_name)
        problem = configurator.get_problem()

        ref_inputs = DATA_FOLDER_PATH / xml_file_name

        problem.write_needed_inputs(ref_inputs)
        problem.read_inputs()
        problem.setup()

        for name, val, unit in zip(variable_name, x, unit_list):
            problem.set_val(name, val=val, units=unit)

        problem.run_model()

        _, _, residuals = problem.model.get_nonlinear_vectors()
        residuals = filter_residuals(residuals)

        # problem.write_outputs()
        y = problem.get_val("data:weight:aircraft:MTOW", units="kg")
        Yt = np.append(Yt, y)


    pce.fit(Xt, Yt)  # Fit the PCE model
    pce.check_variables(Xt)  # Check if the samples correspond to the distributions
    pce.sobols()  # Calculate the Sobol indices
    cil, cih = pce.confidence_interval()  # Calculate the confidence interval

    print(f"CI low is {cil}, CI high is {cih}")

    # The output file will be stored under output folder
    pce.write_outputs()




