import os
import os.path as pth
import logging

import fastoad.api as oad

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


def test_dummy():

    """Test the overall aircraft design process with wing positioning under VLM method."""
    logging.basicConfig(level=logging.WARNING)

    # Define used files depending on options
    xml_file_name = "full_sizing.xml"
    process_file_name = "full_sizing.yml"

    configurator = oad.FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, process_file_name))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, xml_file_name)
    # api.list_modules(pth.join(DATA_FOLDER_PATH, process_file_name), force_text_output=True)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.run_model()
    problem.write_outputs()
