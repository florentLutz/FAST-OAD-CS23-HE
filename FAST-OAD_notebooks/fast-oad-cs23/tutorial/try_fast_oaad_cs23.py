import warnings

warnings.filterwarnings(action="ignore")

import os
import os.path as pth
from fastoad import api as api_cs25
import logging
import shutil

from IPython.core.display import display, HTML

# Define relative path
DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
WORK_FOLDER_PATH = pth.join(pth.dirname(__file__), "workdir")

# Remove work folder
shutil.rmtree(WORK_FOLDER_PATH, ignore_errors=True)
os.mkdir(WORK_FOLDER_PATH)


# Define files
DATA_CONFIGURATION_FILE = pth.join(DATA_FOLDER_PATH, "oad_process.yml")
CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "oad_process.yml")
SOURCE_FILE = pth.join(DATA_FOLDER_PATH, "beechcraft_76.xml")

# For having log messages on screen
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s: %(message)s")

# For using all screen width
display(HTML("<style>.container { width:95% !important; }</style>"))

shutil.copy(DATA_CONFIGURATION_FILE, CONFIGURATION_FILE)
# api_cs25.generate_configuration_file(
#     CONFIGURATION_FILE,
#     overwrite=True,
#     distribution_name="fast-oad-cs23",
#     sample_file_name="fastga.yml",
# )

api_cs25.generate_inputs(CONFIGURATION_FILE, SOURCE_FILE, overwrite=True)

# api_cs25.list_variables(CONFIGURATION_FILE)

from IPython.display import IFrame

N2_FILE = pth.join(WORK_FOLDER_PATH, "n2.html")
api_cs25.write_n2(CONFIGURATION_FILE, N2_FILE, overwrite=True)

IFrame(src=N2_FILE, width="100%", height="500px")

from IPython.display import IFrame

XDSM_FILE = pth.join(WORK_FOLDER_PATH, "xdsm.html")
api_cs25.write_xdsm(CONFIGURATION_FILE, XDSM_FILE, overwrite=True)

IFrame(src=XDSM_FILE, width="100%", height="500px")
eval_problem = api_cs25.evaluate_problem(CONFIGURATION_FILE, overwrite=True)
OUTPUT_FILE = pth.join(WORK_FOLDER_PATH, "problem_outputs.xml")
Beechcraft_OUTPUT_FILE = pth.join(WORK_FOLDER_PATH, "problem_outputs_Beechcraft_800nm_mda.xml")
shutil.copy(OUTPUT_FILE, Beechcraft_OUTPUT_FILE)

api_cs25.variable_viewer(OUTPUT_FILE)


##

CONFIGURATION_FILE_MDO = pth.join(WORK_FOLDER_PATH, "oad_process_mdo.yml")
SOURCE_FILE_MDO = pth.join(WORK_FOLDER_PATH, "problem_outputs_Beechcraft_800nm_mda.xml")
shutil.copy(pth.join(DATA_FOLDER_PATH, "fastga_mdo.yml"), CONFIGURATION_FILE_MDO)

# Set back the inputs from the reference Beechcraft 800 nm
api_cs25.generate_inputs(CONFIGURATION_FILE_MDO, SOURCE_FILE_MDO, overwrite=True)
api_cs25.optimization_viewer(CONFIGURATION_FILE_MDO)
optim_problem = api_cs25.optimize_problem(CONFIGURATION_FILE_MDO, overwrite=True)

OUTPUT_FILE = pth.join(WORK_FOLDER_PATH, "problem_outputs.xml")
CeRAS_OPT_OUTPUT_FILE = pth.join(WORK_FOLDER_PATH, "problem_outputs_Beechcraft_800nm_mdo.xml")
shutil.copy(OUTPUT_FILE, CeRAS_OPT_OUTPUT_FILE)

api_cs25.optimization_viewer(CONFIGURATION_FILE_MDO)

RESULT_FILE = pth.join(WORK_FOLDER_PATH, "problem_outputs.xml")
api_cs25.variable_viewer(RESULT_FILE)
