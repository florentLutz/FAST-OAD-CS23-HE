import os.path as pth
import logging
import fastoad.api as oad

DATA_FOLDER = "data"

WORK_FOLDER = "workdir"

CONFIGURATION_FILE_NAME = pth.join(DATA_FOLDER, "beam_problem.yml")

CUSTOM_MODULES_FOLDER_PATH = pth.join(
    pth.dirname(__file__), "../02_From_Python_to_FAST-OAD/modules"
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s: %(message)s")

oad.list_modules()


from IPython.display import IFrame

N2_FILE = pth.join(WORK_FOLDER, "n2.html")
oad.write_n2(CONFIGURATION_FILE_NAME, N2_FILE, overwrite=True)

IFrame(src=N2_FILE, width="100%", height="500px")

# UNCOMMENT THE FOLLOWING LINE TO GENERATE A BLANK INPUT FILE
# oad.generate_inputs(CONFIGURATION_FILE_NAME, overwrite=True)
INPUT_FILE_NAME = pth.join(DATA_FOLDER, "problem_inputs.xml")
oad.variable_viewer(INPUT_FILE_NAME)

"""oad.generate_inputs(CONFIGURATION_FILE_NAME, DATA_FOLDER,"problem_inputs.xml", overwrite=True)
oad.variable_viewer(INPUT_FILE_NAME)"""


disp_problem = oad.evaluate_problem(CONFIGURATION_FILE_NAME, overwrite=True)

oad.variable_viewer(disp_problem.output_file_path)
