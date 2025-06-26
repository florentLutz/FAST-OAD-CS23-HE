import fastoad.api as oad
import os.path as path
import numpy as np

CUSTOM_MODULES_FOLDER_PATH = path.join(path.dirname(__file__), "modules")

# oad.list_modules(CUSTOM_MODULES_FOLDER_PATH)


CONFIGURATION_FILE_GLOBAL_MDA = path.join(
    path.dirname(__file__), "./working_folder/MDA_global_group/mda_global_group.yml"
)

from IPython.display import IFrame

# Writes the N2 chart
N2_FILE = path.join(path.dirname(__file__), "./working_folder/MDA_global_group/n2.html")
oad.write_n2(CONFIGURATION_FILE_GLOBAL_MDA, N2_FILE, overwrite=True)

# Displays it
IFrame(src=N2_FILE, width="100%", height="500px")

# To generate an empty configuration file. The location of the file is specified inside the configuration
# file. Here we decided to put it next to the configuration file.
# UNCOMMENT NEXT LINE IF YOU DARE TO TRY THIS METHOD.
# oad.generate_inputs(CONFIGURATION_FILE_GLOBAL_MDA, overwrite=True)

# The command to generate the input data file based on the source is very similar, we only need to provide the path to the
# source data file and the newly generated input data file will be placed at the location specified inside the configuration file.
SOURCE_FILE = "./data/source_file.xml"
oad.generate_inputs(CONFIGURATION_FILE_GLOBAL_MDA, SOURCE_FILE, overwrite=True)

mda_global = oad.evaluate_problem(CONFIGURATION_FILE_GLOBAL_MDA, overwrite=True)

print("MTOW:", float(np.round(mda_global.get_val("mtow", units="kg"), 1)), "kg")
print("Mission fuel:", float(np.round(mda_global.get_val("mission_fuel", units="kg"), 1)), "kg")
print("Wing mass:", float(np.round(mda_global.get_val("wing_mass", units="kg"), 1)), "kg")

# oad.variable_viewer(mda_global.output_file_path)


CONFIGURATION_FILE_MDO_MTOW = "./working_folder/MDO_MTOW/mdo_3var.yml"

# Generate the input data file based on configuration and source data file
oad.generate_inputs(CONFIGURATION_FILE_MDO_MTOW, SOURCE_FILE, overwrite=True)

mdo_3var = oad.optimize_problem(CONFIGURATION_FILE_MDO_MTOW, overwrite=True)
print("MTOW:", float(np.round(mdo_3var.get_val("mtow", units="kg"), 1)), "kg")
print("AR:", float(np.round(mdo_3var.get_val("aspect_ratio"), 3)))
print("Mission Fuel:", float(np.round(mdo_3var.get_val("mission_fuel", units="kg"), 5)), "kg")
print("Cruise speed:", float(np.round(mdo_3var.get_val("cruise_speed", units="m/s"), 1)), "m/s")
