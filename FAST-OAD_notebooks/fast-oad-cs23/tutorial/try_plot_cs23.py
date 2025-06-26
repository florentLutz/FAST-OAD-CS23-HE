import os.path as pth

import fastoad.api as oad
import fastga.utils.postprocessing.post_processing_api as api_plots

# For using all screen width
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:95% !important; }</style>"))

DATA_FOLDER_PATH = "data"
WORK_FOLDER_PATH = "workdir"

Beechcraft_800nm_MDA_OUTPUT_FILE = pth.join(
    WORK_FOLDER_PATH, "problem_outputs_Beechcraft_800nm_mda.xml"
)
Beechcraft_1000nm_MDA_OUTPUT_FILE = pth.join(
    WORK_FOLDER_PATH, "problem_outputs_Beechcraft_1000nm_mda.xml"
)
Beechcraft_800nm_MDO_OUTPUT_FILE = pth.join(
    WORK_FOLDER_PATH, "problem_outputs_Beechcraft_800nm_mdo.xml"
)

"""fig = api_plots.aircraft_geometry_plot(
    Beechcraft_800nm_MDA_OUTPUT_FILE, name="Beechcraft 800 nm MDA"
)
fig = api_plots.aircraft_geometry_plot(
    Beechcraft_1000nm_MDA_OUTPUT_FILE, name="Beechcraft 1000 nm MDA", fig=fig
)
fig = api_plots.aircraft_geometry_plot(
    Beechcraft_800nm_MDO_OUTPUT_FILE, name="Beechcraft 800 nm MDO", fig=fig
)
fig.show()"""
fig = oad.wing_geometry_plot(Beechcraft_800nm_MDA_OUTPUT_FILE, name="Beechcraft 800 nm MDA")
fig = oad.wing_geometry_plot(
    Beechcraft_1000nm_MDA_OUTPUT_FILE, name="Beechcraft 1000 nm MDA", fig=fig
)
fig = oad.wing_geometry_plot(
    Beechcraft_800nm_MDO_OUTPUT_FILE, name="Beechcraft 800 nm MDO", fig=fig
)
fig.show()


fig = api_plots.mass_breakdown_sun_plot(Beechcraft_800nm_MDA_OUTPUT_FILE)
fig.show()

fig = api_plots.mass_breakdown_bar_plot(
    Beechcraft_800nm_MDA_OUTPUT_FILE, name="Beechcraft 800 nm MDA"
)
fig = api_plots.mass_breakdown_bar_plot(
    Beechcraft_1000nm_MDA_OUTPUT_FILE, name="Beechcraft 1000 nm MDA", fig=fig
)
fig = api_plots.mass_breakdown_bar_plot(
    Beechcraft_800nm_MDO_OUTPUT_FILE, name="Beechcraft 800 nm MDO", fig=fig
)
fig.show()

Beechcraft_800nm_MDA_MISSION_FILE = pth.join(WORK_FOLDER_PATH, "workdir/mda_mission_results.csv")

mission = oad.MissionViewer()
mission.add_mission(Beechcraft_800nm_MDA_MISSION_FILE, name="Beechracft 800 nm MDA")

mission.missions["Beechracft 800 nm MDA"]

mission.display()

import shutil

CONFIGURATION_FILE_DATA = pth.join(DATA_FOLDER_PATH, "payload_range.yml")
CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "payload_range.yml")

SOURCE_FILE_PAYLOAD_RANGE = pth.join(WORK_FOLDER_PATH, "payload_range.xml")

# First copy the configuration file inside the workfolder, and create a duplicate of the MDA outputs

shutil.copy(CONFIGURATION_FILE_DATA, CONFIGURATION_FILE)
shutil.copy(Beechcraft_800nm_MDA_OUTPUT_FILE, SOURCE_FILE_PAYLOAD_RANGE)

# Then generate the inputs

oad.generate_inputs(CONFIGURATION_FILE, SOURCE_FILE_PAYLOAD_RANGE, overwrite=True)

eval_problem = oad.evaluate_problem(CONFIGURATION_FILE, overwrite=True)

OUTPUT_FILE = pth.join(WORK_FOLDER_PATH, "payload_range_outputs.xml")

fig = api_plots.payload_range(OUTPUT_FILE, name="Beechcraft 800 nm MDA")
fig.show()

import shutil

CONFIGURATION_FILE_DATA = pth.join(DATA_FOLDER_PATH, "aircraft_polar.yml")
CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "aircraft_polar.yml")

SOURCE_FILE_POLAR = pth.join(WORK_FOLDER_PATH, "polar_inputs.xml")

# First copy the configuration file inside the workfolder, and create a duplicate of the MDA outputs

shutil.copy(CONFIGURATION_FILE_DATA, CONFIGURATION_FILE)
shutil.copy(Beechcraft_800nm_MDA_OUTPUT_FILE, SOURCE_FILE_POLAR)

# Then generate the inputs

oad.generate_inputs(CONFIGURATION_FILE, SOURCE_FILE_POLAR, overwrite=True)

eval_problem = oad.evaluate_problem(CONFIGURATION_FILE, overwrite=True)

OUTPUT_FILE = pth.join(WORK_FOLDER_PATH, "polar_outputs.xml")

fig = api_plots.aircraft_polar(OUTPUT_FILE, name="Beechcraft 800 nm MDA")
fig.show()

fig = api_plots.aircraft_polar(OUTPUT_FILE, name="Beechcraft 800 nm MDA", equilibrated=True)
fig.show()

fig = api_plots.propeller_efficiency_map_plot(Beechcraft_800nm_MDA_OUTPUT_FILE, sea_level=True)
fig.show()

fig = api_plots.propeller_efficiency_map_plot(Beechcraft_800nm_MDA_OUTPUT_FILE, sea_level=False)
fig.show()

CONFIGURATION_FILE_DATA = pth.join(DATA_FOLDER_PATH, "propeller_coefficients.yml")
CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "propeller_coefficients.yml")

INPUT_FILE_PROP_DATA = pth.join(DATA_FOLDER_PATH, "propeller_coeff_inputs.xml")
INPUT_FILE_PROP = pth.join(WORK_FOLDER_PATH, "propeller_coeff_inputs.xml")

# First copy the configuration file and the input file inside the workfolder

shutil.copy(CONFIGURATION_FILE_DATA, CONFIGURATION_FILE)
shutil.copy(INPUT_FILE_PROP_DATA, INPUT_FILE_PROP)

eval_problem = oad.evaluate_problem(CONFIGURATION_FILE, overwrite=True)

OUTPUT_FILE = pth.join(WORK_FOLDER_PATH, "propeller_coeff_outputs.xml")

fig = api_plots.propeller_coeff_map_plot(OUTPUT_FILE)
fig.show()
