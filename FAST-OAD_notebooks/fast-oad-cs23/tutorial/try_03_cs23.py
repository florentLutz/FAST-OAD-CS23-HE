import os.path as pth
import os
import shutil
import fastoad.api as oad

import fastga.utils.postprocessing.post_processing_api as api_plots
from fastga.command import api as api_cs23

# Define relative path
DATA_FOLDER_PATH = "data"
WORK_FOLDER_PATH = "workdir"

# Final file names
AIRCRAFT1_FILE = pth.join(WORK_FOLDER_PATH, "geometry_reference.xml")
AIRCRAFT2_FILE = pth.join(WORK_FOLDER_PATH, "geometry_long_wing.xml")

# Clear work folder
shutil.rmtree(WORK_FOLDER_PATH, ignore_errors=True)
os.mkdir(WORK_FOLDER_PATH)

# For using all screen width
# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:95% !important; }</style>"))

# Copy the reference geometry file (limited input parameters) as input file (name specified in .yml)
shutil.copy(
    pth.join(DATA_FOLDER_PATH, "reference_aircraft.xml"),
    pth.join(WORK_FOLDER_PATH, "geometry_inputs.xml"),
)

# Copy the .toml process file to the workdir
CONFIGURATION_FILE = pth.join(WORK_FOLDER_PATH, "geometry.yml")
shutil.copy(pth.join(DATA_FOLDER_PATH, "geometry.yml"), CONFIGURATION_FILE)

# Launch an evaluation to obtain the output file (name specified in the .toml)
eval_problem = oad.evaluate_problem(CONFIGURATION_FILE, overwrite=True)

# Copy this file to a different name to avoid an overwritte when computing secong geometry
shutil.copy(pth.join(WORK_FOLDER_PATH, "geometry_outputs.xml"), AIRCRAFT1_FILE)

from IPython.display import IFrame

N2_FILE = pth.join(WORK_FOLDER_PATH, "n2.html")
oad.write_n2(CONFIGURATION_FILE, N2_FILE, overwrite=True)

IFrame(src=N2_FILE, width="100%", height="500px")

from fastga.models.geometry.geometry import GeometryFixedTailDistance

# Copy reference aircraft file
shutil.copy(pth.join(DATA_FOLDER_PATH, "reference_aircraft.xml"), AIRCRAFT2_FILE)

# Define the wing primary geometry parameters name as a list
var_inputs = [
    "data:geometry:wing:area",
    "data:geometry:wing:aspect_ratio",
    "data:geometry:wing:taper_ratio",
]

# Declare function
compute_geometry = api_cs23.generate_block_analysis(
    GeometryFixedTailDistance(propulsion_id="fastga.wrapper.propulsion.basicIC_engine"),
    var_inputs,
    str(AIRCRAFT2_FILE),
    overwrite=True,
)

# Fetch reference aircraft values
reference_aircraft = oad.DataFile(AIRCRAFT1_FILE)

span_old = reference_aircraft["data:geometry:wing:span"].value[0]
taper_ratio_old = reference_aircraft["data:geometry:wing:taper_ratio"].value[0]
area_old = reference_aircraft["data:geometry:wing:area"].value[0]
y2_ref = reference_aircraft["data:geometry:wing:root:y"].value[0]
y4_ref = reference_aircraft["data:geometry:wing:tip:y"].value[0]


# Define functions
def taper_func(span_mult):
    return span_mult * taper_ratio_old + (1 - span_mult)


def area_func(new_taper, span_mult):
    return (
        (2 * y2_ref + (span_mult * y4_ref - y2_ref) * (1 + new_taper))
        / (2 * y2_ref + (y4_ref - y2_ref) * (1 + taper_ratio_old))
        * area_old
    )


# Calculate parameters
taper_ratio_new = taper_func(1.15)
area = area_func(taper_ratio_new, 1.15)
aspect_ratio_new = (span_old * 1.15) ** 2 / area

# Print results
print("area=" + str(area))
print("aspect_ratio=" + str(aspect_ratio_new))
print("taper_ratio=" + str(taper_ratio_new))

# Compute geometry
inputs_dict = {
    "data:geometry:wing:area": (area, "m**2"),
    "data:geometry:wing:aspect_ratio": (aspect_ratio_new, None),
    "data:geometry:wing:taper_ratio": (taper_ratio_new, None),
}
outputs_dict = compute_geometry(inputs_dict)

# Open viewer
oad.variable_viewer(AIRCRAFT2_FILE)
