import os
import os.path as pth
import plotly.express as px
from PIL import Image
from fastoad.gui.analysis_and_plots import (
    aircraft_geometry_plot,
)


DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


COLORS = px.colors.qualitative.Prism



def test_aircraft_geometry_plot_ATR42():
    results_pipistrel_file_path = pth.join(DATA_FOLDER_PATH, "problem_outputs.xml")

    fig = aircraft_geometry_plot(results_pipistrel_file_path, name="ATR42")

    fig.update_layout(
        height=800,
        width=1600,
        font_size=18,
    )

    pipistrel_top_view = Image.open("data/ATR42_topview.jpg")

    fig.add_layout_image(
        dict(
            source=pipistrel_top_view,
            xref="x",
            yref="y",
            y=6.47,
            x=-10.71 / 2,
            sizex=10.71,
            sizey=6.47,
            sizing="stretch",
            opacity=0.75,
            layer="below",
        )
    )

    # fig.show()
    fig.write_image(pth.join(RESULTS_FOLDER_PATH, "ATR42_geometry.svg"))
    fig.write_image(pth.join(RESULTS_FOLDER_PATH, "ATR42_geometry.pdf"))