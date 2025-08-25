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
    results_pipistrel_file_path = pth.join(DATA_FOLDER_PATH, "plotdelete.xml")

    fig = aircraft_geometry_plot(results_pipistrel_file_path, name="ATR42")

    fig.update_layout(
        height=1600,
        width=800,
        font_size=18,
    )

    pipistrel_top_view = Image.open("data/ATR42_topview3_correct.jpg")

    fig.add_layout_image(
        dict(
            source=pipistrel_top_view,
            xref="x",
            yref="y",
            y=4.43*5.12,
            x=-4.80*5.12 / 2,
            sizex=4.80*5.12,
            sizey=4.43*5.12,
            sizing="stretch",
            opacity=0.75,
            layer="below",
        )
    )

    # fig.show()
    fig.write_image(pth.join(RESULTS_FOLDER_PATH, "ATR42_geometry.svg"))
    fig.write_image(pth.join(RESULTS_FOLDER_PATH, "ATR42_geometry.pdf"))