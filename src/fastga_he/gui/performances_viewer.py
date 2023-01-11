# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import clear_output, display

from fastga_he.powertrain_builder.powertrain import PROMOTION_FROM_MISSION

COLOR_DICTIONARY = {
    "sizing:main_route:climb": px.colors.qualitative.Prism[3],
    "sizing:main_route:cruise": px.colors.qualitative.Prism[4],
    "sizing:main_route:descent": px.colors.qualitative.Prism[5],
    "sizing:main_route:reserve": px.colors.qualitative.Prism[6],
}


class PerformancesViewer:
    """
    A class for displaying the performances of all the elements of the power train along with the
    performances at aircraft level. Vastly inspired by the MissionViewer from fast-oad-core.
    """

    def __init__(
        self,
        power_train_data_file_path: str,
        mission_data_file_path: str = "NeFinisPasPar.csv!",
        plot_height: int = None,
        plot_width: int = None,
    ):

        if power_train_data_file_path.endswith(".csv") and not mission_data_file_path.endswith(
            ".csv"
        ):

            power_train_data = pd.read_csv(power_train_data_file_path, index_col=0)
            # Remove the taxi power train data because they are not stored in the mission data
            # either
            power_train_data = power_train_data.drop([0]).iloc[:-1]
            # We readjust the index
            power_train_data = power_train_data.set_index(np.arange(len(power_train_data.index)))

            all_data = power_train_data

        elif power_train_data_file_path.endswith(".csv") and mission_data_file_path.endswith(
            ".csv"
        ):

            columns_to_drop = []
            for mission_variable_name in list(PROMOTION_FROM_MISSION.keys()):
                columns_to_drop.append(
                    mission_variable_name
                    + " ["
                    + PROMOTION_FROM_MISSION[mission_variable_name]
                    + "]"
                )

            # Read the two CSV and concatenate them so that all data can be displayed against all
            # data
            power_train_data = pd.read_csv(power_train_data_file_path, index_col=0)
            # Remove the taxi power train data because they are not stored in the mission data
            # either
            power_train_data = power_train_data.drop([0]).iloc[:-1]
            # We readjust the index
            power_train_data = power_train_data.set_index(np.arange(len(power_train_data.index)))
            power_train_data = power_train_data.drop(columns_to_drop, axis=1)

            mission_data = pd.read_csv(mission_data_file_path, index_col=0)
            all_data = pd.concat([power_train_data, mission_data], axis=1)

        else:
            raise TypeError("Unknown type for mission and power train data, please use .csv")

        # The figure displayed
        self._fig = None

        # The x selector
        self._x_widget = None

        # The y selector
        self._y_widget = None

        self.plot_height = plot_height
        self.plot_width = plot_width

        self.data = all_data

        self._initialize_widgets()

    def _initialize_widgets(self):
        """
        Initializes the widgets for selecting x and y
        """

        key = list(self.data)
        keys = self.data[key].keys()

        output = widgets.Output()

        def show_plots(change=None):

            with output:

                clear_output(wait=True)

                x_name = self._x_widget.value
                y_name = self._y_widget.value

                fig = go.Figure()

                # If both mission and pt data are present, we change the color according to the
                # phase to make it more readable
                if "name" in self.data.columns:
                    for name in list(set(self.data["name"].to_list())):
                        # pylint: disable=invalid-name # that's a common naming
                        x = self.data.loc[self.data["name"] == name, x_name]
                        # pylint: disable=invalid-name # that's a common naming
                        y = self.data.loc[self.data["name"] == name, y_name]

                        scatter = go.Scatter(
                            x=x,
                            y=y,
                            mode="markers",
                            marker={"color": COLOR_DICTIONARY[name]},
                            name=name,
                        )
                        fig.add_trace(scatter)
                else:

                    # pylint: disable=invalid-name # that's a common naming
                    x = self.data[x_name]
                    # pylint: disable=invalid-name # that's a common naming
                    y = self.data[y_name]

                    scatter = go.Scatter(x=x, y=y, mode="markers")
                    fig.add_trace(scatter)

                fig.update_layout(
                    title_text="Power train performances on the mission",
                    title_x=0.5,
                    xaxis_title=x_name,
                    yaxis_title=y_name,
                    showlegend=True,
                )
                if self.plot_height:
                    fig.update_layout(height=self.plot_height)

                if self.plot_width:
                    fig.update_layout(width=self.plot_width)

                fig.show()

        # Check if time is in column name to put it as the x axis by default
        if "time" in keys:
            index_x = self.data.columns.get_loc("time")
        else:
            index_x = 2

        # By default ground distance
        self._x_widget = widgets.Dropdown(value=keys[index_x], options=keys)
        self._x_widget.observe(show_plots, "value")
        # By default altitude
        self._y_widget = widgets.Dropdown(value=keys[1], options=keys)
        self._y_widget.observe(show_plots, "value")

        toolbar = widgets.HBox(
            [widgets.Label(value="x:"), self._x_widget, widgets.Label(value="y:"), self._y_widget]
        )

        display(toolbar, output)
