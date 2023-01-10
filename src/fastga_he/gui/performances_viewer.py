# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import clear_output, display

COLOR_DICTIONARY = {
    "sizing:main_route:climb": "crimson",
    "sizing:main_route:cruise": "darkblue",
    "sizing:main_route:descent": "lightskyblue",
    "sizing:main_route:reserve": "darkgreen",
}


class PerformancesViewer:
    """
    A class for displaying the performances of all the elements of the power train along with the
    performances at aircraft level. Vastly inspired by the MissionViewer from fast-oad-core.
    """

    def __init__(self, power_train_data_file_path: str, mission_data_file_path: str):

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

        elif not power_train_data_file_path.endswith(".csv") and mission_data_file_path.endswith(
            ".csv"
        ):

            mission_data = pd.read_csv(mission_data_file_path, index_col=0)

            all_data = mission_data

        elif power_train_data_file_path.endswith(".csv") and mission_data_file_path.endswith(
            ".csv"
        ):

            # Read the two CSV and concatenate them so that all data can be displayed against all
            # data
            power_train_data = pd.read_csv(power_train_data_file_path, index_col=0)
            # Remove the taxi power train data because they are not stored in the mission data
            # either
            power_train_data = power_train_data.drop([0]).iloc[:-1]
            # We readjust the index
            power_train_data = power_train_data.set_index(np.arange(len(power_train_data.index)))

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

        self.data = all_data

        self._initialize_widgets()

    def _initialize_widgets(self):
        """
        Initializes the widgets for selecting x and y
        """

        key = list(self.data)
        keys = self.data[key].keys()

        # By default ground distance
        self._x_widget = widgets.Dropdown(value=keys[2], options=keys)
        self._x_widget.observe(self.display, "value")
        # By default altitude
        self._y_widget = widgets.Dropdown(value=keys[1], options=keys)
        self._y_widget.observe(self.display, "value")

    # pylint: disable=unused-argument  # args has to be there for observe() to work
    def display(self, change=None) -> display:
        """
        Display the user interface
        :return the display object
        """
        clear_output(wait=True)
        self._update_plots()
        toolbar = widgets.HBox(
            [widgets.Label(value="x:"), self._x_widget, widgets.Label(value="y:"), self._y_widget]
        )
        # pylint: disable=invalid-name # that's a common naming
        ui = widgets.VBox([toolbar, self._fig])
        return display(ui)

    def _update_plots(self):
        """
        Update the plots
        """
        self._fig = None
        self._build_plots()

    def _build_plots(self):
        """
        Add a plot of the mission
        """

        x_name = self._x_widget.value
        y_name = self._y_widget.value

        if self._fig is None:
            self._fig = go.Figure()

        # If both mission and pt data are present, we change the color according to the phase to
        # make it more readable
        if "name" in self.data.columns:
            for name in list(set(self.data["name"].to_list())):

                # pylint: disable=invalid-name # that's a common naming
                x = self.data.loc[self.data["name"] == name, x_name]
                # pylint: disable=invalid-name # that's a common naming
                y = self.data.loc[self.data["name"] == name, y_name]

                scatter = go.Scatter(
                    x=x, y=y, mode="markers", marker={"color": COLOR_DICTIONARY[name]}, name=name
                )
                self._fig.add_trace(scatter)
        else:

            # pylint: disable=invalid-name # that's a common naming
            x = self.data[x_name]
            # pylint: disable=invalid-name # that's a common naming
            y = self.data[y_name]

            scatter = go.Scatter(x=x, y=y, mode="markers")
            self._fig.add_trace(scatter)

        layout = go.Layout(showlegend=True)

        self._fig = go.FigureWidget(self._fig, layout=layout)

        self._fig.update_layout(
            title_text="Power train performances on the mission",
            title_x=0.5,
            xaxis_title=x_name,
            yaxis_title=y_name,
        )