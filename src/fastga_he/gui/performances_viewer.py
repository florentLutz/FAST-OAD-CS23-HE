# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import clear_output, display

from plotly.subplots import make_subplots

from fastga_he.powertrain_builder.powertrain import PROMOTION_FROM_MISSION

COLOR_DICTIONARY = {
    "sizing:main_route:climb": px.colors.qualitative.Prism[1],
    "sizing:main_route:cruise": px.colors.qualitative.Prism[2],
    "sizing:main_route:descent": px.colors.qualitative.Prism[3],
    "sizing:main_route:reserve": px.colors.qualitative.Prism[4],
}
COLOR_DICTIONARY_AXIS_2 = {
    "sizing:main_route:climb": px.colors.qualitative.Prism[5],
    "sizing:main_route:cruise": px.colors.qualitative.Prism[6],
    "sizing:main_route:descent": px.colors.qualitative.Prism[7],
    "sizing:main_route:reserve": px.colors.qualitative.Prism[8],
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

        # The y2 selector
        self._y2_widget = None

        # The button to ensure same axis
        self._axis_ensurer = None

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
                y2_name = self._y2_widget.value

                if y2_name == "None":
                    fig = go.Figure()
                else:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])

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
                            legendgroup="Primary axis",
                            legendgrouptitle_text="primary_axis",
                        )

                        if y2_name != "None":
                            y_2 = self.data.loc[self.data["name"] == name, y2_name]
                            scatter_2 = go.Scatter(
                                x=x,
                                y=y_2,
                                mode="markers",
                                marker={"color": COLOR_DICTIONARY_AXIS_2[name]},
                                name=name,
                                legendgroup="Secondary axis",
                                legendgrouptitle_text="secondary_axis",
                            )

                            fig.add_trace(scatter, secondary_y=False)
                            fig.add_trace(scatter_2, secondary_y=True)

                        else:
                            fig.add_trace(scatter)
                else:

                    # pylint: disable=invalid-name # that's a common naming
                    x = self.data[x_name]
                    # pylint: disable=invalid-name # that's a common naming
                    y = self.data[y_name]

                    scatter = go.Scatter(
                        x=x,
                        y=y,
                        mode="markers",
                        legendgroup="Primary axis",
                        legendgrouptitle_text="primary_axis",
                    )

                    if y2_name != "None":
                        y_2 = self.data[y2_name]
                        scatter_2 = go.Scatter(
                            x=x,
                            y=y_2,
                            mode="markers",
                            legendgroup="Secondary axis",
                            legendgrouptitle_text="secondary_axis",
                        )

                        fig.add_trace(scatter, secondary_y=False)
                        fig.add_trace(scatter_2, secondary_y=True)
                    else:
                        fig.add_trace(scatter)

                if y2_name != "None":
                    fig.update_layout(
                        title_text="Power train performances on the mission",
                        title_x=0.5,
                        showlegend=True,
                    )
                    fig.update_xaxes(title_text=x_name)
                    fig.update_yaxes(title_text=y_name, secondary_y=False)
                    fig.update_yaxes(title_text=y2_name, secondary_y=True)

                    if self._axis_ensurer.value:
                        y_min = min(y.min(), y_2.min())
                        y_max = max(y.max(), y_2.max())

                        fig.update_yaxes(range=[0.95 * y_min, 1.05 * y_max], secondary_y=False)
                        fig.update_yaxes(range=[0.95 * y_min, 1.05 * y_max], secondary_y=True)

                else:
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

                fig = go.FigureWidget(fig)
                display(fig)

        # Check if time is in column name to put it as the x axis by default
        if "time" in keys:
            index_x = self.data.columns.get_loc("time")
        else:
            index_x = 2

        self._x_widget = widgets.Dropdown(value=keys[index_x], options=keys)
        self._x_widget.observe(show_plots, "value")

        self._y_widget = widgets.Dropdown(value=keys[1], options=keys)
        self._y_widget.observe(show_plots, "value")

        self._y2_widget = widgets.Dropdown(value="None", options=list(keys) + ["None"])
        self._y2_widget.observe(show_plots, "value")

        self._axis_ensurer = widgets.ToggleButton(
            value=False,
            description="Ensure same axis",
            disabled=False,
            button_style="success",
            tooltip="Check me to ensure that the primary and secondary axis will have the same y "
            "range, useful for comparing efficiencies for instance",
            icon="check",
        )
        self._axis_ensurer.observe(show_plots, "value")

        show_plots()

        toolbar = widgets.HBox(
            [
                widgets.Label(value="x:"),
                self._x_widget,
                widgets.Label(value="y:"),
                self._y_widget,
                widgets.Label(value="y2:"),
                self._y2_widget,
                self._axis_ensurer,
            ]
        )

        display(toolbar, output)
