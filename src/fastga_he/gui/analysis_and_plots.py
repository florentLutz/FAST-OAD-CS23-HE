# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

from random import SystemRandom

import numpy as np

import plotly
import plotly.graph_objects as go

from fastoad.io import VariableIO

COLS = plotly.colors.DEFAULT_PLOTLY_COLORS


def aircraft_geometry_plot(
    aircraft_file_path: str, name="", fig=None, plot_nacelle: bool = True, file_formatter=None
) -> go.FigureWidget:
    """
    Returns a figure plot of the top view of the wing.
    Different designs can be superposed by providing an existing fig.
    Each design can be provided a name.

    :param aircraft_file_path: path of data file
    :param name: name to give to the trace added to the figure
    :param fig: existing figure to which add the plot
    :param plot_nacelle: boolean to turn on or off the plotting of the nacelles
    :param file_formatter: the formatter that defines the format of data file. If not provided,
    default format will be assumed.
    :return: wing plot figure.
    """
    variables = VariableIO(aircraft_file_path, file_formatter).read()

    # Wing parameters
    wing_kink_leading_edge_x = variables["data:geometry:wing:kink:leading_edge:x:local"].value[0]
    wing_tip_leading_edge_x = variables["data:geometry:wing:tip:leading_edge:x:local"].value[0]
    wing_root_y = variables["data:geometry:wing:root:y"].value[0]
    wing_kink_y = variables["data:geometry:wing:kink:y"].value[0]
    wing_tip_y = variables["data:geometry:wing:tip:y"].value[0]
    wing_root_chord = variables["data:geometry:wing:root:chord"].value[0]
    wing_kink_chord = variables["data:geometry:wing:kink:chord"].value[0]
    wing_tip_chord = variables["data:geometry:wing:tip:chord"].value[0]

    y_wing = np.array(
        [0, wing_root_y, wing_kink_y, wing_tip_y, wing_tip_y, wing_kink_y, wing_root_y, 0, 0]
    )

    x_wing = np.array(
        [
            0,
            0,
            wing_kink_leading_edge_x,
            wing_tip_leading_edge_x,
            wing_tip_leading_edge_x + wing_tip_chord,
            wing_kink_leading_edge_x + wing_kink_chord,
            wing_root_chord,
            wing_root_chord,
            0,
        ]
    )

    # Horizontal Tail parameters
    ht_root_chord = variables["data:geometry:horizontal_tail:root:chord"].value[0]
    ht_tip_chord = variables["data:geometry:horizontal_tail:tip:chord"].value[0]
    ht_span = variables["data:geometry:horizontal_tail:span"].value[0]
    ht_sweep_0 = variables["data:geometry:horizontal_tail:sweep_0"].value[0]

    ht_tip_leading_edge_x = ht_span / 2.0 * np.tan(ht_sweep_0 * np.pi / 180.0)

    y_ht = np.array([0, ht_span / 2.0, ht_span / 2.0, 0.0, 0.0])

    x_ht = np.array(
        [
            -0.25 * ht_root_chord,
            ht_tip_leading_edge_x - 0.25 * ht_tip_chord,
            ht_tip_leading_edge_x + 0.75 * ht_tip_chord,
            0.75 * ht_root_chord,
            -0.25 * ht_root_chord,
        ]
    )

    # Fuselage parameters
    fuselage_max_width = variables["data:geometry:fuselage:maximum_width"].value[0]
    fuselage_length = variables["data:geometry:fuselage:length"].value[0]
    fuselage_front_length = variables["data:geometry:fuselage:front_length"].value[0]
    fuselage_rear_length = variables["data:geometry:fuselage:rear_length"].value[0]

    x_fuselage = np.array(
        [
            0.0,
            0.0,
            fuselage_front_length,
            fuselage_length - fuselage_rear_length,
            fuselage_length,
            fuselage_length,
        ]
    )

    y_fuselage = np.array(
        [
            0.0,
            fuselage_max_width / 4.0,
            fuselage_max_width / 2.0,
            fuselage_max_width / 2.0,
            fuselage_max_width / 4.0,
            0.0,
        ]
    )

    # CGs
    wing_25mac_x = variables["data:geometry:wing:MAC:at25percent:x"].value[0]
    wing_mac_length = variables["data:geometry:wing:MAC:length"].value[0]
    local_wing_mac_le_x = variables["data:geometry:wing:MAC:leading_edge:x:local"].value[0]
    local_ht_25mac_x = variables["data:geometry:horizontal_tail:MAC:at25percent:x:local"].value[0]
    ht_distance_from_wing = variables[
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"
    ].value[0]

    x_wing = x_wing + wing_25mac_x - 0.25 * wing_mac_length - local_wing_mac_le_x
    x_ht = x_ht + wing_25mac_x + ht_distance_from_wing - local_ht_25mac_x

    # pylint: disable=invalid-name
    # that's a common naming
    x = np.concatenate((x_fuselage, x_wing, x_ht))
    # pylint: disable=invalid-name
    # that's a common naming
    y = np.concatenate((y_fuselage, y_wing, y_ht))

    # pylint: disable=invalid-name
    # that's a common naming
    y = np.concatenate((-y, y))
    # pylint: disable=invalid-name
    # that's a common naming
    x = np.concatenate((x, x))

    if fig is None:
        fig = go.Figure()

    scatter = go.Scatter(x=y, y=x, mode="lines+markers", name=name, showlegend=True)

    fig.add_trace(scatter)

    # Nacelle + propeller
    prop_layout = variables["data:geometry:propulsion:engine:layout"].value[0]
    nac_width = variables["data:geometry:propulsion:nacelle:width"].value[0]
    nac_length = variables["data:geometry:propulsion:nacelle:length"].value[0]
    prop_diam = variables["data:geometry:propeller:diameter"].value[0]
    pos_y_nacelle = np.array(variables["data:geometry:propulsion:nacelle:y"].value)
    pos_x_nacelle = np.array(variables["data:geometry:propulsion:nacelle:x"].value)

    if prop_layout == 1.0:
        x_nacelle_plot = np.array([0.0, nac_length, nac_length, 0.0, 0.0, 0.0])
        y_nacelle_plot = np.array(
            [
                -nac_width / 2,
                -nac_width / 2,
                nac_width / 2,
                nac_width / 2,
                prop_diam / 2,
                -prop_diam / 2,
            ]
        )
    elif prop_layout == 3.0:
        prop_depth = variables["data:geometry:propeller:depth"].value[0]
        x_nacelle_plot = np.array([0.0, nac_length, nac_length, 0.0, 0.0, 0.0]) + prop_depth
        y_nacelle_plot = np.array(
            [
                max(-nac_width / 2, -fuselage_max_width / 4.0),
                -nac_width / 2,
                nac_width / 2,
                min(nac_width / 2, fuselage_max_width / 4.0),
                prop_diam / 2,
                -prop_diam / 2,
            ]
        )
    else:
        x_nacelle_plot = np.array([])
        y_nacelle_plot = np.array([])

    if plot_nacelle:
        if prop_layout == 1.0:

            random_generator = SystemRandom()
            trace_colour = COLS[random_generator.randrange(0, len(COLS))]
            show_legend = True

            for y_nacelle_local, x_nacelle_local in zip(pos_y_nacelle, pos_x_nacelle):

                y_nacelle_left = y_nacelle_plot + y_nacelle_local
                y_nacelle_right = -y_nacelle_plot - y_nacelle_local
                x_nacelle = x_nacelle_local - x_nacelle_plot

                if show_legend:
                    scatter_right = go.Scatter(
                        x=y_nacelle_right,
                        y=x_nacelle,
                        name="right nacelle",
                        legendgroup=name + "nacelle",
                        mode="lines+markers",
                        line=dict(color=trace_colour),
                        legendgrouptitle_text=name + " nacelle + propeller",
                    )

                    fig.add_trace(scatter_right)

                    scatter_left = go.Scatter(
                        x=y_nacelle_left,
                        y=x_nacelle,
                        name="left nacelle",
                        legendgroup=name + "nacelle",
                        mode="lines+markers",
                        line=dict(color=trace_colour),
                    )

                    fig.add_trace(scatter_left)

                    show_legend = False
        else:
            scatter = go.Scatter(
                x=y_nacelle_plot,
                y=x_nacelle_plot,
                mode="lines+markers",
                name=name + " nacelle + propeller",
            )
            fig.add_trace(scatter)

    fig.layout = go.Layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    fig = go.FigureWidget(fig)

    fig.update_layout(
        title_text="Aircraft Geometry",
        title_x=0.5,
        xaxis_title="y",
        yaxis_title="x",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig
