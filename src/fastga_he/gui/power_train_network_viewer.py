# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os
import os.path as pth

from pathlib import Path
import re
import networkx as nx
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    BoxSelectTool,
    LabelSet,
)

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from . import icons
from .layout_generation import HierarchicalLayout

BACKGROUND_COLOR_CODE = "#bebebe"
ELECTRICITY_CURRENT_COLOR_CODE = "#007BFF"
FUEL_FLOW_COLOR_CODE = "#FF5722"
MECHANICAL_POWER_COLOR_CODE = "#2E7D32"

# Image URLs for graph nodes

icons_dict = {
    "battery": [pth.join(icons.__path__[0], "battery.png"), ELECTRICITY_CURRENT_COLOR_CODE],
    "bus_bar": [pth.join(icons.__path__[0], "bus_bar.png"), ELECTRICITY_CURRENT_COLOR_CODE],
    "cable": [pth.join(icons.__path__[0], "cable.png"), ELECTRICITY_CURRENT_COLOR_CODE],
    "e_motor": [
        pth.join(icons.__path__[0], "e_motor.png"),
        [ELECTRICITY_CURRENT_COLOR_CODE, MECHANICAL_POWER_COLOR_CODE],
    ],
    "generator": [
        pth.join(icons.__path__[0], "generator.png"),
        [MECHANICAL_POWER_COLOR_CODE, ELECTRICITY_CURRENT_COLOR_CODE],
    ],
    "ice": [
        pth.join(icons.__path__[0], "ice.png"),
        [ELECTRICITY_CURRENT_COLOR_CODE, MECHANICAL_POWER_COLOR_CODE],
    ],
    "switch": [pth.join(icons.__path__[0], "switch.png"), ELECTRICITY_CURRENT_COLOR_CODE],
    "propeller": [pth.join(icons.__path__[0], "propeller.png"), MECHANICAL_POWER_COLOR_CODE],
    "splitter": [pth.join(icons.__path__[0], "splitter.png"), ELECTRICITY_CURRENT_COLOR_CODE],
    "rectifier": [pth.join(icons.__path__[0], "AC_DC.png"), ELECTRICITY_CURRENT_COLOR_CODE],
    "dc_converter": [pth.join(icons.__path__[0], "DC_DC.png"), ELECTRICITY_CURRENT_COLOR_CODE],
    "inverter": [pth.join(icons.__path__[0], "DC_AC.png"), ELECTRICITY_CURRENT_COLOR_CODE],
    "fuel_tank": [pth.join(icons.__path__[0], "fuel_tank.png"), FUEL_FLOW_COLOR_CODE],
    "fuel_system": [pth.join(icons.__path__[0], "fuel_system.png"), FUEL_FLOW_COLOR_CODE],
    "turbine": [
        pth.join(icons.__path__[0], "turbine.png"),
        [FUEL_FLOW_COLOR_CODE, MECHANICAL_POWER_COLOR_CODE],
    ],
    "gearbox": [pth.join(icons.__path__[0], "gears.png"), MECHANICAL_POWER_COLOR_CODE],
    "fuel_cell": [
        pth.join(icons.__path__[0], "fuel_cell.png"),
        [FUEL_FLOW_COLOR_CODE, ELECTRICITY_CURRENT_COLOR_CODE],
    ],
}

color_icon_dict = {
    "mechanical": pth.join(icons.__path__[0], "mechanical.png"),
    "fuel": pth.join(icons.__path__[0], "fuel.png"),
    "electricity": pth.join(icons.__path__[0], "electricity.png"),
}


def _get_edge_color(source_icon, target_icon):
    """
    Determine edge color based on source and target node types.

    Args:
        source_icon: Icon name of the source node
        target_icon: Icon name of the target node

    Returns:
        Color code for the edge
    """
    source_colors = icons_dict.get(source_icon, [None, "gray"])[1]
    target_colors = icons_dict.get(target_icon, [None, "gray"])[1]

    # Normalize to lists for easier comparison
    source_colors = source_colors if isinstance(source_colors, list) else [source_colors]
    target_colors = target_colors if isinstance(target_colors, list) else [target_colors]

    # Find common color between source and target
    common_colors = set(source_colors) & set(target_colors)

    if common_colors:
        # Use the first common color
        return list(common_colors)[0]

    # If no common color, use source output color (last in list for multi-output components)
    if source_colors:
        return source_colors[-1]

    return "gray"


def _compute_hierarchical_layout(graph, orientation="TB", node_layer_dict=None):
    """
    Compute hierarchical layout using the Sugiyama algorithm.

    Args:
        graph: NetworkX DiGraph object
        orientation: Layout orientation ('TB', 'BT', 'LR', 'RL')
        node_layer_dict: Optional dictionary to override layer assignment

    Returns:
        Dictionary of node positions
    """
    sugiyama = HierarchicalLayout(graph, orientation, node_layer_dict)
    return sugiyama.compute()


def power_train_network_viewer(
    power_train_file_path: str,
    network_file_path: str,
    orientation: str = "TB",
    legend_position: str = "TR",
    static_html: bool = True,
):
    """
    Create an interactive network visualization of a power train using Bokeh with NetworkX layout.

    Args:
        power_train_file_path: Path to the power train configuration file
        network_file_path: Path where the HTML output will be saved
        layout_prog: Layout algorithm ('hierarchical', 'spring', 'circular', 'kamada_kawai')
        'hierarchical' - Sugiyama layered hierarchical layout
        'spring' - Spring model layout
        'circular' - Circular layout
        'kamada_kawai' - Kamada-Kawai layout
        legend_position: String defines the legend box position
        (T: top, M: middle (vertical), Button, L: left, R: right, C: center (horizontal))
        * * T * *
        * * * * *
        * * M * *
        L * C * R
        * * B * *
        orientation: network plot orientation ('TB', 'BT', 'LR', 'RL')
        (T: top, B: button, L: left, R: right)
        static_html: True if using static html
    """

    plot, edge_source, node_source, node_image_sequences, propeller_rotation_sequences = (
        _create_network_plot(
            power_train_file_path=power_train_file_path,
            orientation=orientation,
            legend_position=legend_position,
            static_html=static_html,
        )
    )

    if static_html:
        _save_static_html(plot, network_file_path)


def _create_network_plot(
    power_train_file_path: str,
    orientation: str = "TB",
    legend_position: str = "TR",
    static_html: bool = True,
):
    """
    Create an interactive network visualization of a power train using Bokeh with NetworkX layout.

    Args:
        power_train_file_path: Path to the power train configuration file
        legend_position: String defines the legend box position
        orientation: network plot orientation ('TB', 'BT', 'LR', 'RL')
        static_html: True if using static html
    """

    # Create NetworkX DiGraph object
    graph = nx.DiGraph()

    configurator = FASTGAHEPowerTrainConfigurator()
    configurator.load(power_train_file_path)

    (
        names,
        connections,
        components_type,
        components_om_type,
        icons_name,
        icons_size,
    ) = configurator.get_network_elements_list()

    distance_from_energy_storage = configurator.get_distance_from_energy_storage()

    # Build node attributes dictionaries
    node_sizes = {}
    node_types = {}
    node_om_types = {}
    node_icons = {}

    # For animation purposes
    node_image_sequences = {}
    propeller_rotation_sequences = {}

    for component_name, component_type, om_type, icon_name, icon_size in zip(
        names, components_type, components_om_type, icons_name, icons_size
    ):
        graph.add_node(component_name)
        node_sizes[component_name] = icon_size
        node_types[component_name] = component_type
        node_om_types[component_name] = om_type
        node_icons[component_name] = icon_name

    # Add edges
    for connection in connections:
        # Filter out bus connection output numbers
        source = connection[0][0] if isinstance(connection[0], list) else connection[0]
        target = connection[1][0] if isinstance(connection[1], list) else connection[1]
        graph.add_edge(source, target)

    # Build node_layer_dict from distance_from_energy_storage
    node_layer_dict = {}
    max_distance = max(distance_from_energy_storage.values())
    for node_name, distance in distance_from_energy_storage.items():
        node_layer_dict[node_name] = max_distance - distance

    # Compute layout based on specified algorithm with hierarchy from distance_from_energy_storage
    position_dict = _compute_hierarchical_layout(graph, orientation, node_layer_dict)

    # Normalize positions for Bokeh
    if position_dict:
        x_coordinates = [coords[0] for coords in position_dict.values()]
        y_coordinates = [coords[1] for coords in position_dict.values()]
        x_min, x_max = min(x_coordinates), max(x_coordinates)
        y_min, y_max = min(y_coordinates), max(y_coordinates)

        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        # Scale to a reasonable display size with different orientation
        if orientation == "TB" or orientation == "BT":
            x_factor = 0.5
            y_factor = 1.0
            scale = 550
            plot_width = 1200
            plot_height = 900
            x_range_max = 600
            y_range_max = 600
            icon_factor = 1
            icon_width_factor = 0.8
            x_orientation_offset = 125
            y_orientation_offset = 0.0

        elif orientation == "LR" or orientation == "RL":
            x_factor = 1.0
            y_factor = 0.5
            scale = 600
            plot_width = 1500
            plot_height = 900
            x_range_max = 600
            y_range_max = 600
            icon_factor = 1.75
            icon_width_factor = 0.6
            x_orientation_offset = -25
            y_orientation_offset = 150

        position_dict = {
            node: (
                ((coordinate[0] - x_min) / x_range * scale * x_factor + x_orientation_offset),
                ((coordinate[1] - y_min) / y_range * scale * y_factor + y_orientation_offset),
            )
            for node, coordinate in position_dict.items()
        }

    # Create Bokeh plot
    plot = figure(
        width=plot_width,
        height=plot_height,
        x_range=(-50, x_range_max),
        y_range=(-50, y_range_max),
        toolbar_location="above",
        background_fill_color=BACKGROUND_COLOR_CODE,
        title=_get_file_name(power_train_file_path),
    )

    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.xaxis.visible = False
    plot.yaxis.visible = False

    # Prepare node data
    node_indices = list(graph.nodes())
    node_x = []
    node_y = []
    node_sizes_list = []
    node_types_list = []
    node_om_types_list = []

    for node in node_indices:
        node_x.append(position_dict[node][0])
        node_y.append(position_dict[node][1])
        node_sizes_list.append(node_sizes[node] * icon_factor)
        node_types_list.append(node_types[node])
        node_om_types_list.append(node_om_types[node])

    # Convert file paths to file:// URLs for local images
    if static_html:
        node_image_urls = [
            "file://" + str(Path(icons_dict[node_icons[node]][0]).resolve())
            for node in node_indices
        ]

    color_icon_urls = [
        "file://" + str(Path(color_icon_dict[color_icon]).resolve())
        for color_icon in ["fuel", "mechanical", "electricity"]
    ]

    # Create edge data with colors
    edge_start_x = []
    edge_start_y = []
    edge_end_x = []
    edge_end_y = []
    edge_colors = []

    for edge in graph.edges():
        start, end = edge
        edge_start_x.append(position_dict[start][0])
        edge_start_y.append(position_dict[start][1])
        edge_end_x.append(position_dict[end][0])
        edge_end_y.append(position_dict[end][1])

        # Determine edge color based on connected nodes
        source_icon = node_icons[start]
        target_icon = node_icons[end]
        edge_color = _get_edge_color(source_icon, target_icon)
        edge_colors.append(edge_color)

    # Draw edges
    if static_html:
        edge_source = ColumnDataSource(
            data=dict(
                xs=[[sx, ex] for sx, ex in zip(edge_start_x, edge_end_x)],
                ys=[[sy, ey] for sy, ey in zip(edge_start_y, edge_end_y)],
                line_color=edge_colors,
                line_alpha=[0.7] * len(edge_start_x),
            )
        )

    plot.multi_line(
        xs="xs",
        ys="ys",
        source=edge_source,
        line_color="line_color",
        line_width=3,
        line_alpha="line_alpha",
    )

    # Draw nodes as images
    node_source = ColumnDataSource(
        data=dict(
            x=node_x,
            y=node_y,
            url=node_image_urls,
            w=[s * icon_width_factor for s in node_sizes_list],
            h=[s for s in node_sizes_list],
            name=node_indices,
            type=node_types_list,
        )
    )

    # Add circle to cover edge line
    plot.scatter(
        x="x",
        y="y",
        size=45,
        source=node_source,
        color=BACKGROUND_COLOR_CODE,
        line_alpha=0,
    )

    plot.image_url(
        url="url",
        x="x",
        y="y",
        w="w",
        h="h",
        anchor="center",
        source=node_source,
    )

    # Add labels below nodes
    label_source = ColumnDataSource(
        data=dict(
            x=node_x,
            y=[y - node_sizes_list[i] * 0.7 for i, y in enumerate(node_y)],
            names=node_indices,
        )
    )

    labels = LabelSet(
        x="x",
        y="y",
        text="names",
        source=label_source,
        text_align="center",
        text_baseline="top",
        text_color="white",
        text_font_size="8pt",
    )
    plot.add_layout(labels)

    _add_color_legend_separate(plot, legend_position, color_icon_urls)

    # Add interactive tools
    cleaned_node_types = []
    cleaned_node_om_types = []
    for node_type, node_om_type in zip(node_types_list, node_om_types_list):
        if isinstance(node_type, str):
            cleaned_node_types.append(_string_clean_up(node_type.capitalize()))
        else:
            cleaned_node_types.append(_string_clean_up(node_type))

        cleaned_node_om_types.append(_string_clean_up(node_om_type))

    # Define list info
    hover_source = ColumnDataSource(
        data=dict(
            x=node_x,
            y=node_y,
            w=[s * icon_width_factor for s in node_sizes_list],
            h=[s for s in node_sizes_list],
            name=node_indices,
            type_class=cleaned_node_types,
            component_type=cleaned_node_om_types,
        )
    )

    # Add invisible circles on top for hover interactivity
    plot.scatter(
        x="x",
        y="y",
        size=60,
        source=hover_source,
        fill_alpha=0,
        line_alpha=0,
        hover_fill_alpha=0.1,
        hover_line_alpha=0.3,
    )

    hover = HoverTool(
        tooltips=[
            ("Name", "@name"),
            ("Type class", "@type_class"),
            ("Component type", "@component_type"),
        ]
    )
    plot.add_tools(hover, BoxSelectTool())

    return plot, edge_source, node_source, node_image_sequences, propeller_rotation_sequences


def _add_color_legend_separate(plot, legend_position, color_icon_urls):
    """
    Add a color legend as a separate visual entity within the same plot.

    Args:
        plot: Bokeh figure object
        legend_position: String defines the legend box position
        color_icon_urls: List of URLs for color icons
    """

    if "T" in legend_position:
        legend_y_start = 600
    elif "B" in legend_position:
        legend_y_start = 50
    elif "M" in legend_position:
        legend_y_start = 325

    if "R" in legend_position:
        legend_x_start = 500
    elif "L" in legend_position:
        legend_x_start = -50
    elif "C" in legend_position:
        legend_x_start = 225

    # Draw legend background box using quad
    legend_box_source = ColumnDataSource(
        data=dict(
            left=[legend_x_start],
            right=[legend_x_start + 100],
            top=[legend_y_start],
            bottom=[legend_y_start - 100],
        )
    )
    plot.quad(
        left="left",
        right="right",
        top="top",
        bottom="bottom",
        source=legend_box_source,
        fill_color=BACKGROUND_COLOR_CODE,
        fill_alpha=0.9,
        line_width=0,
    )

    # Legend items
    legend_items = [
        (0, "Fuel Flow"),
        (1, "Mechanical Power"),
        (2, "Electrical Current"),
    ]

    legend_item_height = 22
    icon_size = 12
    legend_item_start_y = legend_y_start - 25

    for i, (icon_idx, description) in enumerate(legend_items):
        y_position = legend_item_start_y - (i * legend_item_height)

        # Create data source for icon
        icon_source = ColumnDataSource(
            data=dict(
                x=[legend_x_start + 10],
                y=[y_position],
                url=[color_icon_urls[icon_idx]],
            )
        )

        # Draw color icon image
        plot.image_url(
            url="url",
            x="x",
            y="y",
            w=icon_size,
            h=icon_size,
            anchor="center",
            source=icon_source,
        )

        # Add text label next to the color icon image
        label_source = ColumnDataSource(
            data=dict(
                x=[legend_x_start + 25],
                y=[y_position],
                text=[description],
            )
        )

        labels = LabelSet(
            x="x",
            y="y",
            text="text",
            source=label_source,
            text_align="left",
            text_baseline="middle",
            text_color="white",
            text_font_size="10pt",
        )
        plot.add_layout(labels)


def _string_clean_up(old_string):
    """
    Clean up list content for better readability.
    """
    # In case for list type definition
    if isinstance(old_string, list):
        old_string = old_string[0].capitalize() + ", " + old_string[1].capitalize()

    # Replace underscore with space
    new_string = re.sub(r"[_:/]+", " ", old_string)

    # Add a space after 'DC' if followed immediately by a letter or number
    new_string = re.sub(r"\bDC(?=[A-Za-z0-9])", "DC ", new_string)
    new_string = re.sub(r"\bDC DC(?=[A-Za-z0-9])", "DC-DC ", new_string)

    # Add a space after 'H2' and 'PEMFC'
    new_string = re.sub(r"\bH2(?=[A-Za-z0-9])", "H2 ", new_string)
    new_string = re.sub(r"\bPEMFC(?=[A-Za-z0-9])", "PEMFC ", new_string)

    # Add space before a capital letter preceded by a lowercase letter
    new_string = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", new_string)

    # Remove extra spaces
    new_string = re.sub(r"\s+", " ", new_string).strip()

    return new_string


def _get_file_name(file_path):
    """
    Using the file name as the plot title
    """
    match_html = re.search(r"[^/\\]+\.yml$", str(file_path))

    if match_html:
        filename = match_html.group()
        filename = re.sub(r"\.yml$", "", filename)
        filename = re.sub(r"[_:/]+", " ", filename).capitalize()

        return f"{filename} powertrain network"


def _save_static_html(plot, file_path):
    """
    Save the network plot as static html.
    """
    # Create directory if it doesn't exist
    directory_to_save_graph = os.path.dirname(file_path)

    if directory_to_save_graph and not os.path.exists(directory_to_save_graph):
        os.makedirs(directory_to_save_graph)

    # Save the plot
    output_file(file_path)
    save(plot)
