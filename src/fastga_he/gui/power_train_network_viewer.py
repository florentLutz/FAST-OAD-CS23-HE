# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import os
import os.path as pth

from pathlib import Path
import re
import pygraphviz as pgv
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    BoxSelectTool,
    LabelSet,
)

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from . import icons

# Image URLs for graph nodes

icons_dict = {
    "battery": pth.join(icons.__path__[0], "battery.png"),
    "bus_bar": pth.join(icons.__path__[0], "bus_bar.png"),
    "cable": pth.join(icons.__path__[0], "cable.png"),
    "e_motor": pth.join(icons.__path__[0], "e_motor.png"),
    "generator": pth.join(icons.__path__[0], "generator.png"),
    "ice": pth.join(icons.__path__[0], "ice.png"),
    "switch": pth.join(icons.__path__[0], "switch.png"),
    "propeller": pth.join(icons.__path__[0], "propeller.png"),
    "splitter": pth.join(icons.__path__[0], "splitter.png"),
    "rectifier": pth.join(icons.__path__[0], "AC_DC.png"),
    "dc_converter": pth.join(icons.__path__[0], "DC_DC.png"),
    "inverter": pth.join(icons.__path__[0], "DC_AC.png"),
    "fuel_tank": pth.join(icons.__path__[0], "fuel_tank.png"),
    "fuel_system": pth.join(icons.__path__[0], "fuel_system.png"),
    "turbine": pth.join(icons.__path__[0], "turbine.png"),
    "gearbox": pth.join(icons.__path__[0], "gears.png"),
}
BACKGROUND_COLOR_CODE = "#bebebe"

def power_train_network_viewer(
        power_train_file_path: str,
        network_file_path: str,
        layout_prog: str = "dot",
        orientation: str = "TB",
):
    """
    Create an interactive network visualization of a power train using Bokeh with PyGraphviz layout.

    Args:
        power_train_file_path: Path to the power train configuration file
        network_file_path: Path where the HTML output will be saved
        layout_prog: Graphviz layout program ('dot', 'neato', 'fdp', 'sfdp', 'circo')
        "dot" - Hierarchical layout
        "neato" - Spring model layout
        "fdp" - Force-directed placement
        "sfdp" - Scalable force-directed placement
        "circo" - Circular layout
    """
    # Create AGraph (PyGraphviz) object
    graph = pgv.AGraph(directed=True, rankdir=orientation)

    configurator = FASTGAHEPowerTrainConfigurator()
    configurator.load(power_train_file_path)

    (
        names,
        connections,
        components_type,
        components_type_om,
        icons_name,
        icons_size,
    ) = configurator.get_network_elements_list()

    # Build node attributes dictionaries
    node_sizes = {}
    node_types = {}
    node_om_types = {}
    node_icons = {}

    for component_name, component_type, om_type, icon_name, icon_size in zip(
            names, components_type, components_type_om, icons_name, icons_size
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

    # Apply Graphviz layout algorithm
    graph.layout(prog=layout_prog)

    # Extract positions from Graphviz layout
    pos = {}
    for node in graph.nodes():
        x, y = map(float, node.attr["pos"].split(","))
        pos[node] = (x, y)

    # Normalize positions for Bokeh
    if pos:
        x_coords = [p[0] for p in pos.values()]
        y_coords = [p[1] for p in pos.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

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
            icon_width_factor = 1.0
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

        pos = {
            node: (
                ((p[0] - x_min) / x_range * scale * x_factor + x_orientation_offset),
                ((p[1] - y_min) / y_range * scale * y_factor + y_orientation_offset)
            )
            for node, p in pos.items()
        }

    # Create Bokeh plot
    plot = figure(
        width=plot_width,
        height=plot_height,
        x_range=(-50, x_range_max),
        y_range=(-50, y_range_max),
        toolbar_location="above",
        background_fill_color=BACKGROUND_COLOR_CODE,
        title=get_file_name(power_train_file_path),
    )
    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.xaxis.visible = False
    plot.yaxis.visible = False

    # Prepare node data
    node_indices = list(graph.nodes())
    node_x = [pos[node][0] for node in node_indices]
    node_y = [pos[node][1] for node in node_indices]
    node_image_urls = [icons_dict[node_icons[node]] for node in node_indices]
    node_sizes_list = [node_sizes[node]*icon_factor for node in node_indices]
    node_types_list = [node_types[node] for node in node_indices]
    node_om_types_list = [node_om_types[node] for node in node_indices]

    # Convert file paths to file:// URLs for local images
    node_image_urls = ["file://" + str(Path(url).resolve()) for url in node_image_urls]

    # Create edge data
    edge_start_x = []
    edge_start_y = []
    edge_end_x = []
    edge_end_y = []

    for edge in graph.edges():
        start, end = edge
        edge_start_x.append(pos[start][0])
        edge_start_y.append(pos[start][1])
        edge_end_x.append(pos[end][0])
        edge_end_y.append(pos[end][1])

    # Draw edges
    edge_source = ColumnDataSource(
        data=dict(
            xs=[[sx, ex] for sx, ex in zip(edge_start_x, edge_end_x)],
            ys=[[sy, ey] for sy, ey in zip(edge_start_y, edge_end_y)],
            line_color=["gray"]*len(edge_start_x)
        )
    )
    plot.multi_line(
        xs="xs",
        ys="ys",
        source=edge_source,
        line_color="line_color",
        line_width=2,
        line_alpha=0.5,
    )

    # Draw nodes as images
    node_source = ColumnDataSource(
        data=dict(
            x=node_x,
            y=node_y,
            url=node_image_urls,
            w=[s*icon_width_factor for s in node_sizes_list],
            h=[s for s in node_sizes_list],
            name=node_indices,
            type=node_types_list,
        )
    )

    # Add circle to cover edge line
    plot.circle(
        x="x",
        y="y",
        size=45,  # Adjust size to match your icon size
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

    # Add interactive tools
    # clean up type class and component type for hove info list
    cleaned_node_types = []
    cleaned_node_om_types = []
    for node_type, node_om_type in zip(node_types_list, node_om_types_list):
        if type(node_type) == str:
            cleaned_node_types.append(string_clean_up(node_type.capitalize()))
        else:
            cleaned_node_types.append(string_clean_up(node_type))

        cleaned_node_om_types.append(string_clean_up(node_om_type))

    # Define list info
    hover_source = ColumnDataSource(
        data=dict(
            x=node_x,
            y=node_y,
            w=[s * icon_width_factor for s in node_sizes_list],
            h=[s for s in node_sizes_list],
            name=node_indices,
            type=cleaned_node_types,
            component_type = cleaned_node_om_types,
        )
    )

    # Add invisible circles on top for hover interactivity
    plot.circle(
        x="x",
        y="y",
        size=60,
        source=hover_source,
        fill_alpha=0,
        line_alpha=0,
        hover_fill_alpha=0.1,  # show faint highlight on component icon
        hover_line_alpha=0.3,
    )

    hover = HoverTool(
        tooltips=[
            ("Name", "@name"),
            ("Type class", "@type"),
            ("Component type", "@component_type"),
        ]
    )
    plot.add_tools(hover, BoxSelectTool())

    # Create directory if it doesn't exist
    directory_to_save_graph = os.path.dirname(network_file_path)
    if directory_to_save_graph and not os.path.exists(directory_to_save_graph):
        os.makedirs(directory_to_save_graph)

    # Save the plot
    output_file(network_file_path)
    save(plot)

    return plot


def string_clean_up(old_string):
    # In case for list type definition
    if type(old_string) == list:
        old_string = old_string[0].capitalize() + ", " + old_string[1].capitalize()

    # Replace underscore with space
    new_string = re.sub(r'[_:/]+', ' ', old_string)

    # Add a space after 'DC' if followed immediately by a letter or number
    new_string = re.sub(r'\bDC(?=[A-Za-z0-9])', 'DC ', new_string)
    new_string = re.sub(r'\bDC DC(?=[A-Za-z0-9])','DC-DC ',new_string)

    # Add space before a capital letter preceded by a lowercase letter
    new_string = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', new_string)

    # Remove extra spaces
    new_string = re.sub(r'\s+', ' ', new_string).strip()

    return new_string

def get_file_name(file_path):
    match_html = re.search(r'[^/\\]+\.yml$', str(file_path))

    if match_html:
        filename = match_html.group()
        filename = re.sub(r'\.yml$', '', filename)
        filename = re.sub(r'[_:/]+', ' ', filename).capitalize()

        return  f"{filename} powertrain network"
