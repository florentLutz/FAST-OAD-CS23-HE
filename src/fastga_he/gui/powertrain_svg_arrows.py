import os
import os.path as pth
from pathlib import Path

import pygraphviz as pgv
from bokeh.plotting import figure, output_file, save
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    TapTool,
    BoxSelectTool,
    LabelSet,
    Segment,
)
from bokeh.transform import transform

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


def power_train_network_viewer_hv_svg(
        power_train_file_path: str,
        network_file_path: str,
        layout_prog: str = "dot",
):
    """
    Create an interactive network visualization of a power train using Bokeh with animated Segment glyphs.

    Args:
        power_train_file_path: Path to the power train configuration file
        network_file_path: Path where the HTML output will be saved
        layout_prog: Graphviz layout program ('dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi')
    """
    # Create AGraph (PyGraphviz) object
    G = pgv.AGraph(directed=True, rankdir="TB")

    configurator = FASTGAHEPowerTrainConfigurator()
    configurator.load(power_train_file_path)

    (
        names,
        connections,
        components_type,
        icons_name,
        icons_size,
    ) = configurator.get_network_elements_list()
    distance_from_prop_loads, prop_loads = configurator.get_distance_from_propulsive_load()

    # Build node attributes dictionaries
    node_sizes = {}
    node_types = {}
    node_icons = {}

    for component_name, component_type, icon_name, icon_size in zip(
            names, components_type, icons_name, icons_size
    ):
        G.add_node(component_name)
        node_sizes[component_name] = icon_size
        node_types[component_name] = component_type
        node_icons[component_name] = icon_name

    # Add edges
    for connection in connections:
        source = connection[0][0] if isinstance(connection[0], list) else connection[0]
        target = connection[1][0] if isinstance(connection[1], list) else connection[1]
        G.add_edge(source, target)

    # Apply Graphviz layout algorithm
    G.layout(prog=layout_prog)

    # Extract positions from Graphviz layout
    pos = {}
    for node in G.nodes():
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

        scale = 500
        pos = {
            node: (
                ((p[0] - x_min) / x_range * scale * 0.8 + scale * 0.1),
                ((p[1] - y_min) / y_range * scale * 0.8 + scale * 0.1)
            )
            for node, p in pos.items()
        }

    # Create Bokeh plot
    plot = figure(
        width=1200,
        height=800,
        x_range=(-50, 600),
        y_range=(-50, 600),
        toolbar_location="above",
        background_fill_color="#bebebe",
        title=f"Power Train Network (Layout: {layout_prog})",
    )
    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.xaxis.visible = False
    plot.yaxis.visible = False

    # Prepare node data
    node_indices = list(G.nodes())
    node_x = [pos[node][0] for node in node_indices]
    node_y = [pos[node][1] for node in node_indices]
    node_image_urls = [icons_dict[node_icons[node]] for node in node_indices]
    node_sizes_list = [node_sizes[node] * 2 for node in node_indices]
    node_types_list = [node_types[node] for node in node_indices]

    # Convert file paths to file:// URLs for local images
    node_image_urls = ["file://" + str(Path(url).resolve()) for url in node_image_urls]

    # Prepare edge data for Segment glyphs
    edge_x0 = []
    edge_y0 = []
    edge_x1 = []
    edge_y1 = []

    for edge in G.edges():
        start, end = edge
        start_x, start_y = pos[start]
        end_x, end_y = pos[end]
        edge_x0.append(start_x)
        edge_y0.append(start_y)
        edge_x1.append(end_x)
        edge_y1.append(end_y)

    # Create edge data source
    edge_source = ColumnDataSource(
        data=dict(
            x0=edge_x0,
            y0=edge_y0,
            x1=edge_x1,
            y1=edge_y1,
        )
    )

    # Add Segment glyphs for edges with animation via CSS/Bokeh styling
    segment_renderer = plot.segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        source=edge_source,
        line_color="#666666",
        line_width=2,
        line_alpha=0.7,
    )

    # Add custom styling for animated dash effect
    segment_renderer.glyph.line_dash = "dotted"

    # Draw nodes as images
    node_source = ColumnDataSource(
        data=dict(
            x=node_x,
            y=node_y,
            url=node_image_urls,
            w=[s for s in node_sizes_list],
            h=[s for s in node_sizes_list],
            name=node_indices,
            type=node_types_list,
        )
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
    hover = HoverTool(
        tooltips=[
            ("Name", "@name"),
            ("Type", "@type"),
        ]
    )
    plot.add_tools(hover, TapTool(), BoxSelectTool())

    # Create directory if it doesn't exist
    directory_to_save_graph = os.path.dirname(network_file_path)
    if directory_to_save_graph and not os.path.exists(directory_to_save_graph):
        os.makedirs(directory_to_save_graph)

    # Save the plot
    output_file(network_file_path)
    save(plot)

    return plot