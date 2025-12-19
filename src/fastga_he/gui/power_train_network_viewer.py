# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import base64

import networkx as nx
import bokeh.plotting as bkplot
import bokeh.models as bkmodel

from bokeh.server.server import Server as bkserver
from bokeh.layouts import column
from pathlib import Path

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from . import icons
from .layout_generation import HierarchicalLayout

BACKGROUND_COLOR_CODE = "#bebebe"
# canvas background color (french gray)
ELECTRICITY_CURRENT_COLOR_CODE = "#007BFF"
# color for electricity transmitting connections (artyClick deep sky blue)
FUEL_FLOW_COLOR_CODE = "#FF5722"
# color for fuel (including hydrogen) transmitting connections (portland orange)
MECHANICAL_POWER_COLOR_CODE = "#2E7D32"
# color for mechanical power transmitting connections (medium forest green)
DEFAULT_COLOR = "gray"

ICON_FOLDER_PATH = Path(icons.__path__[0])

# Image URLs for graph nodes
# "icon_file_name" : [file_path, color_as_source, color_as_target]
icons_dict = {
    "battery": {
        "icon_path": ICON_FOLDER_PATH / "battery.png",
        "source_color": None,
        "target_color": ELECTRICITY_CURRENT_COLOR_CODE,
    },
    "bus_bar": {
        "icon_path": ICON_FOLDER_PATH / "bus_bar.png",
        "source_color": ELECTRICITY_CURRENT_COLOR_CODE,
        "target_color": ELECTRICITY_CURRENT_COLOR_CODE,
    },
    "cable": {
        "icon_path": ICON_FOLDER_PATH / "cable.png",
        "source_color": ELECTRICITY_CURRENT_COLOR_CODE,
        "target_color": ELECTRICITY_CURRENT_COLOR_CODE,
    },
    "e_motor": {
        "icon_path": ICON_FOLDER_PATH / "e_motor.png",
        "source_color": ELECTRICITY_CURRENT_COLOR_CODE,
        "target_color": MECHANICAL_POWER_COLOR_CODE,
    },
    "generator": {
        "icon_path": ICON_FOLDER_PATH / "generator.png",
        "source_color": MECHANICAL_POWER_COLOR_CODE,
        "target_color": ELECTRICITY_CURRENT_COLOR_CODE,
    },
    "ice": {
        "icon_path": ICON_FOLDER_PATH / "ice.png",
        "source_color": FUEL_FLOW_COLOR_CODE,
        "target_color": MECHANICAL_POWER_COLOR_CODE,
    },
    "switch": {
        "icon_path": ICON_FOLDER_PATH / "switch.png",
        "source_color": ELECTRICITY_CURRENT_COLOR_CODE,
        "target_color": ELECTRICITY_CURRENT_COLOR_CODE,
    },
    "propeller": {
        "icon_path": ICON_FOLDER_PATH / "propeller.png",
        "source_color": MECHANICAL_POWER_COLOR_CODE,
        "target_color": None,
    },
    "splitter": {
        "icon_path": ICON_FOLDER_PATH / "splitter.png",
        "source_color": ELECTRICITY_CURRENT_COLOR_CODE,
        "target_color": ELECTRICITY_CURRENT_COLOR_CODE,
    },
    "rectifier": {
        "icon_path": ICON_FOLDER_PATH / "AC_DC.png",
        "source_color": ELECTRICITY_CURRENT_COLOR_CODE,
        "target_color": ELECTRICITY_CURRENT_COLOR_CODE,
    },
    "dc_converter": {
        "icon_path": ICON_FOLDER_PATH / "DC_DC.png",
        "source_color": ELECTRICITY_CURRENT_COLOR_CODE,
        "target_color": ELECTRICITY_CURRENT_COLOR_CODE,
    },
    "inverter": {
        "icon_path": ICON_FOLDER_PATH / "DC_AC.png",
        "source_color": ELECTRICITY_CURRENT_COLOR_CODE,
        "target_color": ELECTRICITY_CURRENT_COLOR_CODE,
    },
    "fuel_tank": {
        "icon_path": ICON_FOLDER_PATH / "fuel_tank.png",
        "source_color": None,
        "target_color": FUEL_FLOW_COLOR_CODE,
    },
    "fuel_system": {
        "icon_path": ICON_FOLDER_PATH / "fuel_system.png",
        "source_color": FUEL_FLOW_COLOR_CODE,
        "target_color": FUEL_FLOW_COLOR_CODE,
    },
    "turbine": {
        "icon_path": ICON_FOLDER_PATH / "turbine.png",
        "source_color": FUEL_FLOW_COLOR_CODE,
        "target_color": MECHANICAL_POWER_COLOR_CODE,
    },
    "gearbox": {
        "icon_path": ICON_FOLDER_PATH / "gears.png",
        "source_color": MECHANICAL_POWER_COLOR_CODE,
        "target_color": MECHANICAL_POWER_COLOR_CODE,
    },
    "fuel_cell": {
        "icon_path": ICON_FOLDER_PATH / "fuel_cell.png",
        "source_color": FUEL_FLOW_COLOR_CODE,
        "target_color": ELECTRICITY_CURRENT_COLOR_CODE,
    },
}

color_icon_dict = {
    "fuel": ICON_FOLDER_PATH / "fuel.png",
    "mechanical": ICON_FOLDER_PATH / "mechanical.png",
    "electricity": ICON_FOLDER_PATH / "electricity.png",
}


def _get_edge_color(source_icon, target_icon):
    """
    Determine edge color based on source and target node types.

    :param source_icon: Icon name of the source node
    :param target_icon: Icon name of the target node

    :return: Color code for the edge
    """
    # edge color for the source component serves as source
    color_as_source = icons_dict.get(source_icon)["source_color"]
    # edge color for the target component serves as source
    color_as_target = icons_dict.get(target_icon)["target_color"]

    if color_as_target:
        return color_as_target

    # For propulsor component which won't be connected as target
    elif color_as_source:
        return color_as_source

    else:
        return DEFAULT_COLOR


def power_train_network_viewer(
    power_train_file_path: str,
    network_file_path: str,
    orientation: str = "TB",
    legend_position: str = "TR",
    static_html: bool = True,
    sorting: bool = True,
    from_propulsor: bool = False,
    plot_scaling: float = 1.0,
    legend_scaling: float = 1.0,
    port: int = 5006,
    address: str = "localhost",
    refresh_rate: int = None,
):
    """
    Create an interactive network visualization of a power train using Bokeh with NetworkX layout.

    :param power_train_file_path: Path to the power train configuration file
    :param network_file_path: Path where the HTML output will be saved
    :param legend_position: String defines the legend position
    :param orientation: network plot orientation
    :param static_html: True if using static html
    :param sorting: True to enable tutte's drawing algorithm for sorting
    :param from_propulsor: Set all propulsor component into reference layer of the hierarchy
    :param plot_scaling: Scaling factor for the main powertrain architecture
    :param legend_scaling: Scaling factor for the legend size
    """

    # Variables for animated powertrain display
    plot, edge_source, node_source, node_image_sequences, propeller_rotation_sequences = (
        _create_network_plot(
            power_train_file_path=power_train_file_path,
            orientation=orientation,
            legend_position=legend_position,
            static_html=static_html,
            sorting=sorting,
            from_propulsor=from_propulsor,
            plot_scaling=abs(plot_scaling),
            legend_scaling=abs(legend_scaling),
        )
    )

    if static_html:
        _save_static_html(plot, network_file_path)

    else:
        if refresh_rate is None:
            refresh_rate = _get_monitor_refresh_rate()

        callback_interval = _calculate_optimal_callback_interval(
            refresh_rate, target_fps=refresh_rate
        )
        animation_frames = _calculate_animation_frames(refresh_rate, animation_duration_ms=1000)
        propeller_frames = 12

        def make_document(doc):
            animation_counter = [0]

            doc.add_root(column(plot))

            def update():
                animation_counter[0] = (animation_counter[0] + 1) % animation_frames
                progress = animation_counter[0] / animation_frames

                num_segments = len(edge_source.data["xs"])
                edge_id_list = edge_source.data["edge_id"]
                num_edges = max(edge_id_list) + 1 if edge_id_list else 1
                segments_per_edge = num_segments // num_edges if num_edges > 0 else 1

                new_alphas = []

                for i in range(num_segments):
                    edge_idx = edge_id_list[i]
                    segment_idx = i % segments_per_edge if segments_per_edge > 0 else 0

                    seg_position = segment_idx / segments_per_edge if segments_per_edge > 0 else 0
                    wave_pos = (-progress + edge_idx * 0.1) % 1.0

                    distance = abs(seg_position - wave_pos)

                    if distance < 0.3:
                        alpha = 0.3 + 0.7 * (1 - (distance / 0.3) ** 2)
                    else:
                        alpha = 0.3

                    new_alphas.append(alpha)

                edge_source.patch({"line_alpha": [(slice(len(new_alphas)), new_alphas)]})

                if propeller_rotation_sequences:
                    frame_index = (
                        animation_counter[0] * propeller_frames // animation_frames
                    ) % propeller_frames

                    new_urls = []
                    updated = False

                    for i, node in enumerate(node_source.data["name"]):
                        if node in propeller_rotation_sequences:
                            seq = propeller_rotation_sequences[node]
                            frame_idx = frame_index % len(seq)
                            new_urls.append(seq[frame_idx])
                            updated = True
                        elif node in node_image_sequences:
                            seq = node_image_sequences[node]
                            frame_idx = (animation_counter[0] * len(seq) // animation_frames) % len(
                                seq
                            )
                            new_urls.append(seq[frame_idx])
                            updated = True
                        else:
                            new_urls.append(node_source.data["url"][i])

                    if updated and new_urls and len(new_urls) == len(node_source.data["url"]):
                        node_source.patch({"url": [(slice(len(new_urls)), new_urls)]})

            doc.add_periodic_callback(update, callback_interval)

        _start_bokeh_server(
            make_document, port, address, refresh_rate, callback_interval, animation_frames
        )


def _create_network_plot(
    power_train_file_path: str,
    orientation: str = "TB",
    legend_position: str = "TR",
    static_html: bool = True,
    sorting: bool = True,
    from_propulsor: bool = False,
    plot_scaling: float = 1.0,
    legend_scaling: float = 1.0,
):
    """
    Create an interactive network visualization of a power train using Bokeh with NetworkX layout.

    :param power_train_file_path: Path to the power train configuration file
    :param orientation: network plot orientation ('TB', 'BT', 'LR', 'RL')
    (T: top, B: button, L: left, R: right)
    :param legend_position: String defines the legend box position
    :param static_html: True if using static html
    :param sorting: True to enable tutte's drawing algorithm for sorting
    :param from_propulsor: Set all propulsor component into reference layer of the hierarchy
    :param plot_scaling: Scaling factor for the main powertrain architecture
    :param legend_scaling: Scaling factor for the legend size
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

    distance_from_energy_storage = configurator.get_component_distance(["tank", "battery_pack"])

    # For animation purposes
    node_image_sequences = {}
    propeller_rotation_sequences = {}

    propeller_names, node_sizes, node_types, node_om_types, node_icons = _define_hierarchy_elements(
        graph, names, connections, components_type, components_om_type, icons_name, icons_size
    )

    # Build node_layer_dict from distance_from_energy_storage
    node_layer_dict = {}
    max_distance = max(distance_from_energy_storage.values())

    for node_name, distance in distance_from_energy_storage.items():
        if max_distance > distance:
            node_layer_dict[node_name] = max_distance - distance

        else:
            if node_name in propeller_names:
                node_layer_dict[node_name] = max_distance - distance

            # Triggered if there is a non-propulsor component at the top level
            else:
                from_propulsor = True
                break

    if from_propulsor:
        distance_from_propulsor = configurator.get_component_distance("propulsor")

        for node_name, distance in distance_from_propulsor.items():
            node_layer_dict[node_name] = distance

    # Compute layout based on specified algorithm with hierarchy from distance_from_energy_storage
    position_dict = HierarchicalLayout(
        graph, orientation, node_layer_dict, sorting
    ).generate_networkx_layout()

    plot, position_dict, icon_factor, icon_width_factor = _create_bokeh_plot(
        power_train_file_path, position_dict, orientation, plot_scaling
    )

    (
        node_source,
        node_x,
        node_y,
        node_width,
        node_height,
        node_name_list,
        node_types_list,
        node_om_types_list,
    ) = _build_node_dict_bokeh(
        graph,
        position_dict,
        node_types,
        node_om_types,
        node_icons,
        icon_width_factor,
        node_sizes,
        icon_factor,
        plot_scaling,
        static_html,
    )

    edge_source = _build_edge_dict_bokeh(graph, position_dict, node_icons, static_html)

    # Draw edge lines
    plot.multi_line(
        xs="xs",
        ys="ys",
        source=edge_source,
        line_color="line_color",
        line_width=3,
        line_alpha="line_alpha",
    )

    # Add circle to cover edge line
    plot.scatter(
        x="x",
        y="y",
        size=45 * plot_scaling,
        source=node_source,
        color=BACKGROUND_COLOR_CODE,
        line_alpha=0,
    )

    # Draw nodes as image
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
    label_source = bkmodel.ColumnDataSource(
        data=dict(
            x=node_x,
            y=[y - 15 * icon_factor * plot_scaling * 0.7 for i, y in enumerate(node_y)],
            names=node_name_list,
        )
    )

    labels = bkmodel.LabelSet(
        x="x",
        y="y",
        text="names",
        source=label_source,
        text_align="center",
        text_baseline="top",
        text_color="white",
        text_font_size=str(8 * plot_scaling) + "pt",
    )
    plot.add_layout(labels)

    color_icon_urls = [
        "file://" + str(Path(color_icon_dict[color_icon]).resolve())
        for color_icon in color_icon_dict.keys()
    ]

    _add_color_legend_separate(plot, legend_position, color_icon_urls, legend_scaling)

    # Add interactive tools
    cleaned_node_types = []
    cleaned_node_om_types = []
    for node_type, node_om_type in zip(node_types_list, node_om_types_list):
        if isinstance(node_type, str):
            cleaned_node_types.append(_string_clean_up(node_type.capitalize()))
        else:
            cleaned_node_types.append(_string_clean_up(node_type))

        cleaned_node_om_types.append(_string_clean_up(node_om_type))

    # Define hover list info
    hover_source = bkmodel.ColumnDataSource(
        data=dict(
            x=node_x,
            y=node_y,
            w=node_width,
            h=node_height,
            name=node_name_list,
            type_class=cleaned_node_types,
            component_type=cleaned_node_om_types,
        )
    )

    # Add invisible circles on top for hover interactivity
    plot.scatter(
        x="x",
        y="y",
        size=55 * plot_scaling,
        source=hover_source,
        fill_alpha=0,
        line_alpha=0,
        hover_fill_alpha=0.1,
        hover_line_alpha=0.3,
    )

    hover = bkmodel.HoverTool(
        tooltips=[
            ("Name", "@name"),
            ("Type class", "@type_class"),
            ("Component type", "@component_type"),
        ]
    )
    plot.add_tools(hover, bkmodel.BoxSelectTool())

    return plot, edge_source, node_source, node_image_sequences, propeller_rotation_sequences


def _define_hierarchy_elements(
    graph, names, connections, components_type, components_om_type, icons_name, icons_size
):
    """Define the nodes and edges for networkx and the node properties for bokeh."""

    # Build node attributes dictionaries
    propeller_names = []
    node_sizes = {}
    node_types = {}
    node_om_types = {}
    node_icons = {}

    for component_name, component_type, om_type, icon_name, icon_size in zip(
        names, components_type, components_om_type, icons_name, icons_size
    ):
        graph.add_node(component_name)
        if component_type == "propulsor":
            propeller_names.append(component_name)
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

    return propeller_names, node_sizes, node_types, node_om_types, node_icons


def _create_bokeh_plot(power_train_file_path, position_dict, orientation, plot_scaling):
    """Create a bokeh plot to add the node and edges from the networkx plot."""

    # Normalize positions for Bokeh
    x_coordinates = []
    y_coordinates = []

    for coords in position_dict.values():
        x_coordinates.append(coords[0])
        y_coordinates.append(coords[1])

    x_min, x_max = min(x_coordinates), max(x_coordinates)
    y_min, y_max = min(y_coordinates), max(y_coordinates)

    x_range = x_max - x_min if x_max > x_min else 1  # For the case of straight structure
    y_range = y_max - y_min if y_max > y_min else 1  # For the case of straight structure

    if orientation not in ["TB", "BT", "LR", "RL"]:
        orientation = "TB"

    # Scale to a reasonable display size with different orientation
    if orientation == "TB" or orientation == "BT":
        x_factor = 0.5
        y_factor = 1.0
        plot_width_factor = 1
        icon_factor = 1
        icon_width_factor = 0.8
        x_orientation_offset = 125
        y_orientation_offset = 0.0

    elif orientation == "LR" or orientation == "RL":
        x_factor = 1.0
        y_factor = 0.5
        plot_width_factor = 1.25
        icon_factor = 1.75
        icon_width_factor = 0.6
        x_orientation_offset = -25
        y_orientation_offset = 150

    # Here update the position from NetworkX to fit in the bokeh plot

    position_dict = {
        node: (
            ((coordinate[0] - x_min) / x_range * 550 * x_factor + x_orientation_offset),
            ((coordinate[1] - y_min) / y_range * 550 * y_factor + y_orientation_offset),
        )
        for node, coordinate in position_dict.items()
    }

    # Create Bokeh plot
    plot = bkplot.figure(
        width=int(1200 * plot_scaling * plot_width_factor),
        height=int(900 * plot_scaling),
        x_range=(-50, 600),
        y_range=(-50, 600),
        toolbar_location="above",
        background_fill_color=BACKGROUND_COLOR_CODE,
        title=_get_file_name(power_train_file_path),
    )

    plot.xgrid.visible = False
    plot.ygrid.visible = False
    plot.xaxis.visible = False
    plot.yaxis.visible = False

    return plot, position_dict, icon_factor, icon_width_factor


def _build_node_dict_bokeh(
    graph,
    position_dict,
    node_types,
    node_om_types,
    node_icons,
    icon_width_factor,
    node_sizes,
    icon_factor,
    plot_scaling,
    static_html,
):
    """Create the dictionary of nodes for add the nodes into bokeh plot."""

    # Prepare node data
    node_name_list = list(graph.nodes())
    node_x = []
    node_y = []
    node_width = []
    node_height = []
    node_types_list = []
    node_om_types_list = []

    for node in node_name_list:
        node_x.append(position_dict[node][0])
        node_y.append(position_dict[node][1])
        node_height.append(node_sizes[node] * icon_factor * plot_scaling)
        node_width.append(node_sizes[node] * icon_factor * icon_width_factor * plot_scaling)
        node_types_list.append(node_types[node])
        node_om_types_list.append(node_om_types[node])

    # Convert file paths to file:// URLs for local images
    if static_html:
        node_image_urls = [
            "file://" + str(Path(icons_dict[node_icons[node]]["icon_path"]).resolve())
            for node in node_name_list
        ]

        node_source = bkmodel.ColumnDataSource(
            data=dict(
                x=node_x,
                y=node_y,
                url=node_image_urls,
                w=node_width,
                h=node_height,
            )
        )

    else:
        node_image_urls = []
        node_image_sequences = {}  # Store animation sequences for nodes
        propeller_rotation_sequences = {}
        rotation_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]  # 12 frames

        # for node in node_name_list:
        #     if (
        #         isinstance(icons_dict[node_icons[node]]["icon_path"], list)
        #         and node_icons[node] != "propeller"
        #     ):
        #         # Animated icon
        #         animation_frames = []
        #         for frame_path in

    return (
        node_source,
        node_x,
        node_y,
        node_width,
        node_height,
        node_name_list,
        node_types_list,
        node_om_types_list,
    )


def _build_edge_dict_bokeh(graph, position_dict, node_icons, static_html):
    """Create the dictionary of edges for add the edges into bokeh plot."""

    # Create edge data with colors
    edge_x_pos = []
    edge_y_pos = []
    edge_colors = []

    for edge in graph.edges():
        start, end = edge
        edge_x_pos.append([position_dict[start][0], position_dict[end][0]])
        edge_y_pos.append([position_dict[start][1], position_dict[end][1]])

        # Determine edge color based on connected nodes
        source_icon = node_icons[start]
        target_icon = node_icons[end]
        edge_color = _get_edge_color(source_icon, target_icon)
        edge_colors.append(edge_color)

    # Draw edges
    if static_html:
        edge_source = bkmodel.ColumnDataSource(
            data=dict(
                xs=edge_x_pos,
                ys=edge_y_pos,
                line_color=edge_colors,
                line_alpha=[0.7] * len(edge_x_pos),
            )
        )
    else:
        seg_xs, seg_ys, seg_alphas, edge_ids, seg_colors = _create_segmented_edges(
            edge_x_pos, edge_y_pos, edge_colors, segments_per_edge=30
        )

        edge_source = bkmodel.ColumnDataSource(
            data=dict(
                xs=seg_xs,
                ys=seg_ys,
                colors=seg_colors,
                line_alpha=seg_alphas,
                edge_id=edge_ids,  # This is the missing definition
            )
        )

    return edge_source


def _add_color_legend_separate(plot, legend_position, color_icon_urls, legend_scaling: float = 1.0):
    """
    Add a color legend as a separate visual entity within the same plot.

    :param plot: Bokeh figure object
    :param legend_position: String defines the legend box position
    (T: top, M: middle (vertical), Button, L: left, R: right, C: center (horizontal))

    +-----+-----+-----+-----+-----+
    |     |     |  T  |     |     |
    +-----+-----+-----+-----+-----+
    |     |     |     |     |     |
    +-----+-----+-----+-----+-----+
    |     |     |  M  |     |     |
    +-----+-----+-----+-----+-----+
    |  L  |     |  C  |     |  R  |
    +-----+-----+-----+-----+-----+
    |     |     |  B  |     |     |
    +-----+-----+-----+-----+-----+

    :param color_icon_urls: List of URLs for color icons
    :param legend_scaling: Scaling factor for the legend size
    """

    if len(legend_position) != 2:
        legend_scaling = "TR"

    if "T" in legend_position:
        legend_y_start = 600
    elif "B" in legend_position:
        legend_y_start = 50
    elif "M" in legend_position:
        legend_y_start = 325
    else:
        legend_y_start = 600

    if "R" in legend_position:
        legend_x_start = 500
    elif "L" in legend_position:
        legend_x_start = -50
    elif "C" in legend_position:
        legend_x_start = 225
    else:
        legend_x_start = 500

    # Legend items
    legend_items = [
        (0, "Fuel Flow"),
        (1, "Mechanical Power"),
        (2, "Electrical Current"),
    ]

    legend_item_height = int(22 * legend_scaling)
    legend_item_start_y = legend_y_start - int(25 * legend_scaling)

    for i, (icon_idx, description) in enumerate(legend_items):
        y_position = legend_item_start_y - (i * legend_item_height)

        # Create data source for icon
        icon_source = bkmodel.ColumnDataSource(
            data=dict(
                x=[legend_x_start + int(10 * legend_scaling)],
                y=[y_position],
                url=[color_icon_urls[icon_idx]],
            )
        )

        # Draw color icon image
        plot.image_url(
            url="url",
            x="x",
            y="y",
            w=9 * legend_scaling,
            h=12 * legend_scaling,
            anchor="center",
            source=icon_source,
        )

        # Add text label next to the color icon image
        label_source = bkmodel.ColumnDataSource(
            data=dict(
                x=[legend_x_start + 25],
                y=[y_position],
                text=[description],
            )
        )

        labels = bkmodel.LabelSet(
            x="x",
            y="y",
            text="text",
            source=label_source,
            text_align="left",
            text_baseline="middle",
            text_color="white",
            text_font_size=str(10 * legend_scaling) + "pt",
        )
        plot.add_layout(labels)


def _string_clean_up(old_string):
    """
    Clean up list content for better readability.
    """
    # In case for list type definition
    if isinstance(old_string, list):
        old_string = ", ".join([old_string[0].capitalize(), old_string[1].capitalize()])

    # Replace underscore with space
    new_string = old_string.replace("_", " ")

    # Add a space after 'DC' if followed immediately by a letter or number
    new_string = new_string.replace("DC", "DC ")
    new_string = new_string.replace("DC DC", "DC-DC")

    # Add a space after 'H2' and 'PEMFC'
    new_string = new_string.replace("H2", "H2 ")
    new_string = new_string.replace("PEMFC", "PEMFC ")

    # Add space before a capital letter preceded by a lowercase letter
    new_string = _add_space_before_caps(new_string)

    # Remove extra spaces
    new_string = " ".join(new_string.split())

    return new_string


def _add_space_before_caps(text):
    """
    Add space before a capital letter preceded by a lowercase letter.
    """
    result = []
    for i, char in enumerate(text):
        if i > 0 and char.isupper() and text[i - 1].islower():
            result.append(" " + char)
        else:
            result.append(char)
    return "".join(result)


def _get_file_name(file_path):
    """
    Using the file name as the plot title.
    """
    file_path = str(file_path)

    # Extract filename from path (handle both / and \ separators)
    filename = file_path.replace("\\", "/").split("/")[-1]

    # Check if it ends with .yml
    if filename.endswith(".yml"):
        # Remove .yml extension and process
        filename = filename[:-4]  # Remove last 4 characters (.yml)
        filename = filename.replace("_", " ").capitalize()

        return f"{filename} powertrain network"


def _save_static_html(plot, file_path):
    """
    Save the network plot as static html.
    """
    file_path = Path(file_path)

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the plot
    bkplot.output_file(str(file_path))
    bkplot.save(plot)

    # Read the generated HTML
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Replace file:// URLs in the HTML string directly
    try:
        # Split by 'file://' and process each part
        parts = html_content.split("file://")
        result = [parts[0]]  # Keep the part before first file://

        for part in parts[1:]:
            # Find the end of the URL (quote or comma or bracket)
            end_chars = ['"', ",", "]", "}"]
            end_idx = len(part)

            for char in end_chars:
                idx = part.find(char)
                if idx != -1 and idx < end_idx:
                    end_idx = idx

            # Extract the URL and convert it
            url = "file://" + part[:end_idx]
            converted = _url_to_base64(url)

            # Reconstruct: converted URL + rest of the part
            result.append(converted + part[end_idx:])

        html_content = "".join(result)
    except Exception as e:
        print(f"Error processing URLs: {e}")

    # Write the updated HTML back
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Static HTML saved to: {file_path}")


def _url_to_base64(url):
    """
    Convert a file:// URL to a Base64 data URI.
    :param url: File path or file:// URL to convert

    Returns: Base64 data URI string, or original URL if conversion fails
    """
    # Skip non-file URLs
    if not url.startswith("file://"):
        return url

    try:
        # Convert file:// URL to local path
        file_path_str = url.replace("file:///", "").replace("file://", "")
        local_path = Path(file_path_str)

        # Read and encode as Base64
        with open(local_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        # Determine MIME type
        suffix = local_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".svg": "image/svg+xml",
        }
        mime_type = mime_types.get(suffix, "image/png")

        # Return data URI
        return f"data:{mime_type};base64,{img_data}"

    except Exception as e:
        print(f"Error processing {url}: {e}")
        return url


def _get_monitor_refresh_rate():
    """
    Detect monitor refresh rate from system settings.

    :return: Monitor refresh rate in Hz (default 60 if detection fails)
    """
    try:
        import platform

        system = platform.system()

        if system == "Windows":
            try:
                import subprocess

                # Use wmic to get refresh rate on Windows
                result = subprocess.run(
                    ["wmic", "path", "win32_videocontroller", "get", "currentrefreshrate"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                if len(lines) > 1:
                    try:
                        rate = int(lines[1])
                        if rate > 0:
                            print(f"✓ Detected monitor refresh rate: {rate} Hz")
                            return rate
                    except (ValueError, IndexError) as e:
                        print(f"✗ Could not parse refresh rate from wmic output: {lines}")
            except FileNotFoundError:
                print("✗ wmic command not found, trying alternative method...")
                try:
                    import subprocess

                    # Alternative: Use Get-WmiObject PowerShell command
                    result = subprocess.run(
                        [
                            "powershell",
                            "-Command",
                            "Get-WmiObject -Namespace root\\cimv2 -Class Win32_VideoController | Select-Object CurrentRefreshRate",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    lines = [
                        line.strip()
                        for line in result.stdout.strip().split("\n")
                        if line.strip() and line.strip().isdigit()
                    ]
                    if lines:
                        rate = int(lines[0])
                        if rate > 0:
                            print(f"✓ Detected monitor refresh rate: {rate} Hz")
                            return rate
                except Exception as e2:
                    print(f"✗ PowerShell method also failed: {e2}")
            except Exception as e:
                print(f"✗ Error detecting refresh rate: {e}")

        elif system == "Linux":
            try:
                import subprocess

                result = subprocess.run(["xrandr"], capture_output=True, text=True, timeout=5)
                # Parse xrandr output for refresh rate
                for line in result.stdout.split("\n"):
                    if "*" in line:  # Current mode
                        parts = line.split()
                        for part in parts:
                            if "+" in part:
                                rate = float(part.replace("+", ""))
                                print(f"✓ Detected monitor refresh rate: {int(rate)} Hz")
                                return int(rate)
            except Exception as e:
                print(f"✗ Error detecting refresh rate: {e}")

    except Exception as e:
        print(f"✗ Unexpected error detecting refresh rate: {e}")

    print("⚠ Could not detect refresh rate, using default: 60 Hz")
    return 60


def _calculate_optimal_callback_interval(refresh_rate, target_fps=None):
    """
    Calculate optimal callback interval based on monitor refresh rate.

    :param refresh_rate (int): Monitor refresh rate in Hz
    :param target_fps (int): Target animation FPS (default: match refresh rate)

    :return callback_interval (int): Milliseconds between callbacks
    """
    if target_fps is None:
        target_fps = refresh_rate

    # Ensure target FPS doesn't exceed refresh rate
    target_fps = min(target_fps, refresh_rate)

    callback_interval = int(1000 / target_fps)
    return callback_interval


def _calculate_animation_frames(refresh_rate, animation_duration_ms=1000):
    """
    Calculate animation frame count for smooth animations synchronized to refresh rate.

    :param refresh_rate (int): Monitor refresh rate in Hz
    :param animation_duration_ms (int): Desired animation duration in milliseconds

    :return frames (int): Number of frames for smooth animation
    """
    # Calculate frames needed for smooth animation

    return max(refresh_rate, int(refresh_rate * animation_duration_ms / 1000))


def _create_rotated_svg(base64_url, rotation_degrees):
    """
    Wrap a base64 image in an SVG with rotation transform without cropping.


    :param base64_url: Data URL of the image (e.g., "data:image/png;base64,...")
    :param rotation_degrees: Rotation angle in degrees (counter-clockwise)

    return: SVG data URL with rotated image
    """

    # Use a larger SVG canvas to prevent clipping during rotation
    # The diagonal of a 100x100 square is ~141, so use 150x150 to be safe.
    # This is based on the size of the original png image of propelller icon.
    svg_template = f"""<svg width="150" height="150" viewBox="0 0 150 150" 
    xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
        <g transform="translate(75 75) rotate({rotation_degrees}) translate(-50 -50)">
            <image href="{base64_url}" x="0" y="0" width="100" height="100"/>
        </g>
    </svg>"""

    svg_b64 = base64.b64encode(svg_template.encode()).decode()
    return f"data:image/svg+xml;base64,{svg_b64}"


def _create_segmented_edges(edge_x_pos, edge_y_pos, edge_colors, segments_per_edge=10):
    """
    Break each edge into multiple segments for flowing animation.

    :param edge_start_x, edge_start_y: Lists of edge starting coordinates
    :param edge_end_x, edge_end_y: Lists of edge ending coordinates
    :param segments_per_edge: Number of segments to divide each edge into

    return: Lists of segment endpoints and metadata for animation
    """
    seg_xs = []  # List of [x1, x2] for each segment
    seg_ys = []  # List of [y1, y2] for each segment
    seg_alphas = []
    seg_colors = []
    edge_ids = []  # Track which edge each segment belongs to

    for edge_idx, (edge_x, edge_y, color) in enumerate(zip(edge_x_pos, edge_y_pos, edge_colors)):
        sx = edge_x[0]
        ex = edge_x[1]
        sy = edge_y[0]
        ey = edge_y[1]
        for seg in range(segments_per_edge):
            # Interpolate segment endpoints
            t_start = seg / segments_per_edge
            t_end = (seg + 1) / segments_per_edge

            x1 = sx + (ex - sx) * t_start
            y1 = sy + (ey - sy) * t_start
            x2 = sx + (ex - sx) * t_end
            y2 = sy + (ey - sy) * t_end

            seg_xs.append([x1, x2])
            seg_ys.append([y1, y2])
            seg_alphas.append(0.7)  # Initial alpha
            seg_colors.append(color)
            edge_ids.append(edge_idx)

    return seg_xs, seg_ys, seg_alphas, edge_ids, seg_colors


def _start_bokeh_server(
    make_document, port, address, refresh_rate, callback_interval, animation_frames
):
    """
    Start and run a Bokeh Server with the provided document maker function.

    :param make_document: Function that creates the Bokeh document
    :param port (int): Port to run the server on
    :param address (str): Server address (default: localhost)
    :param refresh_rate (int): Monitor refresh rate in Hz
    :param callback_interval (int): Milliseconds between callbacks
    :param animation_frames (int): Number of animation frames
    """
    server = bkserver(
        {"/": make_document},
        port=port,
        address=address,
        num_procs=1,
    )

    print(f"\n{'=' * 60}")
    print(f"Bokeh Server started!")
    print(f"Monitor refresh rate: {refresh_rate} Hz")
    print(f"Callback interval: {callback_interval}ms")
    print(f"Animation frames: {animation_frames}")
    print(f"Open your browser and go to:")
    print(f"  http://{address}:{port}/")
    print(f"{'=' * 60}\n")

    server.start()
    server.io_loop.start()
