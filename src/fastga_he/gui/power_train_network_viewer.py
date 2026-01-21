# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import base64
import logging
from pathlib import Path
import networkx as nx
import bokeh.plotting as bkplot
import bokeh.models as bkmodel
import pandas as pd
from bokeh.server.server import Server
from bokeh.layouts import row, column
from tornado.ioloop import IOLoop
import webbrowser

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
ICONS_CONFIG = {
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

COLOR_ICON_CONFIG = {
    "fuel": ICON_FOLDER_PATH / "fuel.png",
    "mechanical": ICON_FOLDER_PATH / "mechanical.png",
    "electricity": ICON_FOLDER_PATH / "electricity.png",
}

# ============================================================================
# Main visualization function
# ============================================================================


def power_train_network_viewer(
    power_train_file_path: str,
    network_file_path: str,
    orientation: str = "TB",
    legend_position: str = "TR",
    static_architecture: bool = True,
    sorting: bool = True,
    from_propulsor: bool = False,
    plot_scaling: float = 1.0,
    legend_scaling: float = 1.0,
    port: int = 5006,
    address: str = "localhost",
    refresh_rate: int = 60,
    pt_watcher_path: str = None,
):
    """
    Create an interactive network visualization of a power train using Bokeh.

    :param power_train_file_path: Path to the power train configuration file
    :param network_file_path: Path where the HTML output will be saved
    :param orientation: Network plot orientation ('TB', 'BT', 'LR', 'RL')
    :param legend_position: Legend position ('TR', 'TL', 'BR', 'BL', etc.)
    :param static_architecture: True for static HTML, False for interactive server
    :param sorting: Enable Tutte's drawing algorithm for sorting
    :param from_propulsor: Set all propulsor components into reference layer
    :param plot_scaling: Scaling factor for the main powertrain architecture
    :param legend_scaling: Scaling factor for the legend size
    :param port: Port for Bokeh server
    :param address: Server address
    :param refresh_rate: Monitor refresh rate
    :param pt_watcher_path: Path to PT watcher file with performance data
    """

    # Build graph
    graph_builder = GraphBuilder(power_train_file_path)
    propeller_names, node_sizes, node_types, node_om_types, node_icons = (
        graph_builder._build_graph()
    )

    # Compute hierarchy
    node_layer_dict = graph_builder._get_hierarchy_layers(propeller_names, from_propulsor)

    # Generate layout
    position_dict = HierarchicalLayout(
        graph_builder.graph, orientation, node_layer_dict, sorting
    ).generate_networkx_layout()

    # Create Bokeh plot
    plot, position_dict, icon_factor, icon_width_factor = BokehPlotBuilder._create_plot(
        power_train_file_path, position_dict, orientation, plot_scaling
    )

    # Build nodes
    (
        node_source,
        node_x,
        node_y,
        node_width,
        node_height,
        node_name_list,
        node_types_list,
        node_om_types_list,
        component_perf,
    ) = NodesBuilder._build_nodes(
        graph_builder.graph,
        position_dict,
        node_types,
        node_om_types,
        node_icons,
        icon_width_factor,
        node_sizes,
        icon_factor,
        plot_scaling,
        static_architecture,
        pt_watcher_path,
    )

    # Build edges
    edge_source, edge_state_dict = EdgesBuilder._build_edges(
        graph_builder.graph, position_dict, node_icons, static_architecture, pt_watcher_path
    )

    # Draw edges
    plot.multi_line(
        xs="xs",
        ys="ys",
        source=edge_source,
        line_color="line_color",
        line_width=3,
        line_alpha="line_alpha",
    )

    # Cover edge lines with circles
    plot.scatter(
        x="x",
        y="y",
        size=45 * plot_scaling,
        source=node_source,
        color=BACKGROUND_COLOR_CODE,
        line_alpha=0,
    )

    # Draw nodes as images
    plot.image_url(
        url="url",
        x="x",
        y="y",
        w="w",
        h="h",
        anchor="center",
        source=node_source,
    )

    # Add node labels
    label_source = bkmodel.ColumnDataSource(
        data=dict(
            x=node_x,
            y=[y - 15 * icon_factor * plot_scaling * 0.7 for y in node_y],
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

    # Add legend
    LegendBuilder._add_legend(plot, legend_position, abs(legend_scaling))

    # Add interactive tools
    hover_source = bkmodel.ColumnDataSource(
        data=dict(
            x=node_x,
            y=node_y,
            w=node_width,
            h=node_height,
            name=node_name_list,
            type_class=[
                _string_cleanup(nt.capitalize() if isinstance(nt, str) else nt)
                for nt in node_types_list
            ],
            component_type=[_string_cleanup(nt) for nt in node_om_types_list],
        )
    )

    InteractiveToolsBuilder._add_interactive_tools(plot, hover_source)

    # Save or serve
    if static_architecture:
        HTMLSaver._save_static_html(plot, network_file_path)
    else:
        doc_builder = InteractiveDocumentBuilder(
            plot,
            edge_source,
            hover_source,
            edge_state_dict,
            component_perf,
            refresh_rate,
            pt_watcher_path,
        )

        BokehServerManager._start_server(doc_builder._build_document, port, address)


# ============================================================================
# Color and icon utilities
# ============================================================================


def _get_edge_color(source_icon: str, target_icon: str) -> str:
    """
    Determine edge color based on source and target node types.

    :param source_icon: Icon name of the source node
    :param target_icon: Icon name of the target node

    :return: Color code for the edge
    """
    color_as_target = ICONS_CONFIG.get(target_icon, {}).get("target_color")
    if color_as_target:
        return color_as_target

    color_as_source = ICONS_CONFIG.get(source_icon, {}).get("source_color")
    if color_as_source:
        return color_as_source

    return DEFAULT_COLOR


def _url_to_base64(url: str) -> str:
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


def _string_cleanup(text: str) -> str:
    """
    Clean up text for better readability.
    """
    # In case for list type definition
    if isinstance(text, list):
        text = ", ".join([text[0].capitalize(), text[1].capitalize()])

    # Replace underscore with space
    text = text.replace("_", " ")

    # Add a space after 'DC' if followed immediately by a letter or number
    text = text.replace("DC", "DC ").replace("DC DC", "DC-DC")

    # Add a space after 'H2' and 'PEMFC'
    text = text.replace("H2", "H2 ").replace("PEMFC", "PEMFC ")

    # Add space before a capital letter preceded by a lowercase letter
    text = _add_space_before_caps(text)

    # Remove extra spaces
    text = " ".join(text.split())

    return text


def _add_space_before_caps(text: str) -> str:
    """
    Add space before capital letters preceded by lowercase.
    """
    result = []
    for i, char in enumerate(text):
        if i > 0 and char.isupper() and text[i - 1].islower():
            result.append(" " + char)
        else:
            result.append(char)
    return "".join(result)


def _get_file_name(file_path: str) -> str:
    """
    Extract and format filename from file path.
    """
    file_path = str(file_path)
    filename = file_path.replace("\\", "/").split("/")[-1]

    if filename.endswith(".yml"):
        filename = filename[:-4]
        filename = filename.replace("_", " ").capitalize()
        return f"{filename} powertrain network"

    return filename


# ============================================================================
# Animation utilities
# ============================================================================


def _calculate_callback_interval(refresh_rate: int) -> int:
    """Calculate optimal callback interval based on monitor refresh rate."""
    return int(1000 / refresh_rate)


def _calculate_animation_frames(refresh_rate: int, animation_duration_ms: int = 1000) -> int:
    """Calculate animation frame count for smooth animations."""
    return max(refresh_rate, int(refresh_rate * animation_duration_ms / 1000))


def _create_segmented_edges(edge_x_pos, edge_y_pos, edge_colors, segments_per_edge: int = 10):
    """Break each edge into multiple segments for flowing animation."""
    seg_xs = []
    seg_ys = []
    seg_alphas = []
    seg_colors = []
    edge_ids = []

    for edge_idx, (edge_x, edge_y, color) in enumerate(zip(edge_x_pos, edge_y_pos, edge_colors)):
        sx, ex = edge_x[0], edge_x[1]
        sy, ey = edge_y[0], edge_y[1]

        for seg in range(segments_per_edge):
            t_start = seg / segments_per_edge
            t_end = (seg + 1) / segments_per_edge

            x1 = sx + (ex - sx) * t_start
            y1 = sy + (ey - sy) * t_start
            x2 = sx + (ex - sx) * t_end
            y2 = sy + (ey - sy) * t_end

            seg_xs.append([x1, x2])
            seg_ys.append([y1, y2])
            seg_alphas.append(0.7)
            seg_colors.append(color)
            edge_ids.append(edge_idx)

    return seg_xs, seg_ys, seg_alphas, edge_ids, seg_colors


# ============================================================================
# Data processing utilities
# ============================================================================


def _extract_edge_working_state(df_pt: pd.DataFrame, start: str, end: str) -> list:
    """
    Extract edge working state from PT watcher dataframe.

    :param df_pt: Pandas dataframe of PT watcher extract from the csv file
    :param start: The component at the source side of the connection
    :param end: The component at the target side of the connection

    :return: the edge working state list of for each flight point
    """
    watcher_variables = df_pt.columns.tolist()
    edge_state_start = None
    edge_state_end = None
    keys = ["current", "torque", "fuel"]

    for variable in watcher_variables:
        if edge_state_start is None and start in variable and any(key in variable for key in keys):
            edge_state_start = [state >= 1e-6 for state in df_pt[variable]]

        if edge_state_end is None and end in variable and any(key in variable for key in keys):
            edge_state_end = [state >= 1e-6 for state in df_pt[variable]]

        # Early exit if both found
        if edge_state_start is not None and edge_state_end is not None:
            break

    # Combine results
    if edge_state_start and edge_state_end:
        return [state_s and state_e for state_s, state_e in zip(edge_state_start, edge_state_end)]
    elif edge_state_start:
        return edge_state_start
    else:
        return edge_state_end


def _extract_component_performance(df_pt: pd.DataFrame, name: str, perf_dict: dict) -> dict:
    """
    Extract specific component performance data from PT watcher dataframe.

    :param df_pt: Pandas dataframe of PT watcher extract from the csv file
    :param name: The component name specified in the PT file
    :param perf_dict: The dictionary saves the performance matrice of each component

    :return: the performance dictionary is returned for the next component
    """
    watcher_variables = df_pt.columns.tolist()

    for variable in watcher_variables:
        if name in variable:
            registered_variable = variable.replace(name + " ", "")

            if name not in perf_dict:
                perf_dict[name] = {}

            perf_dict[name][registered_variable] = [round(value, 3) for value in df_pt[variable]]

    return perf_dict


# ============================================================================
# Graph initialization
# ============================================================================


class GraphBuilder:
    """Build and configure the network graph from power train configuration."""

    def __init__(self, power_train_file_path: str):
        self.configurator = FASTGAHEPowerTrainConfigurator()
        self.configurator.load(power_train_file_path)
        self.graph = nx.DiGraph()

    def _build_graph(self) -> tuple:
        """
        Build the complete graph with all nodes and edges.
        """
        names, connections, components_type, components_om_type, icons_name, icons_size = (
            self.configurator.get_network_elements_list()
        )

        propeller_names = []
        node_sizes = {}
        node_types = {}
        node_om_types = {}
        node_icons = {}

        # Add nodes
        for component_name, component_type, om_type, icon_name, icon_size in zip(
            names, components_type, components_om_type, icons_name, icons_size
        ):
            self.graph.add_node(component_name)
            if component_type == "propulsor":
                propeller_names.append(component_name)
            node_sizes[component_name] = icon_size
            node_types[component_name] = component_type
            node_om_types[component_name] = om_type
            node_icons[component_name] = icon_name

        # Add edges
        for connection in connections:
            source = connection[0][0] if isinstance(connection[0], list) else connection[0]
            target = connection[1][0] if isinstance(connection[1], list) else connection[1]
            self.graph.add_edge(source, target)

        return propeller_names, node_sizes, node_types, node_om_types, node_icons

    def _get_hierarchy_layers(self, propeller_names: list, from_propulsor: bool = False) -> dict:
        """
        Compute hierarchy layers for graph layout.

        :param propeller_names: Component name list of the propeller in powertrain
        :param from_propulsor: Set all propulsor component into reference layer of the hierarchy

        :return: The dictionary of the node hierarchy level
        """
        distance_from_energy = self.configurator.get_component_distance(["tank", "battery_pack"])
        node_layer_dict = {}
        max_distance = max(distance_from_energy.values())

        for node_name, distance in distance_from_energy.items():
            if max_distance > distance:
                node_layer_dict[node_name] = max_distance - distance
            else:
                if node_name in propeller_names:
                    node_layer_dict[node_name] = max_distance - distance
                else:
                    from_propulsor = True
                    break

        if from_propulsor:
            distance_from_propulsor = self.configurator.get_component_distance("propulsor")
            for node_name, distance in distance_from_propulsor.items():
                node_layer_dict[node_name] = distance

        return node_layer_dict


# ============================================================================
# Plot creation and configuration
# ============================================================================


class BokehPlotBuilder:
    """Create and configure the Bokeh plot."""

    @staticmethod
    def _create_plot(
        power_train_file: str, position_dict: dict, orientation: str, plot_scaling: float
    ) -> tuple:
        """
        Create Bokeh plot with proper scaling and positioning.

        :param power_train_file: Path to the power train configuration file
        :param position_dict: The component position dictionary obtained from layout generation
        :param orientation: network plot orientation
        :param plot_scaling: Scaling factor for the main powertrain architecture
        """
        x_coords = [coords[0] for coords in position_dict.values()]
        y_coords = [coords[1] for coords in position_dict.values()]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        orientation_params = BokehPlotBuilder._get_orientation_params(orientation)
        x_factor = orientation_params["x_factor"]
        y_factor = orientation_params["y_factor"]
        plot_width_factor = orientation_params["plot_width_factor"]
        icon_factor = orientation_params["icon_factor"]
        icon_width_factor = orientation_params["icon_width_factor"]
        x_offset = orientation_params["x_offset"]
        y_offset = orientation_params["y_offset"]

        normalized_positions = {
            node: (
                ((coord[0] - x_min) / x_range * 550 * x_factor + x_offset),
                ((coord[1] - y_min) / y_range * 550 * y_factor + y_offset),
            )
            for node, coord in position_dict.items()
        }

        plot = bkplot.figure(
            width=int(1200 * plot_scaling * plot_width_factor),
            height=int(900 * plot_scaling),
            x_range=(-50, 600),
            y_range=(-50, 600),
            toolbar_location="above",
            background_fill_color=BACKGROUND_COLOR_CODE,
            title=_get_file_name(power_train_file),
        )

        plot.xgrid.visible = False
        plot.ygrid.visible = False
        plot.xaxis.visible = False
        plot.yaxis.visible = False

        return plot, normalized_positions, icon_factor, icon_width_factor

    @staticmethod
    def _get_orientation_params(orientation: str) -> dict:
        """
        Get orientation-specific parameters.
        """
        if orientation not in ["TB", "BT", "LR", "RL"]:
            orientation = "TB"

        params = {
            "TB": {
                "x_factor": 0.5,
                "y_factor": 1.0,
                "plot_width_factor": 1,
                "icon_factor": 1,
                "icon_width_factor": 0.8,
                "x_offset": 125,
                "y_offset": 0.0,
            },
            "BT": {
                "x_factor": 0.5,
                "y_factor": 1.0,
                "plot_width_factor": 1,
                "icon_factor": 1,
                "icon_width_factor": 0.8,
                "x_offset": 125,
                "y_offset": 0.0,
            },
            "LR": {
                "x_factor": 1.0,
                "y_factor": 0.5,
                "plot_width_factor": 1.25,
                "icon_factor": 1.75,
                "icon_width_factor": 0.6,
                "x_offset": -25,
                "y_offset": 150,
            },
            "RL": {
                "x_factor": 1.0,
                "y_factor": 0.5,
                "plot_width_factor": 1.25,
                "icon_factor": 1.75,
                "icon_width_factor": 0.6,
                "x_offset": -25,
                "y_offset": 150,
            },
        }

        return params.get(orientation, params["TB"])


# ============================================================================
# Node and edge building
# ============================================================================


class NodesBuilder:
    """Build node data structures for Bokeh visualization."""

    @staticmethod
    def _build_nodes(
        graph: nx.DiGraph,
        position_dict: dict,
        node_types: dict,
        node_om_types: dict,
        node_icons: dict,
        icon_width_factor: float,
        node_sizes: dict,
        icon_factor: float,
        plot_scaling: float,
        static_architecture: bool,
        pt_watcher_file_path: str = None,
    ) -> tuple:
        """
        Build complete node data structure.

        :param position_dict: The component position dictionary obtained from layout generation
        :param node_types: Dictionary mapping component names to their component type classes
        :param node_om_types: Dictionary mapping component names to their om types
        :param node_icons: Dictionary mapping component names to their icon name
        :param icon_width_factor: Factor that adjusts the icon width based on plot orientation
        :param node_sizes: Dictionary mapping component names to their icon size
        :param icon_factor: Factor that adjusts the icon size based on plot orientation
        :param plot_scaling: Scaling factor for the main powertrain architecture
        :param static_architecture: True for static HTML, False for interactive server
        :param pt_watcher_file_path: Path to the PT watcher csv file

        :return: Node properties and node Bokeh dataSource
        """
        node_name_list = list(graph.nodes())
        node_x = []
        node_y = []
        node_width = []
        node_height = []
        node_types_list = []
        node_om_types_list = []
        component_perf = {}

        df_pt = pd.read_csv(pt_watcher_file_path) if pt_watcher_file_path else None

        for node in node_name_list:
            node_x.append(position_dict[node][0])
            node_y.append(position_dict[node][1])
            node_height.append(node_sizes[node] * icon_factor * plot_scaling)
            node_width.append(node_sizes[node] * icon_factor * icon_width_factor * plot_scaling)
            node_types_list.append(node_types[node])
            node_om_types_list.append(node_om_types[node])

            if df_pt is not None:
                component_perf = _extract_component_performance(df_pt, node, component_perf)

        node_image_urls = NodesBuilder._get_node_image_urls(
            node_name_list, node_icons, static_architecture
        )

        node_source = bkmodel.ColumnDataSource(
            data=dict(x=node_x, y=node_y, url=node_image_urls, w=node_width, h=node_height)
        )

        return (
            node_source,
            node_x,
            node_y,
            node_width,
            node_height,
            node_name_list,
            node_types_list,
            node_om_types_list,
            component_perf,
        )

    @staticmethod
    def _get_node_image_urls(
        node_name_list: list, node_icons: dict, static_architecture: bool
    ) -> list:
        """Generate image URLs for nodes."""
        urls = []
        for node in node_name_list:
            icon_path = ICONS_CONFIG[node_icons[node]]["icon_path"]
            file_url = "file://" + str(Path(icon_path).resolve())

            if static_architecture:
                urls.append(file_url)
            else:
                urls.append(_url_to_base64(file_url))

        return urls


class EdgesBuilder:
    """Build edge data structures for Bokeh visualization."""

    @staticmethod
    def _build_edges(
        graph: nx.DiGraph,
        position_dict: dict,
        node_icons: dict,
        static_architecture: bool,
        pt_watcher_file_path: str = None,
    ) -> tuple:
        """
        Build complete edge data structure.

        :param position_dict: The component position dictionary obtained from layout generation
        :param node_icons: Dictionary mapping component names to their icon name
        :param static_architecture: True for static HTML, False for interactive server
        :param pt_watcher_file_path: Path to the PT watcher csv file

        :return: Edge properties and edge Bokeh dataSource
        """
        edge_x_pos = []
        edge_y_pos = []
        edge_colors = []
        edge_state = {}

        df_pt = pd.read_csv(pt_watcher_file_path) if pt_watcher_file_path else None

        for index, (start, end) in enumerate(list(graph.edges())):
            edge_x_pos.append([position_dict[start][0], position_dict[end][0]])
            edge_y_pos.append([position_dict[start][1], position_dict[end][1]])

            source_icon = node_icons[start]
            target_icon = node_icons[end]
            edge_color = _get_edge_color(source_icon, target_icon)
            edge_colors.append(edge_color)

            if df_pt is not None:
                edge_state[index] = _extract_edge_working_state(df_pt, start, end)

        if static_architecture:
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
                    line_color=seg_colors,
                    line_alpha=seg_alphas,
                    edge_id=edge_ids,
                )
            )

        return edge_source, edge_state


# ============================================================================
# Legend building
# ============================================================================


class LegendBuilder:
    """Build and add legend to the plot."""

    @staticmethod
    def _add_legend(plot, legend_position: str, legend_scaling: float = 1.0):
        """
        Add color legend to the plot.

        :param legend_position: Legend position ('TR', 'TL', 'BR', 'BL', etc.)
        :param legend_scaling: Scaling factor for the legend size
        """
        color_icon_urls = [
            "file://" + str(Path(COLOR_ICON_CONFIG[key]).resolve())
            for key in COLOR_ICON_CONFIG.keys()
        ]

        if len(legend_position) != 2:
            legend_position = "TR"

        x_start = LegendBuilder._get_x_position(legend_position)
        y_start = LegendBuilder._get_y_position(legend_position)

        legend_items = [
            (0, "Fuel Flow"),
            (1, "Mechanical Power"),
            (2, "Electrical Current"),
        ]

        legend_item_height = int(22 * legend_scaling)
        legend_item_start_y = y_start - int(25 * legend_scaling)

        for i, (icon_idx, description) in enumerate(legend_items):
            y_position = legend_item_start_y - (i * legend_item_height)
            icon_url = _url_to_base64(color_icon_urls[icon_idx])

            icon_source = bkmodel.ColumnDataSource(
                data=dict(
                    x=[x_start + int(10 * legend_scaling)],
                    y=[y_position],
                    url=[icon_url],
                )
            )

            plot.image_url(
                url="url",
                x="x",
                y="y",
                w=9 * legend_scaling,
                h=12 * legend_scaling,
                anchor="center",
                source=icon_source,
            )

            label_source = bkmodel.ColumnDataSource(
                data=dict(x=[x_start + 25], y=[y_position], text=[description])
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

    @staticmethod
    def _get_x_position(legend_position: str) -> int:
        """Get x position based on legend position code."""
        if "R" in legend_position:
            return 500
        elif "L" in legend_position:
            return -50
        elif "C" in legend_position:
            return 225
        return 500

    @staticmethod
    def _get_y_position(legend_position: str) -> int:
        """Get y position based on legend position code."""
        if "T" in legend_position:
            return 600
        elif "B" in legend_position:
            return 50
        elif "M" in legend_position:
            return 325
        return 600


# ============================================================================
# Interactive tools
# ============================================================================


class InteractiveToolsBuilder:
    """Build interactive hover and selection tools."""

    @staticmethod
    def _add_interactive_tools(
        plot,
        hover_source,
    ):
        """Add hover and tap tools to the plot."""

        hover = bkmodel.HoverTool(
            tooltips=[
                ("Name", "@name"),
                ("Type class", "@type_class"),
                ("Component type", "@component_type"),
            ]
        )

        scatter_glyph = plot.scatter(
            x="x",
            y="y",
            size=55,
            source=hover_source,
            fill_alpha=0,
            line_alpha=0,
            hover_fill_alpha=0.1,
            hover_line_alpha=0.3,
        )

        tap_tool = bkmodel.TapTool(renderers=[scatter_glyph])
        plot.add_tools(hover, tap_tool, bkmodel.BoxSelectTool())

        return scatter_glyph


# ============================================================================
# HTML saving
# ============================================================================


class HTMLSaver:
    """Save Bokeh plots as static HTML with embedded images."""

    @staticmethod
    def _save_static_html(plot, file_path: str):
        """Save the network plot as static HTML with embedded base64 images."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        bkplot.output_file(str(file_path))
        bkplot.save(plot)

        html_content = HTMLSaver._read_html(file_path)
        html_content = HTMLSaver._replace_file_urls_with_base64(html_content)
        HTMLSaver._write_html(file_path, html_content)

        print(f"Static HTML saved to: {file_path}")

    @staticmethod
    def _read_html(file_path: Path) -> str:
        """Read HTML file content."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _write_html(file_path: Path, content: str):
        """Write HTML file content."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _replace_file_urls_with_base64(html_content: str) -> str:
        """Replace all file:// URLs with base64 data URIs."""
        try:
            parts = html_content.split("file://")
            result = [parts[0]]

            for part in parts[1:]:
                end_chars = ['"', ",", "]", "}"]
                end_idx = len(part)

                for char in end_chars:
                    idx = part.find(char)
                    if idx != -1 and idx < end_idx:
                        end_idx = idx

                url = "file://" + part[:end_idx]
                converted = _url_to_base64(url)
                result.append(converted + part[end_idx:])

            return "".join(result)
        except Exception as e:
            print(f"Error processing URLs: {e}")
            return html_content


# ============================================================================
# Server management
# ============================================================================


class BokehServerManager:
    """Manage Bokeh server lifecycle and document setup."""

    @staticmethod
    def _start_server(make_document, port: int, address: str):
        """Start and run a Bokeh Server with the provided document maker."""
        logging.getLogger("bokeh").setLevel(logging.WARNING)
        logging.getLogger("tornado").setLevel(logging.WARNING)

        def make_document_with_tracking(doc):
            make_document(doc)

            def on_destroy(session_context):
                IOLoop.current().stop()

            doc.on_session_destroyed(on_destroy)

        server = Server(
            {"/": make_document_with_tracking},
            port=port,
            address=address,
            num_procs=1,
        )

        server.start()

        def _open_browser():
            webbrowser.open(f"http://{address}:{port}/")

        IOLoop.current().call_later(0.1, _open_browser)
        server.io_loop.start()


# ============================================================================
# Interactive document builder
# ============================================================================


class InteractiveDocumentBuilder:
    """Build document for interactive server mode with animation."""

    def __init__(
        self,
        plot,
        edge_source,
        hover_source,
        edge_state_dict,
        component_perf_dict,
        refresh_rate: int = 60,
        pt_watcher_file_path: str = None,
    ):
        self.plot = plot
        self.edge_source = edge_source
        self.hover_source = hover_source
        self.edge_state_dict = edge_state_dict
        self.component_perf_dict = component_perf_dict
        self.pt_watcher_file_path = pt_watcher_file_path
        self.refresh_rate = refresh_rate
        self.flight_point_slider = None
        self.table_source = None

    def _build_document(self, doc):
        """Build interactive document with sliders and tables."""
        if self.pt_watcher_file_path:
            last_point = len(self.edge_state_dict[0])
            self._setup_flight_point_controls(last_point)
            doc.add_root(row(self.plot, column(self.flight_point_slider, self._create_table())))
        else:
            doc.add_root(self.plot)

        animation_counter = 0
        animation_frames = _calculate_animation_frames(
            refresh_rate=self.refresh_rate, animation_duration_ms=1000
        )

        def _update_animation():
            """Update animation state for flowing edges."""
            nonlocal animation_counter
            animation_counter = (animation_counter + 1) % animation_frames
            progress = animation_counter / animation_frames

            new_alphas = self._calculate_edge_alphas(progress)
            self.edge_source.patch({"line_alpha": [(slice(len(new_alphas)), new_alphas)]})

        callback_interval = _calculate_callback_interval(refresh_rate=self.refresh_rate)
        doc.add_periodic_callback(_update_animation, callback_interval)

    def _setup_flight_point_controls(self, last_point: int):
        """Setup flight point slider and callbacks."""
        self.flight_point_slider = bkmodel.Slider(
            start=1, end=last_point, value=1, step=1, title="Flight Point"
        )
        self.table_source = bkmodel.ColumnDataSource(data=dict(property=[], value=[]))
        self.hover_source.selected.on_change("indices", self._on_node_click)
        self.flight_point_slider.on_change("value", self._on_slider_change)

    def _create_table(self):
        """Create data table for component properties."""
        columns = [
            bkmodel.TableColumn(field="property", title="Property", width=210),
            bkmodel.TableColumn(field="value", title="Value", width=60),
        ]
        return bkmodel.DataTable(
            source=self.table_source, columns=columns, width=280, height=600, index_position=None
        )

    def _on_node_click(self, attr, old, new):
        """Handle node selection."""
        if new:
            node_name = self.hover_source.data["name"][new[0]]
            flight_point = int(self.flight_point_slider.value) - 1
            self._update_table_for_node(node_name, flight_point)
        else:
            self.table_source.data = dict(property=[], value=[])

    def _on_slider_change(self, attr, old, new):
        """Handle slider value change."""
        if self.hover_source.selected.indices:
            self._on_node_click(None, None, self.hover_source.selected.indices)

    def _update_table_for_node(self, node_name: str, flight_point: int):
        """Update table with node performance data."""
        properties, values = [], []

        if node_name in self.component_perf_dict:
            for key, value_list in self.component_perf_dict[node_name].items():
                properties.append(key)
                values.append(
                    str(value_list[flight_point]) if flight_point < len(value_list) else "N/A"
                )

        self.table_source.data = dict(property=properties, value=values)

    def _calculate_edge_alphas(self, progress: float) -> list:
        """Calculate alpha values for edge animation."""
        new_alphas = []
        num_segments = len(self.edge_source.data["xs"])
        edge_ids = self.edge_source.data.get("edge_id", [])
        num_edges = max(edge_ids) + 1 if edge_ids else 1
        segments_per_edge = num_segments // num_edges if num_edges > 0 else 1

        for i in range(num_segments):
            edge_idx = edge_ids[i] if i < len(edge_ids) else 0
            segment_idx = i % segments_per_edge if segments_per_edge > 0 else 0
            seg_position = segment_idx / segments_per_edge if segments_per_edge > 0 else 0
            wave_pos = (-progress + edge_idx * 0.1) % 1.0
            distance = abs(seg_position - wave_pos)

            if self.pt_watcher_file_path:
                flight_point = int(self.flight_point_slider.value) - 1
                is_active = (
                    self.edge_state_dict.get(edge_idx, [False])[flight_point]
                    if flight_point < len(self.edge_state_dict.get(edge_idx, []))
                    else False
                )
                alpha = (
                    0.3 + 0.7 * (1.0 - (distance / 0.3) ** 2.0)
                    if distance < 0.3 and is_active
                    else 0.1
                )
            else:
                alpha = 0.3 + 0.7 * (1.0 - (distance / 0.3) ** 2.0) if distance < 0.3 else 0.3

            new_alphas.append(alpha)

        return new_alphas
