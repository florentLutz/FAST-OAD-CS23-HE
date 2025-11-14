import os
import os.path as pth
from pathlib import Path

import pygraphviz as pgv
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    TapTool,
    BoxSelectTool,
    LabelSet,
)
from bokeh.layouts import column
from bokeh.server.server import Server

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from . import icons

# Image URLs for graph nodes (single image or list of images for animation)
icons_dict = {
    "battery": pth.join(icons.__path__[0], "battery.png"),
    "bus_bar": pth.join(icons.__path__[0], "bus_bar.png"),
    "cable": pth.join(icons.__path__[0], "cable.png"),
    "e_motor": pth.join(icons.__path__[0], "e_motor.png"),
    "generator": pth.join(icons.__path__[0], "generator.png"),
    "ice": pth.join(icons.__path__[0], "ice.png"),
    "switch": pth.join(icons.__path__[0], "switch.png"),
    "propeller": [
        pth.join(icons.__path__[0], "propeller_1.png"),
        pth.join(icons.__path__[0], "propeller_2.png"),
        pth.join(icons.__path__[0], "propeller_3.png"),
    ],  # Animated propeller
    "splitter": pth.join(icons.__path__[0], "splitter.png"),
    "rectifier": pth.join(icons.__path__[0], "AC_DC.png"),
    "dc_converter": pth.join(icons.__path__[0], "DC_DC.png"),
    "inverter": pth.join(icons.__path__[0], "DC_AC.png"),
    "fuel_tank": pth.join(icons.__path__[0], "fuel_tank.png"),
    "fuel_system": pth.join(icons.__path__[0], "fuel_system.png"),
    "turbine": pth.join(icons.__path__[0], "turbine.png"),
    "gearbox": pth.join(icons.__path__[0], "gears.png"),
}


def create_network_plot(power_train_file_path: str, layout_prog: str = "dot"):
    """
    Create an interactive network visualization of a power train using Bokeh Server with animated edges.

    Args:
        power_train_file_path: Path to the power train configuration file
        layout_prog: Graphviz layout program ('dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi')

    Returns:
        plot: Bokeh figure object
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
                ((p[1] - y_min) / y_range * scale * 0.8 + scale * 0.1),
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
    node_sizes_list = [node_sizes[node] * 2 for node in node_indices]
    node_types_list = [node_types[node] for node in node_indices]

    # Get image paths and convert to Base64 URLs (cross-platform)
    import base64

    node_image_urls = []
    node_image_sequences = {}  # Store animation sequences for nodes

    for node in node_indices:
        icon_name = node_icons[node]
        icon_path = icons_dict[icon_name]

        # Check if this is an animated icon (list of paths) or static (single path)
        if isinstance(icon_path, list):
            # Animated icon
            animation_frames = []
            for frame_path in icon_path:
                try:
                    icon_file = Path(frame_path)
                    if not icon_file.exists():
                        raise FileNotFoundError(f"Icon file not found: {frame_path}")

                    with open(icon_file, "rb") as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        ext = icon_file.suffix.lower()
                        mime_types = {
                            ".png": "image/png",
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".gif": "image/gif",
                            ".svg": "image/svg+xml",
                        }
                        mime_type = mime_types.get(ext, "image/png")
                        url = f"data:{mime_type};base64,{img_data}"
                        animation_frames.append(url)
                except Exception as e:
                    print(f"✗ ERROR loading animation frame for {icon_name}: {e}")

            if animation_frames:
                node_image_sequences[node] = animation_frames
                node_image_urls.append(animation_frames[0])  # Start with first frame
                print(f"✓ Loaded animated icon: {icon_name} ({len(animation_frames)} frames)")
            else:
                node_image_urls.append(
                    "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzk5OSIvPjwvc3ZnPg=="
                )
        else:
            # Static icon
            try:
                icon_file = Path(icon_path)
                if not icon_file.exists():
                    raise FileNotFoundError(f"Icon file not found: {icon_path}")

                with open(icon_file, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    ext = icon_file.suffix.lower()
                    mime_types = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".gif": "image/gif",
                        ".svg": "image/svg+xml",
                    }
                    mime_type = mime_types.get(ext, "image/png")
                    url = f"data:{mime_type};base64,{img_data}"
                    node_image_urls.append(url)
                    print(f"✓ Loaded icon: {icon_name}")
            except FileNotFoundError as e:
                print(f"✗ FILE NOT FOUND: {icon_name} - {e}")
                node_image_urls.append(
                    "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzk5OSIvPjwvc3ZnPg=="
                )
            except Exception as e:
                print(f"✗ ERROR loading {icon_name}: {type(e).__name__}: {e}")
                node_image_urls.append(
                    "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzk5OSIvPjwvc3ZnPg=="
                )

    # Prepare edge data for animated multi_line
    edge_start_x = []
    edge_start_y = []
    edge_end_x = []
    edge_end_y = []

    for edge in G.edges():
        start, end = edge
        start_x, start_y = pos[start]
        end_x, end_y = pos[end]
        edge_start_x.append(start_x)
        edge_start_y.append(start_y)
        edge_end_x.append(end_x)
        edge_end_y.append(end_y)

    # Create edge data source with line dash pattern
    edge_source = ColumnDataSource(
        data=dict(
            xs=[[sx, ex] for sx, ex in zip(edge_start_x, edge_end_x)],
            ys=[[sy, ey] for sy, ey in zip(edge_start_y, edge_end_y)],
            colors=["#4472C4"] * len(edge_start_x),
        )
    )

    # Store original edge data for animation
    edge_source.data["edge_start_x"] = edge_start_x
    edge_source.data["edge_start_y"] = edge_start_y
    edge_source.data["edge_end_x"] = edge_end_x
    edge_source.data["edge_end_y"] = edge_end_y

    # Draw animated edges
    plot.multi_line(
        xs="xs",
        ys="ys",
        source=edge_source,
        line_color="colors",
        line_width=2,
        line_alpha=0.7,
        line_dash="dotted",
    )

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

    return plot, edge_source, node_source, node_image_sequences


def power_train_network_viewer_hv_server(
    power_train_file_path: str,
    port: int = 5006,
    address: str = "localhost",
    layout_prog: str = "dot",
):
    """
    Start a Bokeh Server for the power train network visualization.

    Args:
        power_train_file_path: Path to the power train configuration file
        port: Port to run the server on (default: 5006)
        address: Server address (default: localhost)
        layout_prog: Graphviz layout program ('dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi')
    """

    def make_document(doc):
        plot, edge_source, node_source, node_image_sequences = create_network_plot(
            power_train_file_path, layout_prog
        )
        animation_counter = [0]

        doc.add_root(column(plot))

        # RGB gradient colors for animation
        gradient_colors = [
            "#FF0000",  # Red
            "#FF7F00",  # Orange
            "#FFFF00",  # Yellow
            "#00FF00",  # Green
            "#0000FF",  # Blue
            "#4B0082",  # Indigo
            "#9400D3",  # Violet
        ]

        # Add periodic callback for animation
        def update():
            animation_counter[0] = (animation_counter[0] + 1) % len(gradient_colors)

            # Get current color from gradient
            current_color = gradient_colors[animation_counter[0]]

            # Update all edge colors to current gradient color
            new_colors = [current_color] * len(edge_source.data["xs"])

            edge_source.patch({"colors": [(slice(len(new_colors)), new_colors)]})

            # Update animated node icons
            if node_image_sequences:
                frame_index = (animation_counter[0] // 2) % 10  # Slower animation for nodes

                new_urls = []
                for i, node in enumerate(node_source.data["name"]):
                    if node in node_image_sequences:
                        seq = node_image_sequences[node]
                        frame_idx = frame_index % len(seq)
                        new_urls.append(seq[frame_idx])
                    else:
                        new_urls.append(node_source.data["url"][i])

                node_source.patch({"url": [(slice(len(new_urls)), new_urls)]})

        doc.add_periodic_callback(update, 100)  # Update every 100ms for smooth color cycling

    server = Server(
        {"/": make_document},
        port=port,
        address=address,
        num_procs=1,
    )

    print(f"\n{'='*60}")
    print(f"Bokeh Server started!")
    print(f"Open your browser and go to:")
    print(f"  http://{address}:{port}/")
    print(f"{'='*60}\n")

    server.start()
    server.io_loop.start()


if __name__ == "__main__":
    # Example usage
    power_train_file_path = "path/to/your/power_train_config.xml"
    power_train_network_viewer_hv_server(
        power_train_file_path=power_train_file_path,
        port=5006,
        address="localhost",
        layout_prog="dot",
    )
