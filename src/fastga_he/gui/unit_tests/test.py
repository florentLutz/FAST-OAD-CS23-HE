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

from fastga_he.gui import icons

DATA_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "data")
RESULTS_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "results")

# Image URLs for graph nodes
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
        pth.join(icons.__path__[0], "propeller_4.png"),
    ],
    "splitter": pth.join(icons.__path__[0], "splitter.png"),
    "rectifier": pth.join(icons.__path__[0], "AC_DC.png"),
    "dc_converter": pth.join(icons.__path__[0], "DC_DC.png"),
    "inverter": pth.join(icons.__path__[0], "DC_AC.png"),
    "fuel_tank": pth.join(icons.__path__[0], "fuel_tank.png"),
    "fuel_system": pth.join(icons.__path__[0], "fuel_system.png"),
    "turbine": pth.join(icons.__path__[0], "turbine.png"),
    "gearbox": pth.join(icons.__path__[0], "gears.png"),
}


def create_segmented_edges(edge_start_x, edge_start_y, edge_end_x, edge_end_y,
                           segments_per_edge=10):
    """
    Break each edge into multiple segments for flowing animation.

    Args:
        edge_start_x, edge_start_y: Lists of edge starting coordinates
        edge_end_x, edge_end_y: Lists of edge ending coordinates
        segments_per_edge: Number of segments to divide each edge into

    Returns:
        Lists of segment endpoints and metadata for animation
    """
    seg_xs = []  # List of [x1, x2] for each segment
    seg_ys = []  # List of [y1, y2] for each segment
    seg_alphas = []
    edge_ids = []  # Track which edge each segment belongs to

    for edge_idx, (sx, sy, ex, ey) in enumerate(
            zip(edge_start_x, edge_start_y, edge_end_x, edge_end_y)):
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
            edge_ids.append(edge_idx)

    return seg_xs, seg_ys, seg_alphas, edge_ids

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
        components_om_type,
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

                    with open(icon_file, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        ext = icon_file.suffix.lower()
                        mime_types = {
                            '.png': 'image/png',
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.gif': 'image/gif',
                            '.svg': 'image/svg+xml',
                        }
                        mime_type = mime_types.get(ext, 'image/png')
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
                    "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzk5OSIvPjwvc3ZnPg==")
        else:
            # Static icon
            try:
                icon_file = Path(icon_path)
                if not icon_file.exists():
                    raise FileNotFoundError(f"Icon file not found: {icon_path}")

                with open(icon_file, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode()
                    ext = icon_file.suffix.lower()
                    mime_types = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif',
                        '.svg': 'image/svg+xml',
                    }
                    mime_type = mime_types.get(ext, 'image/png')
                    url = f"data:{mime_type};base64,{img_data}"
                    node_image_urls.append(url)
                    print(f"✓ Loaded icon: {icon_name}")
            except FileNotFoundError as e:
                print(f"✗ FILE NOT FOUND: {icon_name} - {e}")
                node_image_urls.append(
                    "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzk5OSIvPjwvc3ZnPg==")
            except Exception as e:
                print(f"✗ ERROR loading {icon_name}: {type(e).__name__}: {e}")
                node_image_urls.append(
                    "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzk5OSIvPjwvc3ZnPg==")

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

    # Create segmented edges for flowing animation
    seg_xs, seg_ys, seg_alphas, edge_ids = create_segmented_edges(
        edge_start_x, edge_start_y, edge_end_x, edge_end_y, segments_per_edge=60
    )

    edge_source = ColumnDataSource(
        data=dict(
            xs=seg_xs,
            ys=seg_ys,
            colors=["#4472C4"] * len(seg_xs),
            line_alpha=seg_alphas,
            edge_id=edge_ids,  # This is the missing definition
        )
    )

    # Draw animated edges
    plot.multi_line(
        xs="xs",
        ys="ys",
        source=edge_source,
        line_color="colors",
        line_width=3,
        line_alpha="line_alpha",
        line_dash="solid",
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

    # Draw nodes as images and store renderer for hover tool
    image_renderer = plot.image_url(
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
        ],
        renderers=[image_renderer],  # Explicitly reference the image renderer
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
            power_train_file_path, layout_prog)
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
            animation_counter[0] = (animation_counter[0] + 1) % 100
            progress = animation_counter[0] / 100.0  # 0 to 1

            num_segments = len(edge_source.data['xs'])
            num_edges = max(edge_source.data['edge_id']) + 1 if edge_source.data['edge_id'] else 1
            segments_per_edge = num_segments // num_edges

            new_alphas = []

            for i in range(num_segments):
                edge_idx = edge_source.data['edge_id'][i]
                segment_idx = i % segments_per_edge

                # Normalize segment position within edge (0 to 1)
                seg_position = segment_idx / segments_per_edge

                # Calculate wave position for this segment
                # The wave moves from 0 to 1 as progress goes from 0 to 1
                wave_pos = (-progress + edge_idx * 0.1) % 1.0 # Offset each edge slightly

                # Distance from segment to wave center
                distance = abs(seg_position - wave_pos)

                # Use gaussian-like falloff for smooth wave
                if distance < 0.3:  # Wave width
                    alpha = 0.3 + 0.7 * (1 - (distance / 0.3) ** 2)
                else:
                    alpha = 0.3

                new_alphas.append(alpha)

            edge_source.patch({
                'line_alpha': [(slice(len(new_alphas)), new_alphas)]
            })

            # Update animated node icons
            if node_image_sequences:
                frame_index = (animation_counter[0] // 2) % 5  # Slower animation for nodes

                new_urls = []
                for i, node in enumerate(node_source.data['name']):
                    if node in node_image_sequences:
                        seq = node_image_sequences[node]
                        frame_idx = frame_index % len(seq)
                        new_urls.append(seq[frame_idx])
                    else:
                        new_urls.append(node_source.data['url'][i])

                node_source.patch({
                    'url': [(slice(len(new_urls)), new_urls)]
                })

        doc.add_periodic_callback(update,
                                  50)  # Update every 50ms for smooth motion for smooth color
        # cycling

    server = Server(
        {"/": make_document},
        port=port,
        address=address,
        num_procs=1,
    )

    print(f"\n{'=' * 60}")
    print(f"Bokeh Server started!")
    print(f"Open your browser and go to:")
    print(f"  http://{address}:{port}/")
    print(f"{'=' * 60}\n")

    server.start()
    server.io_loop.start()


if __name__ == "__main__":
    # Example usage
    power_train_file_path = os.path.join(DATA_FOLDER_PATH, "simple_assembly_tri_prop_two_chainz.yml")
    power_train_network_viewer_hv_server(
        power_train_file_path=power_train_file_path,
        layout_prog="dot",
    )