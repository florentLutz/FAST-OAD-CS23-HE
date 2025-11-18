import os
import os.path as pth
from pathlib import Path
import re
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



def get_monitor_refresh_rate():
    """
    Detect monitor refresh rate from system settings.

    Returns:
        refresh_rate (int): Monitor refresh rate in Hz (default 60 if detection fails)
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
                    timeout=5
                )
                lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
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
                        ["powershell", "-Command",
                         "Get-WmiObject -Namespace root\\cimv2 -Class Win32_VideoController | Select-Object CurrentRefreshRate"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    lines = [line.strip() for line in result.stdout.strip().split('\n') if
                             line.strip() and line.strip().isdigit()]
                    if lines:
                        rate = int(lines[0])
                        if rate > 0:
                            print(f"✓ Detected monitor refresh rate: {rate} Hz")
                            return rate
                except Exception as e2:
                    print(f"✗ PowerShell method also failed: {e2}")
            except Exception as e:
                print(f"✗ Error detecting refresh rate: {e}")

        elif system == "Darwin":  # macOS
            try:
                import subprocess
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "120 Hz" in result.stdout:
                    print("✓ Detected monitor refresh rate: 120 Hz")
                    return 120
                elif "144 Hz" in result.stdout:
                    print("✓ Detected monitor refresh rate: 144 Hz")
                    return 144
                else:
                    print("✓ Detected monitor refresh rate: 60 Hz (default)")
                    return 60
            except Exception as e:
                print(f"✗ Error detecting refresh rate: {e}")

        elif system == "Linux":
            try:
                import subprocess
                result = subprocess.run(
                    ["xrandr"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Parse xrandr output for refresh rate
                for line in result.stdout.split('\n'):
                    if '*' in line:  # Current mode
                        parts = line.split()
                        for part in parts:
                            if '+' in part:
                                rate = float(part.replace('+', ''))
                                print(f"✓ Detected monitor refresh rate: {int(rate)} Hz")
                                return int(rate)
            except Exception as e:
                print(f"✗ Error detecting refresh rate: {e}")

    except Exception as e:
        print(f"✗ Unexpected error detecting refresh rate: {e}")

    print("⚠ Could not detect refresh rate, using default: 60 Hz")
    return 60


def calculate_optimal_callback_interval(refresh_rate, target_fps=None):
    """
    Calculate optimal callback interval based on monitor refresh rate.

    Args:
        refresh_rate (int): Monitor refresh rate in Hz
        target_fps (int): Target animation FPS (default: match refresh rate)

    Returns:
        callback_interval (int): Milliseconds between callbacks
    """
    if target_fps is None:
        target_fps = refresh_rate

    # Ensure target FPS doesn't exceed refresh rate
    target_fps = min(target_fps, refresh_rate)

    callback_interval = int(1000 / target_fps)
    print(f"✓ Animation FPS: {target_fps}, Callback interval: {callback_interval}ms")
    return callback_interval


def calculate_animation_frames(refresh_rate, animation_duration_ms=1000):
    """
    Calculate animation frame count for smooth animations synchronized to refresh rate.

    Args:
        refresh_rate (int): Monitor refresh rate in Hz
        animation_duration_ms (int): Desired animation duration in milliseconds

    Returns:
        frames (int): Number of frames for smooth animation
    """
    # Calculate frames needed for smooth animation
    frames = max(refresh_rate, int(refresh_rate * animation_duration_ms / 1000))
    print(
        f"✓ Animation will use {frames} frames over {animation_duration_ms}ms at {refresh_rate}Hz")
    return frames


def create_rotated_svg(base64_url, rotation_degrees):
    """
    Wrap a base64 image in an SVG with rotation transform without cropping.

    Args:
        base64_url: Data URL of the image (e.g., "data:image/png;base64,...")
        rotation_degrees: Rotation angle in degrees (counter-clockwise)

    Returns:
        SVG data URL with rotated image
    """
    import base64

    # Use a larger SVG canvas to prevent clipping during rotation
    # The diagonal of a 100x100 square is ~141, so use 150x150 to be safe
    svg_template = f'''<svg width="150" height="150" viewBox="0 0 150 150" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
        <g transform="translate(75 75) rotate({rotation_degrees}) translate(-50 -50)">
            <image href="{base64_url}" x="0" y="0" width="100" height="100"/>
        </g>
    </svg>'''

    svg_b64 = base64.b64encode(svg_template.encode()).decode()
    return f"data:image/svg+xml;base64,{svg_b64}"


def create_segmented_edges(edge_start_x, edge_start_y, edge_end_x, edge_end_y, edge_colors,
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
    seg_colors = []
    edge_ids = []  # Track which edge each segment belongs to


    for edge_idx, (sx, sy, ex, ey, color) in enumerate(
            zip(edge_start_x, edge_start_y, edge_end_x, edge_end_y, edge_colors)):
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

def get_edge_color(source_icon, target_icon):
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

def create_network_plot(power_train_file_path: str, layout_prog: str = "dot",orientation: str = "TB"):
    """
    Create an interactive network visualization of a power train using Bokeh Server with animated edges.

    Args:
        power_train_file_path: Path to the power train configuration file
        layout_prog: Graphviz layout program ('dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi')

    Returns:
        plot: Bokeh figure object
    """
    # Create AGraph (PyGraphviz) object
    G = pgv.AGraph(directed=True, rankdir=orientation)

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
        G.add_node(component_name)
        node_sizes[component_name] = icon_size
        node_types[component_name] = component_type
        node_om_types[component_name] = om_type
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
                ((p[1] - y_min) / y_range * scale * y_factor + y_orientation_offset),
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
    node_sizes_list = [node_sizes[node] * icon_factor for node in node_indices]
    node_types_list = [node_types[node] for node in node_indices]
    node_om_types_list = [node_om_types[node] for node in node_indices]

    color_icon_urls = [
        "file://" + str(Path(color_icon_dict[color_icon]).resolve())
        for color_icon in ["fuel", "mechanical", "electricity"]
    ]

    # Get image paths and convert to Base64 URLs (cross-platform)
    import base64
    node_image_urls = []
    node_image_sequences = {}  # Store animation sequences for nodes
    propeller_rotation_sequences = {}
    rotation_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]  # 12 frames

    for node in node_indices:
        icon_name = node_icons[node]
        icon_path = icons_dict[icon_name][0]

        base_url = None  # Store the loaded image URL

        # Check if this is an animated icon (list of paths) or static (single path)
        if isinstance(icon_path, list) and icon_name != "propeller":
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
                base_url = animation_frames[0]  # Start with first frame
                node_image_urls.append(base_url)
                print(f"✓ Loaded animated icon: {icon_name} ({len(animation_frames)} frames)")
            else:
                base_url = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzk5OSIvPjwvc3ZnPg=="
                node_image_urls.append(base_url)
        else:
            # Static icon (including propellers)
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
                    base_url = f"data:{mime_type};base64,{img_data}"
                    node_image_urls.append(base_url)
                    print(f"✓ Loaded icon: {icon_name}")
            except FileNotFoundError as e:
                print(f"✗ FILE NOT FOUND: {icon_name} - {e}")
                base_url = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzk5OSIvPjwvc3ZnPg=="
                node_image_urls.append(base_url)
            except Exception as e:
                print(f"✗ ERROR loading {icon_name}: {type(e).__name__}: {e}")
                base_url = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iIzk5OSIvPjwvc3ZnPg=="
                node_image_urls.append(base_url)

        # NOW create propeller rotation frames AFTER loading the base image
        if icon_name == "propeller" and base_url:
            rotation_frames = []
            for angle in rotation_angles:
                rotated_url = create_rotated_svg(base_url, angle)
                rotation_frames.append(rotated_url)

            propeller_rotation_sequences[node] = rotation_frames
            # Set initial image to first frame
            node_image_urls[-1] = rotation_frames[0]  # Update the last appended URL
            print(f"✓ Created rotation animation for propeller: {node}")

    # Prepare edge data for animated multi_line
    edge_start_x = []
    edge_start_y = []
    edge_end_x = []
    edge_end_y = []
    edge_colors = []

    for edge in G.edges():
        start, end = edge
        edge_start_x.append(pos[start][0])
        edge_start_y.append(pos[start][1])
        edge_end_x.append(pos[end][0])
        edge_end_y.append(pos[end][1])

        # Determine edge color based on connected nodes
        source_icon = node_icons[start]
        target_icon = node_icons[end]
        edge_color = get_edge_color(source_icon, target_icon)
        edge_colors.append(edge_color)

    # Create segmented edges for flowing animation
    seg_xs, seg_ys, seg_alphas, edge_ids, seg_colors = create_segmented_edges(
        edge_start_x, edge_start_y, edge_end_x, edge_end_y, edge_colors, segments_per_edge=30
    )

    edge_source = ColumnDataSource(
        data=dict(
            xs=seg_xs,
            ys=seg_ys,
            colors=seg_colors,
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
            w=[s * icon_width_factor for s in node_sizes_list],
            h=[s for s in node_sizes_list],
            name=node_indices,
            type=node_types_list,
        )
    )

    plot.scatter(
        x="x",
        y="y",
        size=45,  # Adjust size to match your icon size
        source=node_source,
        color="#bebebe",
        line_alpha=0,
    )

    # Draw nodes as images and store renderer for hover tool
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

    # clean up type class and component type for hove info list
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
            type=cleaned_node_types,
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

    return plot, edge_source, node_source, node_image_sequences, propeller_rotation_sequences


def power_train_network_viewer_hv_server(
        power_train_file_path: str,
        port: int = 5006,
        address: str = "localhost",
        layout_prog: str = "dot",
        refresh_rate: int = None,
        orientation: str = "TB",
):
    """
    Start a Bokeh Server for the power train network visualization.

    Args:
        power_train_file_path: Path to the power train configuration file
        port: Port to run the server on (default: 5006)
        address: Server address (default: localhost)
        layout_prog: Graphviz layout program ('dot', 'neato', 'fdp', 'sfdp', 'circo', 'twopi')
        refresh_rate: Monitor refresh rate in Hz (auto-detected if None)
    """

    # Auto-detect refresh rate if not provided
    if refresh_rate is None:
        refresh_rate = get_monitor_refresh_rate()

    # Calculate optimal animation parameters
    callback_interval = calculate_optimal_callback_interval(refresh_rate, target_fps=refresh_rate)
    animation_frames = calculate_animation_frames(refresh_rate, animation_duration_ms=1000)
    propeller_frames = 12  # Use 12 frames for propeller (divisible by common refresh rates)

    def make_document(doc):
        plot, edge_source, node_source, node_image_sequences, propeller_rotation_sequences = create_network_plot(
            power_train_file_path, layout_prog, orientation=orientation)
        animation_counter = [0]

        doc.add_root(column(plot))

        # Add periodic callback for animation
        def update():
            animation_counter[0] = (animation_counter[0] + 1) % animation_frames
            progress = animation_counter[0] / animation_frames  # 0 to 1

            num_segments = len(edge_source.data['xs'])
            edge_id_list = edge_source.data['edge_id']
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

            edge_source.patch({
                'line_alpha': [(slice(len(new_alphas)), new_alphas)]
            })

            # Update animated node icons
            if propeller_rotation_sequences:
                # Synchronize propeller rotation with refresh rate
                frame_index = (animation_counter[
                                   0] * propeller_frames // animation_frames) % propeller_frames

                new_urls = []
                updated = False

                for i, node in enumerate(node_source.data['name']):
                    if node in propeller_rotation_sequences:
                        seq = propeller_rotation_sequences[node]
                        frame_idx = frame_index % len(seq)
                        new_urls.append(seq[frame_idx])
                        updated = True
                    elif node in node_image_sequences:
                        seq = node_image_sequences[node]
                        frame_idx = (animation_counter[0] * len(seq) // animation_frames) % len(seq)
                        new_urls.append(seq[frame_idx])
                        updated = True
                    else:
                        new_urls.append(node_source.data['url'][i])

                if updated and new_urls and len(new_urls) == len(node_source.data['url']):
                    node_source.patch({
                        'url': [(slice(len(new_urls)), new_urls)]
                    })

        doc.add_periodic_callback(update, callback_interval)

    server = Server(
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

def _string_clean_up(old_string):
    # In case for list type definition
    if isinstance(old_string, list):
        old_string = old_string[0].capitalize() + ", " + old_string[1].capitalize()

    # Replace underscore with space
    new_string = re.sub(r"[_:/]+", " ", old_string)

    # Add a space after 'DC' if followed immediately by a letter or number
    new_string = re.sub(r"\bDC(?=[A-Za-z0-9])", "DC ", new_string)
    new_string = re.sub(r"\bDC DC(?=[A-Za-z0-9])", "DC-DC ", new_string)

    # Add space before a capital letter preceded by a lowercase letter
    new_string = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", new_string)

    # Remove extra spaces
    new_string = re.sub(r"\s+", " ", new_string).strip()

    return new_string

if __name__ == "__main__":
    # Example usage
    power_train_file_path = os.path.join(DATA_FOLDER_PATH, "simple_assembly_tri_prop_two_chainz.yml")
    power_train_network_viewer_hv_server(
        power_train_file_path=power_train_file_path,
        layout_prog="dot",
        orientation="LR"
    )