# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Port and edge management logic.

Mixin class :class:`PortEdgeMixin` is not meant to be instantiated directly;
it is composed into :class:`BuilderHandler` via multiple inheritance.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np

from fastga_he.gui.power_train_network_viewer import ICONS_CONFIG, DEFAULT_COLOR
from .power_train_builder_state import (
    DEFAULT_NODE_ICON_SIZE,
    EDGE_COLUMNS,
    NODE_RADIUS,
    PORT_COLUMNS,
    PORT_RADIUS,
    copy_column_data,
    empty_columns,
    select_edges,
)

_LOGGER = logging.getLogger(__name__)

# Clearance between the icon edge and each port centre at the default icon size
_PORT_GAP = 20.0

# Fill and outline opacity of the port balls of the selected node (others are hidden)
_SELECTED_PORT_FILL_ALPHA = 0.3
_SELECTED_PORT_LINE_ALPHA = 0.5


# For IDE type-checking only
if TYPE_CHECKING:
    from .power_train_builder_state import BuilderState


class PortEdgeMixin:
    """
    Handles all node, port, and edge geometry: finding the nearest placed
    node (:meth:`_best_possible_node`), the nearest port
    (:meth:`_find_nearest_port`), and the nearest edge
    (:meth:`_find_nearest_edge`) to a tap — a cohesive hit-testing trio used
    by delete mode, selection, and the port-connection workflow — plus
    adding, removing, and redrawing ports and edges on the Bokeh canvas.

    Depends on ``self.state`` (a :class:`BuilderState` instance) being set by
    the concrete class before any method is called.
    """

    state: "BuilderState"

    # -----------------------------------------------------------------------
    # Nearest-node helper
    # -----------------------------------------------------------------------

    def _best_possible_node(self, x: float, y: float):
        """
        Find the nearest placed node to canvas coordinates ``(x, y)``.

        :param x: Tap x coordinate in canvas data-units.
        :param y: Tap y coordinate in canvas data-units.

        :return: ``(best_index, shortest_distance, nodes_data)`` where ``best_index`` is the
            zero-based index of the closest node (``None`` if nothing is within snap
            distance), ``shortest_distance`` is the Euclidean distance from the tap to that node
            (``None`` if no node is, close enough), and ``nodes_data`` is the raw
            ``placed_nodes_source.data`` dict.
        """
        nodes_data = self.state.placed_nodes_source.data
        xs = list(nodes_data.get("x", []))
        ys = list(nodes_data.get("y", []))
        if not xs:
            return None, None, nodes_data

        snap = self.icon_size
        best_index = None
        shortest_distance = float("inf")
        for node_position_index, (node_x, node_y) in enumerate(zip(xs, ys, strict=True)):
            distance_to_tap = ((x - node_x) ** 2 + (y - node_y) ** 2) ** 0.5
            # Only consider nodes within snap distance and track the closest one
            if distance_to_tap < snap and distance_to_tap < shortest_distance:
                shortest_distance = distance_to_tap
                best_index = node_position_index

        if best_index is None:
            shortest_distance = None

        return best_index, shortest_distance, nodes_data

    # -----------------------------------------------------------------------
    # Port hit-testing and connection workflow
    # -----------------------------------------------------------------------

    def _find_nearest_port(self, x: float, y: float) -> dict | None:
        """
        Return metadata for the nearest port ball within snap distance, or ``None``.

        :param x: Tap x coordinate in canvas data-units.
        :param y: Tap y coordinate in canvas data-units.

        :return: A dict with keys ``kind``, ``x``, ``y``, ``color``, ``label``,
            and ``node_index`` for the nearest port, or ``None``.
        """
        snap = PORT_RADIUS * 2.5
        nearest_port, nearest_port_distance = None, snap
        for kind, port_source in [
            ("source", self.state.source_port_source),
            ("target", self.state.target_port_source),
        ]:
            if port_source is None:
                continue
            data = port_source.data
            for i, (port_x, port_y) in enumerate(
                zip(list(data.get("x", [])), list(data.get("y", [])), strict=True)
            ):
                distance = ((x - port_x) ** 2 + (y - port_y) ** 2) ** 0.5
                if distance < nearest_port_distance:
                    nearest_port_distance = distance
                    nearest_port = {
                        "kind": kind,
                        "x": port_x,
                        "y": port_y,
                        "color": list(data["color"])[i],
                        "label": list(data["label"])[i],
                        "node_index": list(data["node_index"])[i],
                    }
        return nearest_port

    def _cancel_pending_connection(self):
        """Cancel any port selection waiting for a second click."""
        self.state.awaiting_second_port_click = None
        if self.state.awaiting_second_port_click_source is not None:
            self.state.awaiting_second_port_click_source.data = dict(x=[], y=[], color=[])

    def _handle_port_tap(self, port: dict):
        """
        Manage the two-click port-connection workflow.

        First click stores the port as pending; second click on a compatible
        port (opposite kind, same energy colour) draws a permanent edge.
        Tapping the same port again cancels the pending selection.

        An undo snapshot is pushed **before** the permanent edge is committed
        (i.e. when the second compatible port is tapped) so the user can
        reverse a mis-wired connection.

        :param port: Port metadata dict with keys ``kind``, ``x``, ``y``,
            ``color``, ``label``, and ``node_index``.
        """
        self._clear_temp_edges()
        pending = self.state.awaiting_second_port_click
        if pending is None:
            self.state.awaiting_second_port_click = port
            if self.state.awaiting_second_port_click_source is not None:
                self.state.awaiting_second_port_click_source.data = dict(
                    x=[port["x"]], y=[port["y"]], color=[port["color"]]
                )
            return
        # Same port tapped again -> cancel
        if (
            port["kind"] == pending["kind"]
            and port["node_index"] == pending["node_index"]
            and port["label"] == pending["label"]
        ):
            self._cancel_pending_connection()
            return
        # Connect source <-> target of the same energy type (colour).
        # Push undo *before* committing the edge so the action is reversible
        if port["kind"] != pending["kind"] and port["color"] == pending["color"]:
            self._push_undo()
            self._add_edge(pending, port)
            self._rebuild_all_ports()
        self._cancel_pending_connection()

    # -----------------------------------------------------------------------
    # Edge hit-testing
    # -----------------------------------------------------------------------
    @staticmethod
    def _cursor_to_segment_distance(
        px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """
        Return the minimum Euclidean distance from ``(px, py)`` to the segment
        ``(x1, y1)``–``(x2, y2)``.

        Used by :meth:`_find_nearest_edge` for click-to-delete on edges.

        :param px: Cursor x coordinate in canvas data-units.
        :param py: Cursor y coordinate in canvas data-units.
        :param x1: Segment start x coordinate.
        :param y1: Segment start y coordinate.
        :param x2: Segment end x coordinate.
        :param y2: Segment end y coordinate.
        :return: Minimum distance from the cursor to the segment, in data-units.
        """
        segment_delta_x, segment_delta_y = x2 - x1, y2 - y1
        if segment_delta_x == 0 and segment_delta_y == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        projection_parameter = float(
            np.clip(
                ((px - x1) * segment_delta_x + (py - y1) * segment_delta_y)
                / (segment_delta_x**2 + segment_delta_y**2),
                0.0,
                1.0,
            )
        )
        return (
            (px - (x1 + projection_parameter * segment_delta_x)) ** 2
            + (py - (y1 + projection_parameter * segment_delta_y)) ** 2
        ) ** 0.5

    def _find_nearest_edge(self, x: float, y: float, snap: float = 12.0) -> int | None:
        """
        Return the index of the edge closest to ``(x, y)`` within *snap* pixels,
        or ``None`` if no edge is close enough.

        :param x: Tap x coordinate in canvas data-units.
        :param y: Tap y coordinate in canvas data-units.
        :param snap: Maximum allowed distance (data-units) for a hit.

        :return: Zero-based row index into ``edge_source``, or ``None``.
        """
        if self.state.edge_source is None:
            return None

        edge_data = self.state.edge_source.data
        best_index, best_distance = None, snap
        for i, (xs, ys) in enumerate(
            zip(edge_data.get("xs", []), edge_data.get("ys", []), strict=True)
        ):
            if len(xs) < 2:
                continue
            distance = self._cursor_to_segment_distance(x, y, xs[0], ys[0], xs[1], ys[1])
            if distance < best_distance:
                best_distance = distance
                best_index = i
        return best_index

    # -----------------------------------------------------------------------
    # Permanent edges
    # -----------------------------------------------------------------------

    def _add_edge(self, starting_port: dict, ending_port: dict):
        """
        Append a permanent edge between two ports to ``edge_source``.

        Guards against already-occupied ports and loop connections; rejects
        the edge silently if either check fails.

        .. note::
            This method does **not** push an undo snapshot. Callers that
            invoke ``_add_edge`` directly (e.g. Apply, see
            :meth:`~.power_train_builder_handler_config_panel.ConfigPanelMixin.apply_node_configurations`)
            are responsible for calling :meth:`_push_undo` beforehand.

        :param starting_port: Metadata dict for the source-side (starting) port.
        :param ending_port: Metadata dict for the target-side (ending) port.
        """
        if self.state.edge_source is None:
            return
        edge_data = copy_column_data(self.state.edge_source.data)

        connected_port_keys = _connected_port_keys(edge_data)
        if any(
            (port["kind"], port["node_index"], port["label"]) in connected_port_keys
            for port in (starting_port, ending_port)
        ):
            _LOGGER.info("Port already connected; edge rejected.")
            return

        # A second edge between the same two nodes, in either direction
        node_pair = {starting_port["node_index"], ending_port["node_index"]}
        if any(
            {starting_node_index, ending_node_index} == node_pair
            for starting_node_index, ending_node_index in zip(
                edge_data.get("starting_node_index", []),
                edge_data.get("ending_node_index", []),
                strict=True,
            )
        ):
            _LOGGER.info("Loop connection detected; edge rejected.")
            return

        # Update the local copy of the edge data and push it back to the Bokeh source
        new_edge_row = {
            "xs": [starting_port["x"], ending_port["x"]],
            "ys": [starting_port["y"], ending_port["y"]],
            "color": starting_port["color"],
        }
        for end, port in (("starting", starting_port), ("ending", ending_port)):
            new_edge_row[f"{end}_node_index"] = port["node_index"]
            new_edge_row[f"{end}_port_label"] = port["label"]
            new_edge_row[f"{end}_port_kind"] = port["kind"]
        for column_name in EDGE_COLUMNS:
            edge_data[column_name].append(new_edge_row[column_name])
        self.state.edge_source.data = edge_data

        if self.state.selected_node_index in node_pair:
            self._mark_ports_connected(starting_port, ending_port)
            self._refresh_connections_table(self.state.selected_node_index)

        self._mark_unsaved()
        _LOGGER.info(
            "Edge: %s port %s (node %d) <-> %s port %s (node %d)",
            starting_port["kind"],
            starting_port["label"],
            starting_port["node_index"],
            ending_port["kind"],
            ending_port["label"],
            ending_port["node_index"],
        )

    def _mark_ports_connected(self, *ports: dict):
        """
        Flag each port in *ports* as connected in its port data source.

        :param ports: Port metadata dicts with keys ``kind``, ``node_index`` and ``label``.
        """
        source_data = copy_column_data(self.state.source_port_source.data)
        target_data = copy_column_data(self.state.target_port_source.data)
        for port in ports:
            self._mark_connected(source_data if port["kind"] == "source" else target_data, port)
        self.state.source_port_source.data = source_data
        self.state.target_port_source.data = target_data

    @staticmethod
    def _mark_connected(data: dict, port: dict) -> None:
        """
        Mark the port matching *port* as connected in the *data* snapshot.

        Mutates ``data["connected"]`` in-place within the caller's local copy;
        the caller is responsible for writing the result back to the data source.

        :param data: Snapshot dict of a port data source (source or target).
        :param port: Port metadata dict with keys ``node_index`` and ``label``.
        """
        data["connected"] = [
            (
                "True"
                if (
                    data["node_index"][i] == port["node_index"]
                    and data["label"][i] == port["label"]
                )
                else data["connected"][i]
            )
            for i in range(len(data["node_index"]))
        ]

    # -----------------------------------------------------------------------
    # Temporary / preview edges
    # -----------------------------------------------------------------------

    def _add_edge_temp(self, starting_port: dict, ending_port: dict):
        """
        Draw a dashed preview line between two ports in ``panel_preview_edge_source``.

        The line is **not** registered in ``edge_source``; it becomes permanent
        only when :meth:`apply_node_configurations` is called.

        :param starting_port: Metadata dict for the source-side (starting) port.
        :param ending_port: Metadata dict for the target-side (ending) port.
        """
        if self.state.panel_preview_edge_source is None:
            return
        temp_edge_data = copy_column_data(self.state.panel_preview_edge_source.data)
        # Update the temp edge data with the new edge coordinates and color
        temp_edge_data["xs"].append([starting_port["x"], ending_port["x"]])
        temp_edge_data["ys"].append([starting_port["y"], ending_port["y"]])
        temp_edge_data["color"].append(starting_port["color"])
        # Push the updated temp edge data back to the Bokeh source
        self.state.panel_preview_edge_source.data = temp_edge_data
        _LOGGER.info(
            "Temp edge: %s:%s -> %s:%s",
            starting_port["kind"],
            starting_port["label"],
            ending_port["kind"],
            ending_port["label"],
        )

    def _clear_temp_edges(self):
        """Wipe all dashed preview edges and the pending-connection list."""
        self.state.panel_pending_connections.clear()
        self._clear_temp_edge_visuals()

    def _clear_temp_edge_visuals(self):
        """Clear only the canvas dashed lines, leave panel_pending_connections intact."""
        if self.state.panel_preview_edge_source is not None:
            self.state.panel_preview_edge_source.data = dict(xs=[], ys=[], color=[])

    def _rebuild_edges(self):
        """
        Recompute xs/ys in edge_source from stored port-identity columns. Here synchronise
        the `edge_data` with the current port state in the newly configured `source_port_source`
        and `target_port_source`.

        Called at the end of :meth:`_rebuild_all_ports` so edges track port
        states after port-count spinner changes.  Edges whose ports no
        longer exist (e.g. port count reduced) are silently dropped.
        """
        if self.state.edge_source is None:
            return
        edge_data = copy_column_data(self.state.edge_source.data)
        if not edge_data.get("starting_node_index"):
            return
        # Build a lookup: (kind, node_index, label) -> (x, y) for every port currently
        port_position_dict: dict = {}
        for kind, data_source in [
            ("source", self.state.source_port_source),
            ("target", self.state.target_port_source),
        ]:
            if data_source is None:
                continue
            dataset = data_source.data
            for i, (port_x, port_y) in enumerate(
                zip(list(dataset.get("x", [])), list(dataset.get("y", [])), strict=True)
            ):
                key = (kind, list(dataset["node_index"])[i], list(dataset["label"])[i])
                port_position_dict[key] = (port_x, port_y)
        # Walk every stored edge and look up fresh coordinates for its two endpoints
        new_xs, new_ys, valid_edge_indices = [], [], []
        for i in range(len(edge_data["starting_node_index"])):
            starting_port_key = (
                edge_data["starting_port_kind"][i],
                edge_data["starting_node_index"][i],
                edge_data["starting_port_label"][i],
            )
            ending_port_key = (
                edge_data["ending_port_kind"][i],
                edge_data["ending_node_index"][i],
                edge_data["ending_port_label"][i],
            )
            if starting_port_key in port_position_dict and ending_port_key in port_position_dict:
                starting_x, starting_y = port_position_dict[starting_port_key]
                ending_x, ending_y = port_position_dict[ending_port_key]
                new_xs.append([starting_x, ending_x])
                new_ys.append([starting_y, ending_y])
                valid_edge_indices.append(i)  # only keep edges whose ports still exist

        new_edge_data = select_edges(edge_data, valid_edge_indices)
        new_edge_data["xs"] = new_xs
        new_edge_data["ys"] = new_ys
        self.state.edge_source.data = new_edge_data

    # -----------------------------------------------------------------------
    # Port rebuilding
    # -----------------------------------------------------------------------

    def _rebuild_all_ports(self, override_counts: dict = None):
        """
        Recompute every port ball position from the current placed-nodes data
        and push the result into ``source_port_source`` / ``target_port_source``,
        then reposition/prune ``edge_source`` to match via :meth:`_rebuild_edges`.

        Called whenever nodes are added, moved, or deleted, and after every
        *committed* port-count change (i.e. after Apply writes the new
        ``n_sources`` / ``n_targets`` into ``placed_nodes_source``).

        :param override_counts: Optional ``{node_index: (n_source, n_target)}``
            used **only** for this call's rendering — it never writes back
            into ``placed_nodes_source``. This is how the port-count spinners
            get a live visual preview (see
            :meth:`~.power_train_builder_handler_config_panel.ConfigPanelMixin._on_port_count_spinner_change`)
            while the change stays uncommitted until Apply.

            Passing a non-empty ``override_counts`` still runs the normal
            :meth:`_rebuild_edges` pass — so an edge attached to a port that
            the preview count removes disappears from the canvas immediately,
            matching what Apply would actually do — but it is preceded by a
            snapshot/restore step so nothing is lost if the preview is
            cancelled:

            * The *first* preview call in a session snapshots the current
              ``edge_source.data`` into ``self.state.edge_preview_backup``.
            * Every subsequent preview call restores that snapshot into
              ``edge_source`` **before** recomputing, so each tick prunes
              against the true, last-committed edge list rather than the
              previous tick's already-pruned one — spinning the count back
              up brings a hidden edge straight back.
            * The next *real* (``override_counts`` falsy) call restores the
              snapshot and clears it, so navigating away without clicking
              Apply — reselecting, deleting, placing a new node, opening
              Delete mode — leaves ``edge_source`` exactly as it was before
              the preview ever started.
        """
        if self.state.source_port_source is None or self.state.target_port_source is None:
            return

        override_counts = override_counts or {}
        self._sync_edge_preview_backup(is_preview=bool(override_counts))

        # Port positions scale with the icon-size slider: every term of the
        # orbit radius (icon radius + port radius + gap) is multiplied by
        # size_factor = icon_size / DEFAULT_NODE_ICON_SIZE, so each port sits at
        # exactly `size_factor` times its default offset from the node centre
        size_factor = self.icon_size / DEFAULT_NODE_ICON_SIZE

        placed_nodes_data = self.state.placed_nodes_source.data
        n_sources_list = list(placed_nodes_data.get("n_sources", []))
        n_targets_list = list(placed_nodes_data.get("n_targets", []))
        source_ports = empty_columns(PORT_COLUMNS)
        target_ports = empty_columns(PORT_COLUMNS)

        for i, (center_x, center_y, icon_type, node_type, node_name) in enumerate(
            zip(
                list(placed_nodes_data.get("x", [])),
                list(placed_nodes_data.get("y", [])),
                list(placed_nodes_data.get("icon_type", [])),
                list(placed_nodes_data.get("node_type", [])),
                list(placed_nodes_data.get("name", [])),
                # Columns missing from an older save are empty: draw no ports
                # rather than fail
                strict=False,
            )
        ):
            n_source, n_target = override_counts.get(i) or (
                _stored_port_count(n_sources_list, i, self.state.default_source_count, node_type),
                _stored_port_count(n_targets_list, i, self.state.default_target_count, node_type),
            )

            ports = self.compute_ports(
                center_x,
                center_y,
                NODE_RADIUS * size_factor,
                PORT_RADIUS * size_factor,
                n_source,
                n_target,
                gap_pixels=_PORT_GAP * size_factor,
            )
            # Port balls are only visible on the selected node
            is_selected = i == self.state.selected_node_index
            node_row = {
                "node_index": i,
                "node_name": [node_name],
                "node_type": [node_type],
                "fill_alpha": _SELECTED_PORT_FILL_ALPHA if is_selected else 0.0,
                "line_alpha": _SELECTED_PORT_LINE_ALPHA if is_selected else 0.0,
            }
            icon_config = ICONS_CONFIG.get(icon_type, {})
            for port_columns, side_ports, color in (
                (source_ports, ports["outputs"], icon_config.get("source_color")),
                (target_ports, ports["inputs"], icon_config.get("target_color")),
            ):
                for port in side_ports:
                    _append_port_row(port_columns, port, color or DEFAULT_COLOR, node_row)

        connected_port_keys = _connected_port_keys(
            self.state.edge_source.data if self.state.edge_source is not None else {}
        )
        for kind, port_columns in (("source", source_ports), ("target", target_ports)):
            port_columns["connected"] = [
                "True" if (kind, node_index, label) in connected_port_keys else "False"
                for node_index, label in zip(
                    port_columns["node_index"], port_columns["label"], strict=True
                )
            ]
        self.state.source_port_source.data = source_ports
        self.state.target_port_source.data = target_ports

        # Always reposition/prune edges against whatever ports were just
        # drawn (real or preview) — see the snapshot/restore logic at the
        # top of this method for why a preview pass is still safe to reverse
        self._rebuild_edges()

        if self.state.selected_node_index is not None:
            self._refresh_connections_table(self.state.selected_node_index)

    def _sync_edge_preview_backup(self, is_preview: bool):
        """
        Keep the true edge list safe while a port-count preview hides edges.

        See :meth:`_rebuild_all_ports` for the full snapshot / restore protocol.

        :param is_preview: ``True`` when the current rebuild is a port-count preview.
        """
        if self.state.edge_source is None:
            return
        if is_preview:
            if self.state.edge_preview_backup is None:
                # First preview tick of this session: remember the truth (backup)
                self.state.edge_preview_backup = copy_column_data(self.state.edge_source.data)
            else:
                # Later ticks: always prune from the truth (backup), not from
                # whatever the previous tick (preview) already hid
                self.state.edge_source.data = copy_column_data(self.state.edge_preview_backup)
        elif self.state.edge_preview_backup is not None:
            # Real rebuild after a preview: put the true edges back before
            # this call's own (real, committed-count-based) pruning runs
            self.state.edge_source.data = copy_column_data(self.state.edge_preview_backup)
            self.state.edge_preview_backup = None

    # -----------------------------------------------------------------------
    # Port placement
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_ports(
        center_x: float,
        center_y: float,
        node_radius: float = NODE_RADIUS,
        port_radius: float = PORT_RADIUS,
        number_of_outputs: int = 0,
        number_of_inputs: int = 0,
        spread_angle_degrees: float = 120.0,
        gap_pixels: float = 20.0,  # = _PORT_GAP
    ) -> dict:
        """
        Place port balls evenly around the component node placed on canvas.

        Output ports fan around the top half (centred at −90°);
        input ports fan around the bottom half (centred at +90°).

        :param center_x: Component node centre x in canvas coordinates.
        :param center_y: Component node centre y in canvas coordinates.
        :param node_radius: Radius of the node circle in data-units / pixels.
        :param port_radius: Radius of each port ball in data-units / pixels.
        :param number_of_outputs: Number of output (source) ports.
        :param number_of_inputs: Number of input (target) ports.
        :param spread_angle_degrees: Angular fan for each half in degrees (default 120).
        :param gap_pixels: Clearance between the node edge and each port centre in pixels.
        :return: A dictionary ``{"outputs": [Port, …], "inputs": [Port, …]}`` where
            each Port is a dict with keys ``index``, ``kind``, ``angle_deg``, ``x``,
            and ``y``.
        """
        orbit_radius = node_radius + port_radius + gap_pixels

        def _angle_for_port(port_index: int, total_port_count: int, centre_angle: float) -> float:
            """Angle (degrees) of port *port_index*, spread evenly around *centre_angle*."""
            if total_port_count == 1:
                return centre_angle
            start_angle = centre_angle - spread_angle_degrees / 2
            angle_step = spread_angle_degrees / (total_port_count - 1)
            return start_angle + port_index * angle_step

        def _coordinates_from_angle(angle_degrees: float):
            """Canvas ``(x, y)`` of the point on the port orbit at *angle_degrees*."""
            angle_radians = float(np.radians(angle_degrees))
            return (
                float(center_x + orbit_radius * np.cos(angle_radians)),
                float(center_y + orbit_radius * np.sin(angle_radians)),
            )

        output_ports = []
        for port_index in range(number_of_outputs):
            angle_degrees = _angle_for_port(port_index, number_of_outputs, -90.0)
            port_x, port_y = _coordinates_from_angle(angle_degrees)
            output_ports.append(
                {
                    "index": port_index,
                    "kind": "output",
                    "angle_deg": angle_degrees,
                    "x": port_x,
                    "y": port_y,
                }
            )

        input_ports = []
        for port_index in range(number_of_inputs):
            angle_degrees = _angle_for_port(port_index, number_of_inputs, 90.0)
            port_x, port_y = _coordinates_from_angle(angle_degrees)
            input_ports.append(
                {
                    "index": port_index,
                    "kind": "input",
                    "angle_deg": angle_degrees,
                    "x": port_x,
                    "y": port_y,
                }
            )

        return {"outputs": output_ports, "inputs": input_ports}


def _append_port_row(port_columns: dict, port: dict, color: str, node_row: dict) -> None:
    """
    Append one port ball to port data built with :data:`PORT_COLUMNS`.

    The ``connected`` column is filled afterwards, once every port is known.

    :param port_columns: Port data being built, edited in place.
    :param port: Port from :meth:`PortEdgeMixin.compute_ports` (``x``, ``y``, ``index``).
    :param color: Energy-type colour of the port.
    :param node_row: Values shared by every port of the node (``node_index``,
        ``node_name``, ``node_type``, ``fill_alpha``, ``line_alpha``).
    """
    port_columns["x"].append(port["x"])
    port_columns["y"].append(port["y"])
    port_columns["color"].append(color)
    port_columns["label"].append(str(port["index"] + 1))
    for column_name, value in node_row.items():
        port_columns[column_name].append(value)


def _connected_port_keys(edge_data: dict) -> set:
    """
    ``(kind, node_index, label)`` of every port that has an edge.

    :param edge_data: ``edge_source.data`` (or a copy of it).
    :return: Set of port keys.
    """
    connected_port_keys: set = set()
    for edge_position in range(len(edge_data.get("starting_node_index", []))):
        for end in ("starting", "ending"):
            connected_port_keys.add(
                (
                    edge_data[f"{end}_port_kind"][edge_position],
                    edge_data[f"{end}_node_index"][edge_position],
                    edge_data[f"{end}_port_label"][edge_position],
                )
            )
    return connected_port_keys


def _stored_port_count(port_counts: list, node_index: int, default_counts: dict, node_type: str):
    """
    Port count of one side of a node, falling back to its type's default.

    :param port_counts: ``n_sources`` or ``n_targets`` column.
    :param node_index: Zero-based index of the node.
    :param default_counts: ``default_source_count`` or ``default_target_count``.
    :param node_type: Component type of the node.
    :return: Number of ports on that side.
    """
    if node_index < len(port_counts):
        return int(port_counts[node_index])
    return default_counts.get(node_type, 0)
