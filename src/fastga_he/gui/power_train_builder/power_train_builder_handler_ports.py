# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
Port and edge management logic.

Mixin class :class:`PortEdgeMixin` is not meant to be instantiated directly;
it is composed into :class:`PlacementHandler` via multiple inheritance.
"""

import logging
from typing import TYPE_CHECKING

from fastga_he.gui.power_train_network_viewer import ICONS_CONFIG, DEFAULT_COLOR
from .power_train_builder_state import NODE_RADIUS, PORT_RADIUS
from .power_train_builder_state import compute_ports

_LOGGER = logging.getLogger(__name__)

# For IDE type-checking only
if TYPE_CHECKING:
    from .power_train_builder_state import BuilderState


class PortEdgeMixin:
    """
    Handles all port and edge geometry: finding, adding, removing, and redrawing
    ports and edges on the Bokeh canvas.

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
        for node_position_index, (node_x, node_y) in enumerate(zip(xs, ys)):
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
                zip(list(data.get("x", [])), list(data.get("y", [])))
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
        self.state.pending_port = None
        if self.state.pending_port_source is not None:
            self.state.pending_port_source.data = dict(x=[], y=[], color=[])

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
        pending = self.state.pending_port
        if pending is None:
            self.state.pending_port = port
            if self.state.pending_port_source is not None:
                self.state.pending_port_source.data = dict(
                    x=[port["x"]], y=[port["y"]], color=[port["color"]]
                )
            return
        # Same port tapped again → cancel
        if (
            port["kind"] == pending["kind"]
            and port["node_index"] == pending["node_index"]
            and port["label"] == pending["label"]
        ):
            self._cancel_pending_connection()
            return
        # Connect source ↔ target of the same energy type (colour).
        # Push undo *before* committing the edge so the action is reversible.
        if port["kind"] != pending["kind"] and port["color"] == pending["color"]:
            self._push_undo()
            self._add_edge(pending, port)
            self._rebuild_all_ports()
        self._cancel_pending_connection()

    # -----------------------------------------------------------------------
    # Edge hit-testing
    # -----------------------------------------------------------------------

    def _cursor_to_segment_distance(
        self, px: float, py: float, x1: float, y1: float, x2: float, y2: float
    ) -> float:
        """
        Return the minimum Euclidean distance from ``(px, py)`` to the segment
        ``(x1, y1)``–``(x2, y2)``.

        Used by :meth:`_find_nearest_edge` for click-to-delete on edges.
        """
        segment_delta_x, segment_delta_y = x2 - x1, y2 - y1
        if segment_delta_x == 0 and segment_delta_y == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        projection_parameter = max(
            0.0,
            min(
                1.0,
                ((px - x1) * segment_delta_x + (py - y1) * segment_delta_y)
                / (segment_delta_x**2 + segment_delta_y**2),
            ),
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
        for i, (xs, ys) in enumerate(zip(edge_data.get("xs", []), edge_data.get("ys", []))):
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
            This method does **not** push an undo snapshot.  Callers that
            invoke ``_add_edge`` directly (e.g. the Apply button in
            :class:`~.power_train_builder_launcher.PowertrainBuilderLauncher`)
            are responsible for calling :meth:`_push_undo` beforehand.

        :param starting_port: Metadata dict for the source-side (starting) port.
        :param ending_port: Metadata dict for the target-side (ending) port.
        """
        if self.state.edge_source is None:
            return
        edge_data = {key: list(value) for key, value in self.state.edge_source.data.items()}

        for i in range(len(edge_data.get("starting_node_index", []))):
            starting_port_already_used = (
                edge_data["starting_node_index"][i] == starting_port["node_index"]
                and edge_data["starting_port_label"][i] == starting_port["label"]
                and edge_data["starting_port_kind"][i] == starting_port["kind"]
            ) or (
                edge_data["ending_node_index"][i] == starting_port["node_index"]
                and edge_data["ending_port_label"][i] == starting_port["label"]
                and edge_data["ending_port_kind"][i] == starting_port["kind"]
            )
            ending_port_already_used = (
                edge_data["starting_node_index"][i] == ending_port["node_index"]
                and edge_data["starting_port_label"][i] == ending_port["label"]
                and edge_data["starting_port_kind"][i] == ending_port["kind"]
            ) or (
                edge_data["ending_node_index"][i] == ending_port["node_index"]
                and edge_data["ending_port_label"][i] == ending_port["label"]
                and edge_data["ending_port_kind"][i] == ending_port["kind"]
            )
            if starting_port_already_used or ending_port_already_used:
                _LOGGER.info("Port already connected; edge rejected.")
                return

        loop_connection = any(
            (
                edge_data["starting_node_index"][i] == ending_port["node_index"]
                and edge_data["ending_node_index"][i] == starting_port["node_index"]
            )
            or (
                edge_data["starting_node_index"][i] == starting_port["node_index"]
                and edge_data["ending_node_index"][i] == ending_port["node_index"]
            )
            for i in range(len(edge_data.get("starting_node_index", [])))
        )
        if loop_connection:
            _LOGGER.info("Loop connection detected; edge rejected.")
            return

        edge_data["xs"].append([starting_port["x"], ending_port["x"]])
        edge_data["ys"].append([starting_port["y"], ending_port["y"]])
        edge_data["color"].append(starting_port["color"])
        edge_data["starting_node_index"].append(starting_port["node_index"])
        edge_data["starting_port_label"].append(starting_port["label"])
        edge_data["starting_port_kind"].append(starting_port["kind"])
        edge_data["ending_node_index"].append(ending_port["node_index"])
        edge_data["ending_port_label"].append(ending_port["label"])
        edge_data["ending_port_kind"].append(ending_port["kind"])
        self.state.edge_source.data = edge_data

        if self.state.selected_node_index is not None and (
            starting_port["node_index"] == self.state.selected_node_index
            or ending_port["node_index"] == self.state.selected_node_index
        ):
            source_data = {
                key: list(value) for key, value in self.state.source_port_source.data.items()
            }
            target_data = {
                key: list(value) for key, value in self.state.target_port_source.data.items()
            }

            if starting_port["kind"] == "source":
                self._mark_connected(source_data, starting_port)
                self._mark_connected(target_data, ending_port)
            else:
                self._mark_connected(target_data, starting_port)
                self._mark_connected(source_data, ending_port)

            self.state.source_port_source.data = source_data
            self.state.target_port_source.data = target_data
            self._refresh_connections_table(self.state.selected_node_index)

        self._mark_unsaved()
        _LOGGER.info(
            "Edge: %s port %s (node %d) ↔ %s port %s (node %d)",
            starting_port["kind"],
            starting_port["label"],
            starting_port["node_index"],
            ending_port["kind"],
            ending_port["label"],
            ending_port["node_index"],
        )

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
            "True"
            if (data["node_index"][i] == port["node_index"] and data["label"][i] == port["label"])
            else data["connected"][i]
            for i in range(len(data["node_index"]))
        ]

    # -----------------------------------------------------------------------
    # Temporary / preview edges
    # -----------------------------------------------------------------------

    def _add_edge_temp(self, starting_port: dict, ending_port: dict):
        """
        Draw a dashed preview line between two ports in ``temp_edge_source``.

        The line is **not** registered in ``edge_source``; it becomes permanent
        only when :meth:`apply_node_configurations` is called.
        """
        if self.state.temp_edge_source is None:
            return
        temp_edge_data = {
            key: list(value) for key, value in self.state.temp_edge_source.data.items()
        }
        temp_edge_data["xs"].append([starting_port["x"], ending_port["x"]])
        temp_edge_data["ys"].append([starting_port["y"], ending_port["y"]])
        temp_edge_data["color"].append(starting_port["color"])
        self.state.temp_edge_source.data = temp_edge_data
        _LOGGER.info(
            "Temp edge: %s:%s → %s:%s",
            starting_port["kind"],
            starting_port["label"],
            ending_port["kind"],
            ending_port["label"],
        )

    def _clear_temp_edges(self):
        """Wipe all dashed preview edges and the pending-connection list."""
        self.state.pending_connections.clear()
        self._clear_temp_edge_visuals()

    def _clear_temp_edge_visuals(self):
        """Clear only the canvas dashed lines, leave pending_connections intact."""
        if self.state.temp_edge_source is not None:
            self.state.temp_edge_source.data = dict(xs=[], ys=[], color=[])

    def _rebuild_edges(self):
        """
        Recompute xs/ys in edge_source from stored port-identity columns.

        Called at the end of :meth:`_rebuild_all_ports` so edges track port
        positions after port-count spinner changes.  Edges whose ports no
        longer exist (e.g. port count reduced) are silently dropped.
        """
        if self.state.edge_source is None:
            return
        edge_data = {key: list(value) for key, value in self.state.edge_source.data.items()}
        if not edge_data.get("starting_node_index"):
            return

        position: dict = {}
        for kind, data_source in [
            ("source", self.state.source_port_source),
            ("target", self.state.target_port_source),
        ]:
            if data_source is None:
                continue
            dataset = data_source.data
            for i, (px, py) in enumerate(
                zip(list(dataset.get("x", [])), list(dataset.get("y", [])))
            ):
                key = (kind, list(dataset["node_index"])[i], list(dataset["label"])[i])
                position[key] = (px, py)

        new_xs, new_ys, valid = [], [], []
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
            if starting_port_key in position and ending_port_key in position:
                starting_x, starting_y = position[starting_port_key]
                ending_x, ending_y = position[ending_port_key]
                new_xs.append([starting_x, ending_x])
                new_ys.append([starting_y, ending_y])
                valid.append(i)

        new_edge_data = {k: [edge_data[k][j] for j in valid] for k in edge_data}
        new_edge_data["xs"] = new_xs
        new_edge_data["ys"] = new_ys
        self.state.edge_source.data = new_edge_data

    # -----------------------------------------------------------------------
    # Port rebuilding
    # -----------------------------------------------------------------------

    def _rebuild_all_ports(self):
        """
        Recompute every port ball position from the current placed-nodes data
        and push the result into ``source_port_source`` / ``target_port_source``.

        Called whenever nodes are added, moved, deleted, or their port counts
        are changed via the spinner widgets.
        """
        if self.state.source_port_source is None or self.state.target_port_source is None:
            return

        selected_node_fill_alpha, selected_node_line_alpha = 0.3, 0.5

        placed_nodes_data = self.state.placed_nodes_source.data
        xs = list(placed_nodes_data.get("x", []))
        ys = list(placed_nodes_data.get("y", []))
        icon_types = list(placed_nodes_data.get("icon_type", []))
        node_types = list(placed_nodes_data.get("node_type", []))
        node_names = list(placed_nodes_data.get("name", []))
        n_sources_list = list(placed_nodes_data.get("n_sources", []))
        n_targets_list = list(placed_nodes_data.get("n_targets", []))

        source_x, source_y, source_color, source_label = [], [], [], []
        source_node_index, source_node_name, source_node_type = [], [], []
        source_fill_alpha, source_line_alpha = [], []

        target_x, target_y, target_color, target_label = [], [], [], []
        target_node_index, target_node_name, target_node_type = [], [], []
        target_fill_alpha, target_line_alpha = [], []

        connected_ports: set = set()
        if self.state.edge_source is not None:
            edge_data = self.state.edge_source.data
            for i in range(len(edge_data.get("starting_node_index", []))):
                connected_ports.add(
                    (
                        edge_data["starting_port_kind"][i],
                        edge_data["starting_node_index"][i],
                        edge_data["starting_port_label"][i],
                    )
                )
                connected_ports.add(
                    (
                        edge_data["ending_port_kind"][i],
                        edge_data["ending_node_index"][i],
                        edge_data["ending_port_label"][i],
                    )
                )

        has_selected_node = self.state.selected_node_index is not None
        selected_node_index = self.state.selected_node_index

        for i, (cx, cy, icon_type, node_type, node_name) in enumerate(
            zip(xs, ys, icon_types, node_types, node_names)
        ):
            n_source = (
                int(n_sources_list[i])
                if i < len(n_sources_list)
                else self.state.default_source_count.get(node_type, 0)
            )
            n_target = (
                int(n_targets_list[i])
                if i < len(n_targets_list)
                else self.state.default_target_count.get(node_type, 0)
            )

            cfg = ICONS_CONFIG.get(icon_type, {})
            raw_source_color = cfg.get("source_color") or DEFAULT_COLOR
            raw_target_color = cfg.get("target_color") or DEFAULT_COLOR

            ports = compute_ports(cx, cy, NODE_RADIUS, PORT_RADIUS, n_source, n_target)
            is_selected = has_selected_node and i == selected_node_index
            fill_alpha = selected_node_fill_alpha if is_selected else 0.0
            line_alpha = selected_node_line_alpha if is_selected else 0.0

            for port in ports["outputs"]:
                source_x.append(port["x"])
                source_y.append(port["y"])
                source_color.append(raw_source_color)
                source_label.append(str(port["index"] + 1))
                source_node_index.append(i)
                source_node_name.append([node_name])
                source_node_type.append([node_type])
                source_fill_alpha.append(fill_alpha)
                source_line_alpha.append(line_alpha)

            for port in ports["inputs"]:
                target_x.append(port["x"])
                target_y.append(port["y"])
                target_color.append(raw_target_color)
                target_label.append(str(port["index"] + 1))
                target_node_index.append(i)
                target_node_name.append([node_name])
                target_node_type.append([node_type])
                target_fill_alpha.append(fill_alpha)
                target_line_alpha.append(line_alpha)

        self.state.source_port_source.data = dict(
            x=source_x,
            y=source_y,
            color=source_color,
            label=source_label,
            node_index=source_node_index,
            node_name=source_node_name,
            node_type=source_node_type,
            fill_alpha=source_fill_alpha,
            line_alpha=source_line_alpha,
            kind=["source"] * len(source_x),
            connected=[
                "True"
                if ("source", source_node_index[i], source_label[i]) in connected_ports
                else "False"
                for i in range(len(source_x))
            ],
        )
        self.state.target_port_source.data = dict(
            x=target_x,
            y=target_y,
            color=target_color,
            label=target_label,
            node_index=target_node_index,
            node_name=target_node_name,
            node_type=target_node_type,
            fill_alpha=target_fill_alpha,
            line_alpha=target_line_alpha,
            kind=["target"] * len(target_x),
            connected=[
                "True"
                if ("target", target_node_index[i], target_label[i]) in connected_ports
                else "False"
                for i in range(len(target_x))
            ],
        )
        self._rebuild_edges()

        if self.state.selected_node_index is not None:
            self._refresh_connections_table(self.state.selected_node_index)
