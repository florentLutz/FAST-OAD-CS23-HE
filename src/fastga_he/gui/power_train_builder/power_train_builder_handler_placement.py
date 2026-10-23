# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
New components, icon size and view scale.

Creates new components on the canvas, and keeps everything drawn around them
in proportion to the icon size and to the window scale.

Mixin class :class:`PlacementMixin` is not meant to be instantiated directly;
it is composed into :class:`BuilderHandler` via multiple inheritance.
"""

import json
import logging

from .power_train_builder_metadata import icon_data_url
from .power_train_builder_state import (
    DEFAULT_NODE_ICON_SIZE,
    HOVER_RING_BASE_SIZE,
    NODE_COLUMNS,
    NODE_LABEL_BASE_OFFSET,
    SELECTION_RING_BASE_SIZE,
    _EMPTY,
    _NO_SYMMETRY_PEER,
    BUILDER_ICONS_CONFIG,
    UNDEFINED_PORT_TYPE,
    UNKNOWN_COMPONENT_TYPE,
    UNKNOWN_DEFAULT_PORT_COUNT,
    UNKNOWN_ICON_KEY,
    append_row,
    copy_column_data,
    default_options,
    default_position,
    BuilderState,
)

_LOGGER = logging.getLogger(__name__)


class PlacementMixin:
    """
    Creates new components on the canvas and keeps them sized to the icon-size slider.

    Depends on ``self.state`` (a :class:`BuilderState` instance) and
    ``self.icon_size`` being set by the concrete class before any method is
    called.
    """

    state: "BuilderState"
    icon_size: int

    # -----------------------------------------------------------------------
    # Node creation
    # -----------------------------------------------------------------------

    def _place_new_node(self, x: float, y: float):
        """
        Place a component of the armed toolbox type at ``(x, y)``, as one undoable step.

        The selected node is deselected first. The new component is named
        ``<icon_key>_<n>`` from ``placed_counter`` and gets the defaults of its
        type (:meth:`_new_component_defaults`). The toolbox component stays
        armed, so several can be placed with successive clicks.

        :param x: Canvas x coordinate (data-units) where the component is placed.
        :param y: Canvas y coordinate (data-units) where the component is placed.
        """
        # Deselect any previously selected node before placing a new component
        if self.state.selected_node_index is not None:
            self.state.selected_node_index = None
            self._clear_temp_edges()
            self._clear_node_table()

        # Capture state before placing a new component
        self._push_undo()

        icon_key = self.state.selected_component
        placement_count = self.state.placed_counter.get(icon_key, 0) + 1
        self.state.placed_counter[icon_key] = placement_count
        node_name = f"{icon_key}_{placement_count}"

        # One value per column of placed_nodes_source for the new component
        new_node_row = {
            "x": x,
            "y": y,
            "url": icon_data_url(BUILDER_ICONS_CONFIG[icon_key]["icon_path"]),
            "w": self.icon_size,
            "h": self.icon_size,
            "name": node_name,
            "icon_type": icon_key,
            "symmetry_name": _EMPTY,
            "symmetry_node_index": _NO_SYMMETRY_PEER,
            **self._new_component_defaults(icon_key),
        }
        self.state.placed_nodes_source.data = append_row(
            self.state.placed_nodes_source.data, NODE_COLUMNS, new_node_row
        )

        self._rebuild_all_ports()
        self._mark_unsaved()
        _LOGGER.info(
            "Placed %s (node_type=%s, position=%s) at (%.1f, %.1f)",
            node_name,
            new_node_row["node_type"],
            new_node_row["position"],
            x,
            y,
        )

    def _new_component_defaults(self, icon_key: str) -> dict:
        """
        Type, position, options, port counts and port types of a new component.

        A known component takes the first component type of its icon and that
        type's defaults. An unknown component starts as type ``"unknown"``,
        with no position or options, one source and one target port, both of
        undefined energy type.

        :param icon_key: Toolbox icon of the new component.
        :return: Those columns of the new ``placed_nodes_source`` row.
        """
        if icon_key == UNKNOWN_ICON_KEY:
            return {
                "node_type": UNKNOWN_COMPONENT_TYPE,
                "position": _EMPTY,
                "options": "{}",
                "n_sources": UNKNOWN_DEFAULT_PORT_COUNT,
                "n_targets": UNKNOWN_DEFAULT_PORT_COUNT,
                "source_port_type": UNDEFINED_PORT_TYPE,
                "target_port_type": UNDEFINED_PORT_TYPE,
            }
        component_type = self.state.icon_to_component_types.get(icon_key, [icon_key])[0]
        return {
            "node_type": component_type,
            "position": default_position(self.state.possible_position.get(component_type, [])),
            "options": json.dumps(
                default_options(self.state.possible_options.get(component_type, {}))
            ),
            "n_sources": self.state.default_source_count.get(component_type, 0),
            "n_targets": self.state.default_target_count.get(component_type, 0),
            # Known components take their port colours from their icon
            "source_port_type": _EMPTY,
            "target_port_type": _EMPTY,
        }

    # -----------------------------------------------------------------------
    # Node icon size (toolbox slider)
    # -----------------------------------------------------------------------

    def _on_node_size_slider_change(self, attr, old, new):
        """
        Bokeh ``on_change`` callback of the icon-size slider.

        Sets the size used for every node (existing and future placements) and
        rescales everything drawn around a node by ``new / DEFAULT_NODE_ICON_SIZE``.
        This is a display setting only: it is not an undoable edit and does not
        mark the design as unsaved.

        :param attr: Bokeh attribute name (unused, required by protocol).
        :param old: Previous slider value (unused).
        :param new: New icon size in canvas data units.
        """
        self.icon_size = int(new)
        self._apply_node_icon_size()
        # Port orbits depend on the icon size; this also re-routes the edges
        self._rebuild_all_ports()
        if self.state.selected_node_index is not None:
            self._refresh_connections_table(self.state.selected_node_index)

    def _on_view_scale_change(self, attr, old, new):
        """
        Bokeh ``on_change`` callback of the hidden ``view_scale_input``.

        The browser writes a JSON report here (see the launcher's
        ``_JS_FIT_VIEW``)::

            {"k": 1.27, "area": [x0, x1, y0, y1], "area_k": 1.27}

        * ``k`` -> ``state.view_scale``: data units per pixel of the fitted
          view. Icons are sized in data units, so they shrink on screen when
          it is > 1; the rings and label offset are in screen pixels and are
          rescaled to match.
        * ``area`` / ``area_k`` -> ``state.canvas_area``: what an empty design
          shows in this window, used to spread automatic layouts.

        A bare number (e.g. ``"1.25"``) is accepted as ``k`` alone. Anything
        malformed is ignored, part by part.
        """
        report = _parse_view_report(new)
        if report is None:
            return

        canvas_area = _canvas_area_from_report(report)
        if canvas_area is not None:
            self.state.canvas_area = canvas_area

        view_scale = _view_scale_from_report(report)
        if view_scale is None or view_scale == self.state.view_scale:
            return
        self.state.view_scale = view_scale
        self._apply_node_icon_size()

    def _apply_node_icon_size(self):
        """
        Resize every icon, and the rings and name labels around it, to ``self.icon_size``.

        * Icons: ``w`` / ``h`` of every placed node = ``icon_size`` (data units).
        * Hover ring, selection ring and name-label offset (screen pixels):
          base size x ``size_factor``, with
          ``size_factor = icon_size / DEFAULT_NODE_ICON_SIZE / state.view_scale``,
          so they keep hugging the icon whatever the window size. The port
          balls use the same factor without ``view_scale`` (see ``_draw_ports``),
          since they are placed in data units.

        Also called after undo / redo and loading, whose node data may carry
        icon sizes from before the slider moved. Does not rebuild ports.
        """
        size = self.icon_size
        size_factor = size / DEFAULT_NODE_ICON_SIZE / (self.state.view_scale or 1.0)

        nodes_data = self.state.placed_nodes_source.data
        num_nodes = len(nodes_data.get("x", []))
        if num_nodes and (
            list(nodes_data.get("w", [])) != [size] * num_nodes
            or list(nodes_data.get("h", [])) != [size] * num_nodes
        ):
            new_nodes_data = copy_column_data(nodes_data)
            new_nodes_data["w"] = [size] * num_nodes
            new_nodes_data["h"] = [size] * num_nodes
            self.state.placed_nodes_source.data = new_nodes_data

        for renderer, base_size in (
            (self.state.hover_ring_renderer, HOVER_RING_BASE_SIZE),
            (self.state.selection_ring_renderer, SELECTION_RING_BASE_SIZE),
        ):
            if renderer is None:
                continue
            renderer.glyph.size = base_size * size_factor
            # The hover ring has an explicit hover_glyph copy that must follow
            hover_glyph = getattr(renderer, "hover_glyph", None)
            if hover_glyph is not None and hasattr(hover_glyph, "size"):
                hover_glyph.size = base_size * size_factor

        if self.state.node_label_set is not None:
            self.state.node_label_set.y_offset = NODE_LABEL_BASE_OFFSET * size_factor


# ============================================================================
# View-scale report of the browser
# ============================================================================


def _parse_view_report(report_text) -> dict | None:
    """
    Decode the JSON the browser writes into ``view_scale_input``.

    :param report_text: New value of ``view_scale_input``.
    :return: The report as a dict (a bare number becomes ``{"k": number}``),
        or ``None`` if it is not valid JSON or not an object / number.
    """
    try:
        report = json.loads(report_text)
    except (TypeError, ValueError):
        return None
    if isinstance(report, (int, float)):
        return {"k": report}
    return report if isinstance(report, dict) else None


def _canvas_area_from_report(report: dict) -> dict | None:
    """
    Canvas area of an empty design in the user's window, from a view report.

    :param report: Decoded report (see :func:`_parse_view_report`).
    :return: ``{"x0", "x1", "y0", "y1", "per_px"}`` in data units, or ``None``
        if the report has no valid ``area`` (four numbers, increasing) or its
        scale (``area_k``, else ``k``) is not a positive number.
    """
    area = report.get("area")
    if not (isinstance(area, list) and len(area) == 4):
        return None
    try:
        x0, x1, y0, y1 = (float(value) for value in area)
        per_px = float(report.get("area_k", report.get("k", 1.0)))
    except (TypeError, ValueError):
        return None
    if x1 > x0 and y1 > y0 and per_px > 0:
        return dict(x0=x0, x1=x1, y0=y0, y1=y1, per_px=per_px)
    return None


def _view_scale_from_report(report: dict) -> float | None:
    """
    Data units per pixel of the fitted view, from a view report.

    :param report: Decoded report (see :func:`_parse_view_report`).
    :return: ``k`` as a positive float, or ``None`` if missing or invalid.
    """
    try:
        view_scale = float(report.get("k"))
    except (TypeError, ValueError):
        return None
    return view_scale if view_scale > 0 else None
