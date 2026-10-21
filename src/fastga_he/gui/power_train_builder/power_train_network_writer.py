# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
YAML serialiser for the powertrain network.

:class:`PowerTrainYAMLWriter` collects component and connection data from a
:class:`BuilderState` snapshot and writes a formatted YAML powertrain
configuration file, mirroring the schema consumed by FAST-OAD_CS23-HE.

Unless the design is saved as a final version, the file also carries a
GUI-only ``canvas_layout`` section that stores the canvas (x, y) of every
component, so the same YAML can be re-opened in the builder for later edits
with the exact same arrangement (see :mod:`.power_train_network_reader`).
FAST-OAD_CS23-HE ignores unknown top-level keys (it only logs a warning), so
the section does not affect the analysis.
"""

import json
import re
import logging
from io import StringIO
from ruamel.yaml import YAML

from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS

from .power_train_builder_state import (
    UNKNOWN_COMPONENT_TYPE,
    VARIABLE_PORT_DEFAULT,
    copy_column_data,
    default_options,
    edge_end_ports,
    is_unknown_node,
)

_LOGGER = logging.getLogger(__name__)

# Top-level YAML keys (the first four mirror FAST-OAD_CS23-HE's powertrain.py)
KEY_TITLE = "title"
KEY_PT_COMPONENTS = "power_train_components"
KEY_PT_CONNECTIONS = "component_connections"
KEY_PT_WATCHER = "watcher_file_path"
# GUI-only section: {component_name: {"x": float, "y": float}}
KEY_CANVAS_LAYOUT = "canvas_layout"

_CANVAS_LAYOUT_HEADER = "# Captured canvas positions used only by the builder GUI to reconstruct.\n"

# Keywords used to tell which "number_of_*" attribute of a variable-port
# component drives its source-side vs target-side port count. Shared with the
# reader so that save and load stay symmetric
SOURCE_SIDE_KEYWORDS = ("_inputs", "_tanks")
TARGET_SIDE_KEYWORDS = ("_outputs", "_engines", "_power_sources")

# Decimal places kept for canvas coordinates (sub-pixel precision is noise)
_LAYOUT_DECIMALS = 1

# Prefix of FAST-OAD_CS23-HE component ids. An unknown component's type is
# written as "<prefix><type>", unless it is already a full id (has a dot)
COMPONENT_ID_PREFIX = "fastga_he.pt_component."

# Extra canvas_layout keys of an unknown component: GUI-only settings that its
# FAST-OAD entry cannot hold, kept so that an editable file re-opens as saved
UNKNOWN_LAYOUT_KEYS = ("n_sources", "n_targets", "source_port_type", "target_port_type")

# KNOWN_COMPONENTS entry for each component type (the first one, if several share it)
_KNOWN_COMPONENT_BY_TYPE: dict = {}
for _known_component in KNOWN_COMPONENTS:
    _KNOWN_COMPONENT_BY_TYPE.setdefault(_known_component["components_type"], _known_component)


class PowerTrainYAMLWriter:
    """
    Assemble and write a YAML powertrain configuration from builder state data.

    The class reads node and edge data from the live :class:`BuilderState` at
    construction time, so all subsequent calls to :meth:`add_component`,
    :meth:`add_connection`, and :meth:`write` work on an immutable snapshot.

    Callers that have already taken their own ``copy_column_data`` snapshot
    can pass it in directly via ``node_data`` / ``edge_data`` to avoid
    re-copying the same live sources a second time. Either one left as
    ``None`` is copied from ``state``, so the class remains safe to construct
    standalone with just ``state``.
    """

    def __init__(
        self,
        state,
        node_data: dict = None,
        edge_data: dict = None,
        include_canvas_layout: bool = True,
    ):
        """
        Snapshot the builder state needed for serialisation.

        :param state: :class:`BuilderState` instance whose data sources are
            copied at construction time (used for either data argument left
            as ``None``, and always for ``possible_options`` /
            ``default_source_count``).
        :param node_data: Optional pre-copied ``placed_nodes_source.data``
            (via :func:`copy_column_data`). Copied from *state* if ``None``.
        :param edge_data: Optional pre-copied ``edge_source.data``. Copied
            from *state* if ``None``.
        :param include_canvas_layout: ``True`` (default) to append the GUI-only
            ``canvas_layout`` section for later edits; ``False`` for a "final
            version" file containing only what FAST-OAD_CS23-HE reads.
        """
        self.include_canvas_layout = include_canvas_layout
        self.data = {
            KEY_TITLE: "",
            KEY_PT_COMPONENTS: {},
            KEY_PT_CONNECTIONS: [],
            KEY_PT_WATCHER: "",
            KEY_CANVAS_LAYOUT: {},
        }
        # Fall back to copying the live sources, as documented above
        self.node_data = (
            node_data if node_data is not None else copy_column_data(state.placed_nodes_source.data)
        )
        self.edge_data = (
            edge_data if edge_data is not None else copy_column_data(state.edge_source.data)
        )
        self.possible_options = state.possible_options
        self.default_source_count = state.default_source_count

    def set_title(self, title: str) -> None:
        """
        Set the ``title`` field of the YAML output.

        :param title: Human-readable title for the powertrain design.
        """
        self.data[KEY_TITLE] = title

    def add_component(self, node_index: int) -> None:
        """
        Append a single component entry to ``power_train_components``.

        Reads component metadata (name, type, port counts, symmetry, options)
        from the snapshotted node data at *node_index*, merges any unset
        options with per-type defaults, and writes the resulting dict into the
        YAML data structure.

        For variable-port components (default source count
        ``VARIABLE_PORT_DEFAULT``),
        the current port counts are written as ``number_of_*`` option keys so
        that FAST-OAD can reconstruct the correct topology.

        An unknown component is written with the id built from its type (see
        :func:`unknown_component_id`) and exactly the options typed for it.

        The component's canvas (x, y) is also recorded under ``canvas_layout``
        so the file can be re-opened in the builder, with an unknown
        component's port counts and port types (:data:`UNKNOWN_LAYOUT_KEYS`).

        :param node_index: Zero-based row index into the snapshotted node data arrays.
        """
        component_name = self.node_data["name"][node_index]
        self.data[KEY_CANVAS_LAYOUT][component_name] = self._layout_entry(node_index)
        symmetry_node_index = self.node_data["symmetry_node_index"][node_index]
        # Written on one node of each pair only; an unknown component's peer
        # that is not on the canvas (index -1) is always written
        symmetry_component_name = (
            self.node_data["symmetry_name"][node_index]
            if symmetry_node_index < node_index
            else None
        )
        if is_unknown_node(self.node_data, node_index):
            component_id = unknown_component_id(self.node_data["node_type"][node_index])
            options_dict = self._saved_options(node_index)
        else:
            component_id, options_dict = self._known_id_and_options(node_index)

        # Keys in the order FAST-OAD files use: id, options, position, symmetrical
        component_entry = {"id": component_id}
        if options_dict:
            component_entry["options"] = options_dict
        component_entry["position"] = self.node_data["position"][node_index]
        if symmetry_component_name:
            component_entry["symmetrical"] = symmetry_component_name
        self.data[KEY_PT_COMPONENTS][component_name] = component_entry

    def _layout_entry(self, node_index: int) -> dict:
        """
        ``canvas_layout`` entry of a node: its canvas (x, y), plus the
        :data:`UNKNOWN_LAYOUT_KEYS` of an unknown component.

        :param node_index: Zero-based row index into the snapshotted node data arrays.
        :return: The entry.
        """
        entry = {
            "x": round(float(self.node_data["x"][node_index]), _LAYOUT_DECIMALS),
            "y": round(float(self.node_data["y"][node_index]), _LAYOUT_DECIMALS),
        }
        if is_unknown_node(self.node_data, node_index):
            entry.update({key: self.node_data[key][node_index] for key in UNKNOWN_LAYOUT_KEYS})
        return entry

    def _saved_options(self, node_index: int) -> dict:
        """
        Options stored on the node, decoded from their JSON string.

        This already contains every option the user configured in the GUI,
        including boolean toggles (e.g. adjust_sfc) stored as booleans.

        :param node_index: Zero-based row index into the snapshotted node data arrays.
        :return: ``{option_name: value}``; empty if the JSON cannot be read.
        """
        raw_options_string = self.node_data["options"][node_index]
        try:
            return json.loads(raw_options_string) if raw_options_string else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def _known_id_and_options(self, node_index: int) -> tuple[str, dict]:
        """
        Component id and options of a known component, as FAST-OAD reads them.

        Options never touched in the GUI get their type's default, and a
        variable-port component gets its port counts as ``number_of_*`` options.

        :param node_index: Zero-based row index into the snapshotted node data arrays.
        :return: ``(component_id, options)``.
        """
        component_type = self.node_data["node_type"][node_index]
        known_component = _KNOWN_COMPONENT_BY_TYPE.get(component_type, {})
        options_dict = self._saved_options(node_index)

        # Fill in the per-component-type default of any option not already
        # present in the saved JSON (covers options that were never touched in
        # the GUI and therefore never written to placed_nodes_source["options"])
        for option_name, default_value in default_options(
            self.possible_options.get(component_type, {})
        ).items():
            options_dict.setdefault(option_name, default_value)

        # Variable-port components store their port counts as number_of_*
        # options, which FAST-OAD reads to build the topology
        if self.default_source_count.get(component_type, 0) == VARIABLE_PORT_DEFAULT:
            options_dict.update(
                _variable_port_options(
                    known_component.get("attributes") or [],
                    self.node_data["n_sources"][node_index],
                    self.node_data["n_targets"][node_index],
                )
            )
        return known_component["id"], options_dict

    def add_connection(self) -> None:
        """
        Append all edge connections to ``component_connections``.

        Iterates over every edge in the snapshotted edge data, determines
        which endpoint is the source and which is the target, and emits a
        ``{"source": …, "target": …}`` dict for each connection.

        For components with more than one port on a given side the port
        index is appended to the name using the list syntax ``[name, N]``
        as expected by FAST-OAD_CS23-HE.
        """
        for starting_port, ending_port in edge_end_ports(self.edge_data):
            # An edge starts at whichever port was clicked first: write it source -> target
            source_port, target_port = (
                (starting_port, ending_port)
                if starting_port[0] == "source"
                else (ending_port, starting_port)
            )
            self.data[KEY_PT_CONNECTIONS].append(
                {
                    "source": self._connection_endpoint(source_port, "n_sources"),
                    "target": self._connection_endpoint(target_port, "n_targets"),
                }
            )

    def _connection_endpoint(self, port: tuple, port_count_column: str):
        """
        One end of a connection as FAST-OAD_CS23-HE expects it.

        :param port: ``(kind, node_index, label)`` key of the port.
        :param port_count_column: ``"n_sources"`` or ``"n_targets"``, the port
            count of the port's side.
        :return: The component name, or ``"[name, N]"`` when the component has
            several ports on that side.
        """
        _, node_index, port_label = port
        component_name = self.node_data["name"][node_index]
        if self.node_data[port_count_column][node_index] > 1:
            return f"[{component_name}, {int(port_label)}]"
        return component_name

    def set_watcher_file_path(self, watcher_path: str) -> None:
        """
        Set the ``watcher_file_path`` field of the YAML output. The watcher file csv shows
        the changes of specified performance parameters for defined powertrain components during
        each time step of the flight mission.

        :param watcher_path: Filesystem path to the watcher CSV file, or an
            empty string to omit the field from the output.
        """
        self.data[KEY_PT_WATCHER] = watcher_path

    def _build_output(self) -> str:
        """
        Serialise and format the YAML content, returning it as a string.

        Post-processes the raw ruamel.yaml output to:

        * Remove quotes around ``[component_name, N]`` port-index patterns.
        * Collapse any block-scalar ``watcher_file_path`` back to a single
          inline assignment.
        * Insert a blank line before each top-level key (except the first).
        * Insert a blank line between each ``- source:`` connection entry.
        * Insert a blank line between each component under
          ``power_train_components``.
        * Leave the trailing ``canvas_layout`` section (omitted entirely when
          ``include_canvas_layout`` is ``False``) compact and prefix it with a
          short "GUI-only" comment.

        :return: The formatted YAML string ready to write to disk.
        """
        yaml_serialiser = YAML()
        yaml_serialiser.default_flow_style = False
        yaml_serialiser.indent(mapping=2, sequence=4, offset=2)

        string_buffer = StringIO()
        data_to_dump = {
            key: value
            for key, value in self.data.items()
            if not (key == KEY_CANVAS_LAYOUT and not self.include_canvas_layout)
            # watcher_file_path is optional for FAST-OAD: only write it when set
            and not (key == KEY_PT_WATCHER and not value)
        }
        yaml_serialiser.dump(data_to_dump, string_buffer)
        output_string = string_buffer.getvalue()

        # Remove quotes around [component_name, N] patterns
        output_string = re.sub(r"'(\[[^\]]+\])'", r"\1", output_string)

        # Fix watcher_file_path: ruamel.yaml may emit the value on the next line
        # as a block scalar. Collapse it back to a single inline assignment
        output_string = re.sub(
            r"(watcher_file_path:)\s*\n\s+(\S[^\n]*)",
            r"\1 \2",
            output_string,
        )

        # The GUI-only canvas_layout section is always dumped last (it is the
        # last key of self.data). Split it off so the spacing rules below only
        # apply to the FAST-OAD part of the file and the layout stays compact
        layout_marker = f"\n{KEY_CANVAS_LAYOUT}:"
        fastoad_part, marker, layout_part = output_string.partition(layout_marker)

        # Add a blank line before each top-level key (except the first one)
        top_level_keys = [KEY_TITLE, KEY_PT_COMPONENTS, KEY_PT_CONNECTIONS, KEY_PT_WATCHER]
        for key in top_level_keys[1:]:
            fastoad_part = fastoad_part.replace(f"\n{key}:", f"\n\n{key}:")

        # Add a blank line between each connection entry (- source:)
        fastoad_part = re.sub(r"\n(\s+- source:)", r"\n\n\1", fastoad_part)

        # Add a blank line between each component under power_train_components
        # (2-space-indented keys)
        fastoad_part = re.sub(r"\n(\s{2}\w)", r"\n\n\1", fastoad_part)

        if not marker:
            return fastoad_part

        # Re-attach the layout section, preceded by a blank line and a short
        # comment so readers of the YAML know it is GUI-only
        return fastoad_part + "\n\n" + _CANVAS_LAYOUT_HEADER + marker.lstrip("\n") + layout_part

    def write(self, file_path: str) -> None:
        """
        Write the formatted YAML content to a file on disk.

        :param file_path: Destination path for the YAML configuration file.
        """
        with open(file_path, "w", encoding="utf-8") as output_file:
            output_file.write(self._build_output())

    def write_to_stream(self, output_stream) -> None:
        """
        Write the formatted YAML content to an open file-like object.

        :param output_stream: A writable file-like object (e.g. ``io.StringIO``).
        """
        output_stream.write(self._build_output())


def _variable_port_options(
    component_attributes: list, source_port_count: int, target_port_count: int
) -> dict:
    """
    ``number_of_*`` options of a variable-port component, from its port counts.

    Each such component has exactly two ``number_of_`` attributes in
    ``KNOWN_COMPONENTS``, one per side (see :func:`port_side_of`).

    :param component_attributes: ``attributes`` of the component's ``KNOWN_COMPONENTS`` entry.
    :param source_port_count: Number of source ports on the canvas.
    :param target_port_count: Number of target ports on the canvas.
    :return: Dictionary mapping each ``number_of_*`` attribute to its port count.
    """
    port_counts = {"source": source_port_count, "target": target_port_count}
    return {
        attribute_name: port_counts[port_side_of(attribute_name)]
        for attribute_name in component_attributes
        if port_side_of(attribute_name) is not None
    }


def unknown_component_id(component_type: str) -> str:
    """
    FAST-OAD id written for an unknown component of type *component_type*.

    :param component_type: Type typed in the config panel, e.g. ``"my_part"``,
        or a full id such as ``"my_package.pt_component.my_part"``.
    :return: ``"fastga_he.pt_component.my_part"``, or the full id unchanged.
    """
    component_type = component_type or UNKNOWN_COMPONENT_TYPE
    return component_type if "." in component_type else COMPONENT_ID_PREFIX + component_type


def unknown_component_type(component_id) -> str:
    """
    Type shown for an unknown component read with id *component_id*.

    The inverse of :func:`unknown_component_id`: the FAST-OAD_CS23-HE prefix is
    dropped, any other id is kept whole.

    :param component_id: ``id`` of the component in the file (may be missing).
    :return: The component type to show and edit.
    """
    component_id = str(component_id or UNKNOWN_COMPONENT_TYPE)
    return component_id.removeprefix(COMPONENT_ID_PREFIX) or UNKNOWN_COMPONENT_TYPE


def port_side_of(attribute_name: str) -> str | None:
    """
    Port side whose count a variable-port attribute holds.

    The single rule shared by save (:func:`_variable_port_options`) and load
    (``PowerTrainYAMLReader._port_counts``), so both always agree: a
    ``number_of_*`` attribute with a keyword of :data:`SOURCE_SIDE_KEYWORDS`
    (e.g. inputs, tanks) counts the source ports, one with a keyword of
    :data:`TARGET_SIDE_KEYWORDS` (e.g. outputs, engines, power_sources) the
    target ports.

    :param attribute_name: An attribute name of a ``KNOWN_COMPONENTS`` entry.
    :return: ``"source"``, ``"target"``, or ``None`` for any other attribute.
    """
    if "number_of_" not in attribute_name:
        return None
    if any(keyword in attribute_name for keyword in SOURCE_SIDE_KEYWORDS):
        return "source"
    if any(keyword in attribute_name for keyword in TARGET_SIDE_KEYWORDS):
        return "target"
    return None
