# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

"""
YAML serialiser for the powertrain network.

:class:`PowerTrainYAML` collects component and connection data from a
:class:`BuilderState` snapshot and writes a formatted YAML powertrain
configuration file, mirroring the schema consumed by FAST-OAD_CS23-HE.
"""

import json
import re
import logging
from io import StringIO
from ruamel.yaml import YAML

from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS


_LOGGER = logging.getLogger(__name__)


class PowerTrainYAML:
    """
    Assemble and write a YAML powertrain configuration from builder state data.

    The class reads node, edge, and port data from the live
    :class:`BuilderState` at construction time, so all subsequent calls to
    :meth:`add_component`, :meth:`add_connection`, and :meth:`write` work on
    an immutable snapshot.
    """

    def __init__(self, state):
        """
        Snapshot the builder state needed for serialisation.

        :param state: :class:`BuilderState` instance whose data sources are
            copied at construction time.
        """
        self.data = {
            "title": "",
            "power_train_components": {},
            "component_connections": [],
            "watcher_file_path": "",
        }
        self.node_data = {
            key: list(values) for key, values in state.placed_nodes_source.data.items()
        }
        self.edge_data = {key: list(values) for key, values in state.edge_source.data.items()}
        self.source_port_data = {
            key: list(values) for key, values in state.source_port_source.data.items()
        }
        self.target_port_data = {
            key: list(values) for key, values in state.target_port_source.data.items()
        }
        self.possible_options = state.possible_options
        self.default_source_count = state.default_source_count

    def set_title(self, title: str) -> None:
        """
        Set the ``title`` field of the YAML output.

        :param title: Human-readable title for the powertrain design.
        """
        self.data["title"] = title

    def add_component(self, node_index: int) -> None:
        """
        Append a single component entry to ``power_train_components``.

        Reads component metadata (name, type, port counts, symmetry, options)
        from the snapshotted node data at *node_index*, merges any unset
        options with per-type defaults, and writes the resulting dict into the
        YAML data structure.

        For variable-port components (those whose default source count is 3),
        the current port counts are written as ``number_of_*`` option keys so
        that FAST-OAD can reconstruct the correct topology.

        :param node_index: Zero-based row index into the snapshotted node data arrays.
        """
        component_name = self.node_data["name"][node_index]
        component_type = self.node_data["node_type"][node_index]
        source_port_count = self.node_data["n_sources"][node_index]
        target_port_count = self.node_data["n_targets"][node_index]
        symmetry_node_index = self.node_data["symmetry_node_index"][node_index]
        symmetry_component_name = (
            self.node_data["symmetry_name"][node_index]
            if symmetry_node_index < node_index
            else None
        )
        component_id = next(
            (comp for comp in KNOWN_COMPONENTS if comp["components_type"] == component_type), None
        )["id"]

        # Parse the options JSON string for this node into a dict.
        # This already contains every option the user configured in the GUI,
        # including boolean toggles (e.g. adjust_sfc) stored as booleans.
        raw_options_string = self.node_data["options"][node_index]
        try:
            options_dict = json.loads(raw_options_string) if raw_options_string else {}
        except (json.JSONDecodeError, TypeError):
            options_dict = {}

        # Merge in per-component-type option defaults for any key not already
        # present in the saved JSON (covers options that were never touched in
        # the GUI and therefore never written to placed_nodes_source["options"]).
        component_option_defaults = self.possible_options.get(component_type, {})
        for option_name, option_values in component_option_defaults.items():
            if option_name not in options_dict and option_values:
                options_dict[option_name] = option_values[0]

        # Add multi-port options for variable-port components (default count == 3).
        # Each such component has exactly two "number_of_" attributes in KNOWN_COMPONENTS:
        # one for the source side and one for the target side, in that order.
        # We assign source_port_count to the source-side attribute and
        # target_port_count to the target-side attribute, mirroring the convention
        # in _build_port_count_defaults.
        # Source-side keywords: inputs, tanks.
        # Target-side keywords: outputs, engines, power_sources.
        if self.default_source_count.get(component_type, 0) == 3:
            component_attributes = (
                next(
                    (
                        comp
                        for comp in KNOWN_COMPONENTS
                        if comp["components_type"] == component_type
                    ),
                    {},
                ).get("attributes")
                or []
            )
            _SOURCE_SIDE_KEYWORDS = ("_inputs", "_tanks")
            _TARGET_SIDE_KEYWORDS = ("_outputs", "_engines", "_power_sources")
            for attribute_name in component_attributes:
                if "number_of_" not in attribute_name:
                    continue
                if any(keyword in attribute_name for keyword in _SOURCE_SIDE_KEYWORDS):
                    options_dict[attribute_name] = source_port_count
                elif any(keyword in attribute_name for keyword in _TARGET_SIDE_KEYWORDS):
                    options_dict[attribute_name] = target_port_count

        if not options_dict:
            if not symmetry_component_name:
                self.data["power_train_components"][component_name] = {
                    "id": component_id,
                    "position": self.node_data["position"][node_index],
                }
            else:
                self.data["power_train_components"][component_name] = {
                    "id": component_id,
                    "position": self.node_data["position"][node_index],
                    "symmetrical": symmetry_component_name,
                }
        else:
            if not symmetry_component_name:
                self.data["power_train_components"][component_name] = {
                    "id": component_id,
                    "options": options_dict,
                    "position": self.node_data["position"][node_index],
                }
            else:
                self.data["power_train_components"][component_name] = {
                    "id": component_id,
                    "options": options_dict,
                    "position": self.node_data["position"][node_index],
                    "symmetrical": symmetry_component_name,
                }

    def add_connection(self) -> None:
        """
        Append all edge connections to ``component_connections``.

        Iterates over every edge in the snapshotted edge data, determines
        which endpoint is the source and which is the target, and emits a
        ``{"source": …, "target": …}`` dict for each connection.

        For components with more than one port on a given side the port
        index is appended to the name using the list syntax ``[name, N]``
        as expected by FAST-OAD_CS23-HE.

        Emits a warning to the logger if any port is still unconnected so
        the user is aware that the saved YAML may be incomplete.
        """
        # Warn (but do not abort) when any port is still unconnected so the
        # user knows the saved YAML may be incomplete. The "connected" column
        # stores the string "True" / "False", so we normalise with str().
        source_has_unconnected = any(
            str(value) == "False" for value in self.source_port_data.get("connected", [])
        )
        target_has_unconnected = any(
            str(value) == "False" for value in self.target_port_data.get("connected", [])
        )
        if source_has_unconnected or target_has_unconnected:
            _LOGGER.warning("At least one port is not connected; the saved YAML may be incomplete.")

        for connection_index in range(len(self.edge_data["starting_node_index"])):
            starting_node_name = self.node_data["name"][
                self.edge_data["starting_node_index"][connection_index]
            ]
            ending_node_name = self.node_data["name"][
                self.edge_data["ending_node_index"][connection_index]
            ]

            if self.edge_data["starting_port_kind"][connection_index] == "source":
                source_name = starting_node_name
                source_port_count = self.node_data["n_sources"][
                    self.edge_data["starting_node_index"][connection_index]
                ]
                target_name = ending_node_name
                target_port_count = self.node_data["n_targets"][
                    self.edge_data["ending_node_index"][connection_index]
                ]
                if source_port_count > 1:
                    source_name = f"[{source_name}, {int(self.edge_data['starting_port_label'][connection_index])}]"
                if target_port_count > 1:
                    target_name = f"[{target_name}, {int(self.edge_data['ending_port_label'][connection_index])}]"

            else:
                source_name = ending_node_name
                source_port_count = self.node_data["n_sources"][
                    self.edge_data["ending_node_index"][connection_index]
                ]
                target_name = starting_node_name
                target_port_count = self.node_data["n_targets"][
                    self.edge_data["starting_node_index"][connection_index]
                ]
                if source_port_count > 1:
                    source_name = f"[{source_name}, {int(self.edge_data['ending_port_label'][connection_index])}]"
                if target_port_count > 1:
                    target_name = f"[{target_name}, {int(self.edge_data['starting_port_label'][connection_index])}]"

            self.data["component_connections"].append(
                {"source": source_name, "target": target_name}
            )

    def set_watcher_file_path(self, watcher_path: str) -> None:
        """
        Set the ``watcher_file_path`` field of the YAML output.

        :param watcher_path: Filesystem path to the watcher CSV file, or an
            empty string to leave the field blank.
        """
        self.data["watcher_file_path"] = watcher_path

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

        :return: The formatted YAML string ready to write to disk.
        """
        yaml_serialiser = YAML()
        yaml_serialiser.default_flow_style = False
        yaml_serialiser.indent(mapping=2, sequence=4, offset=2)

        string_buffer = StringIO()
        yaml_serialiser.dump(self.data, string_buffer)
        output_string = string_buffer.getvalue()

        # Remove quotes around [component_name, N] patterns.
        output_string = re.sub(r"'(\[[^\]]+\])'", r"\1", output_string)

        # Fix watcher_file_path: ruamel.yaml may emit the value on the next line
        # as a block scalar. Collapse it back to a single inline assignment.
        output_string = re.sub(
            r"(watcher_file_path:)\s*\n\s+(\S[^\n]*)",
            r"\1 \2",
            output_string,
        )

        # Add a blank line before each top-level key (except the first one).
        top_level_keys = [
            "title",
            "power_train_components",
            "component_connections",
            "watcher_file_path",
        ]
        for key in top_level_keys[1:]:
            output_string = output_string.replace(f"\n{key}:", f"\n\n{key}:")

        # Add a blank line between each connection entry (- source:).
        output_string = re.sub(r"\n(\s+- source:)", r"\n\n\1", output_string)

        # Add a blank line between each component under power_train_components
        # (2-space-indented keys).
        output_string = re.sub(r"\n(\s{2}\w)", r"\n\n\1", output_string)

        return output_string

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
