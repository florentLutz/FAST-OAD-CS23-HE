# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import json
import re
import logging
from io import StringIO
from ruamel.yaml import YAML

from fastga_he.powertrain_builder.resources.registered_components import KNOWN_COMPONENTS


_LOGGER = logging.getLogger(__name__)


class PowerTrainYAML:
    def __init__(self, state):
        self.data = {
            "title": "",
            "power_train_components": {},
            "component_connections": [],
            "watcher_file_path": "",
        }
        self.node_data = {k: list(v) for k, v in state.placed_nodes_source.data.items()}
        self.edge_data = {k: list(v) for k, v in state.edge_source.data.items()}
        self.source_data = {k: list(v) for k, v in state.source_port_source.data.items()}
        self.target_data = {k: list(v) for k, v in state.target_port_source.data.items()}
        self.possible_options = state.possible_options
        self.default_source_count = state.default_source_count

    def set_title(self, title):
        self.data["title"] = title

    def add_component(self, node_index):
        name = self.node_data["name"][node_index]
        component_type = self.node_data["node_type"][node_index]
        source_count = self.node_data["n_sources"][node_index]
        target_count = self.node_data["n_targets"][node_index]
        symmetry_node_index = self.node_data["symmetry_node_index"][node_index]
        symetry = (
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
        raw_opts = self.node_data["options"][node_index]
        try:
            options_dict = json.loads(raw_opts) if raw_opts else {}
        except (json.JSONDecodeError, TypeError):
            options_dict = {}

        # Merge in per-component-type option defaults for any key not already
        # present in the saved JSON (covers options that were never touched in
        # the GUI and therefore never written to placed_nodes_source["options"]).
        comp_opts_def = self.possible_options.get(component_type, {})
        for opt_name, opt_values in comp_opts_def.items():
            if opt_name not in options_dict and opt_values:
                options_dict[opt_name] = opt_values[0]

        # Add multi-port options for variable-port components (default count == 3).
        # Each such component has exactly two "number_of_" attributes in KNOWN_COMPONENTS:
        # one for the source side and one for the target side, in that order.
        # We assign source_count to the source-side attribute and target_count to the
        # target-side attribute, mirroring the convention in _build_port_count_defaults.
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
            _SOURCE_KEYWORDS = ("_inputs", "_tanks")
            _TARGET_KEYWORDS = ("_outputs", "_engines", "_power_sources")
            for attr in component_attributes:
                if "number_of_" not in attr:
                    continue
                if any(keyword in attr for keyword in _SOURCE_KEYWORDS):
                    options_dict[attr] = source_count
                elif any(keyword in attr for keyword in _TARGET_KEYWORDS):
                    options_dict[attr] = target_count

        if not options_dict:
            if not symetry:
                self.data["power_train_components"][name] = {
                    "id": component_id,
                    "position": self.node_data["position"][node_index],
                }
            else:
                self.data["power_train_components"][name] = {
                    "id": component_id,
                    "position": self.node_data["position"][node_index],
                    "symmetrical": symetry,
                }
        else:
            if not symetry:
                self.data["power_train_components"][name] = {
                    "id": component_id,
                    "options": options_dict,
                    "position": self.node_data["position"][node_index],
                }

            else:
                self.data["power_train_components"][name] = {
                    "id": component_id,
                    "options": options_dict,
                    "position": self.node_data["position"][node_index],
                    "symmetrical": symetry,
                }

    def add_connection(self):
        # Warn (but do not abort) when any port is still unconnected so the
        # user knows the saved YAML may be incomplete.  The "connected" column
        # stores the string "True"/"False", so we normalise with str().
        unconnected_src = any(str(v) == "False" for v in self.source_data.get("connected", []))
        unconnected_tgt = any(str(v) == "False" for v in self.target_data.get("connected", []))
        if unconnected_src or unconnected_tgt:
            _LOGGER.warning("At least one port is not connected; the saved YAML may be incomplete.")

        for connection in range(len(self.edge_data["node_a_idx"])):
            start_port_name = self.node_data["name"][self.edge_data["node_a_idx"][connection]]
            end_port_name = self.node_data["name"][self.edge_data["node_b_idx"][connection]]

            if self.edge_data["a_kind"][connection] == "source":
                source = start_port_name
                source_count = self.node_data["n_sources"][self.edge_data["node_a_idx"][connection]]
                target = end_port_name
                target_count = self.node_data["n_targets"][self.edge_data["node_b_idx"][connection]]
                if source_count > 1:
                    source = f"[{source}, {int(self.edge_data['a_label'][connection])}]"
                if target_count > 1:
                    target = f"[{target}, {int(self.edge_data['b_label'][connection])}]"

            else:
                source = end_port_name
                source_count = self.node_data["n_sources"][self.edge_data["node_b_idx"][connection]]
                target = start_port_name
                target_count = self.node_data["n_targets"][self.edge_data["node_a_idx"][connection]]
                if source_count > 1:
                    source = f"[{source}, {int(self.edge_data['b_label'][connection])}]"
                if target_count > 1:
                    target = f"[{target}, {int(self.edge_data['a_label'][connection])}]"

            self.data["component_connections"].append({"source": source, "target": target})

    def set_watcher_file_path(self, path):
        self.data["watcher_file_path"] = path

    def _build_output(self) -> str:
        """Serialise and format the YAML content, returning it as a string."""
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.indent(mapping=2, sequence=4, offset=2)

        # Dump to a string buffer first
        buf = StringIO()
        yaml.dump(self.data, buf)
        output = buf.getvalue()

        # Remove quotes around [component_name, N] patterns
        output = re.sub(r"'(\[[^\]]+\])'", r"\1", output)

        # Fix watcher_file_path: ruamel.yaml may emit the value on the next line
        # as a block scalar. Collapse it back to a single inline assignment.
        output = re.sub(
            r"(watcher_file_path:)\s*\n\s+(\S[^\n]*)",
            r"\1 \2",
            output,
        )

        # Add a blank line before each top-level key (except the first one)
        top_level_keys = [
            "title",
            "power_train_components",
            "component_connections",
            "watcher_file_path",
        ]
        for key in top_level_keys[1:]:
            output = output.replace(f"\n{key}:", f"\n\n{key}:")

        # Add blank line between each connection entry (- source:)
        output = re.sub(r"\n(\s+- source:)", r"\n\n\1", output)

        # Add blank line between each component under power_train_components (2-space indented keys)
        output = re.sub(r"\n(\s{2}\w)", r"\n\n\1", output)

        return output

    def write(self, filepath):
        """Write the formatted YAML content to a file on disk."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self._build_output())

    def write_to_stream(self, stream) -> None:
        """Write the formatted YAML content to an open file-like object (e.g. io.StringIO)."""
        stream.write(self._build_output())
