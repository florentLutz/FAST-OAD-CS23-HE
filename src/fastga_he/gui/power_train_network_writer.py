# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

import json
import re
import logging
from io import StringIO
from ruamel.yaml import YAML


from .constants import MULTI_PORT_OPTIONS
from ..powertrain_builder.resources.registered_components import KNOWN_COMPONENTS


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

    def set_title(self, title):
        self.data["title"] = title

    def add_component(self, node_index):
        name = self.node_data["name"][node_index]
        component_type = self.node_data["node_type"][node_index]
        source_count = self.node_data["n_sources"][node_index]
        target_count = self.node_data["n_targets"][node_index]
        symetry = self.node_data["symmetry_name"][node_index]
        component_id = next(
            (comp for comp in KNOWN_COMPONENTS if comp["components_type"] == component_type), None
        )["id"]

        # Parse the options JSON string for this node into a dict
        raw_opts = self.node_data["options"][node_index]
        try:
            options_dict = json.loads(raw_opts) if raw_opts else {}
        except (json.JSONDecodeError, TypeError):
            options_dict = {}

        # Add multi-port options if the component type is in MULTI_PORT_OPTIONS
        if component_type in MULTI_PORT_OPTIONS:
            options_dict[MULTI_PORT_OPTIONS[component_type][0]] = source_count
            options_dict[MULTI_PORT_OPTIONS[component_type][1]] = target_count

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
        if "False" in self.source_data["connected"] or "False" in self.target_data["connected"]:
            _LOGGER.warning("Skipping connection because at least one port is not connected.")
            return  # Skip if any port is not connected

        for connection in range(len(self.edge_data["node_a_idx"])):
            start_port_name = self.node_data["name"][self.edge_data["node_a_idx"][connection]]
            end_port_name = self.node_data["name"][self.edge_data["node_b_idx"][connection]]
            start_port_component_type = self.node_data["node_type"][
                self.edge_data["node_a_idx"][connection]
            ]
            end_port_component_type = self.node_data["node_type"][
                self.edge_data["node_b_idx"][connection]
            ]

            if self.edge_data["a_kind"][connection] == "source":
                source = start_port_name
                source_count = self.node_data["n_sources"][self.edge_data["node_a_idx"][connection]]
                target = end_port_name
                target_count = self.node_data["n_targets"][self.edge_data["node_b_idx"][connection]]
                if source_count > 1 or start_port_component_type in MULTI_PORT_OPTIONS:
                    source = f"[{source}, {int(self.edge_data['a_label'][connection])}]"

                if target_count > 1 or end_port_component_type in MULTI_PORT_OPTIONS:
                    target = f"[{target}, {int(self.edge_data['b_label'][connection])}]"

            else:
                source = end_port_name
                source_count = self.node_data["n_sources"][self.edge_data["node_b_idx"][connection]]
                target = start_port_name
                target_count = self.node_data["n_targets"][self.edge_data["node_a_idx"][connection]]
                if source_count > 1 or end_port_component_type in MULTI_PORT_OPTIONS:
                    source = f"[{source}, {int(self.edge_data['b_label'][connection])}]"

                if target_count > 1 or start_port_component_type in MULTI_PORT_OPTIONS:
                    target = f"[{target}, {int(self.edge_data['a_label'][connection])}]"

            self.data["component_connections"].append({"source": source, "target": target})

    def set_watcher_file_path(self, path):
        self.data["watcher_file_path"] = path

    def write(self, filepath):
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.indent(mapping=2, sequence=4, offset=2)

        # Dump to a string buffer first
        buf = StringIO()
        yaml.dump(self.data, buf)
        output = buf.getvalue()

        # Remove quotes around [component_name, N] patterns
        output = re.sub(r"'(\[[^\]]+\])'", r"\1", output)

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

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(output)
