# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2026 ISAE-SUPAERO

from ruamel.yaml import YAML

class PowerTrainYAML:
    def __init__(self):
        self.data = {
            "title": "",
            "power_train_components": {},
            "component_connections": [],
            "watcher_file_path": ""
        }

    def set_title(self, title):
        self.data["title"] = title

    def add_component(self, name, component_data):
        self.data["power_train_components"][name] = component_data

    def add_connection(self, source, target):
        self.data["component_connections"].append({
            "source": source,
            "target": target
        })

    def set_watcher_file_path(self, path):
        self.data["watcher_file_path"] = path