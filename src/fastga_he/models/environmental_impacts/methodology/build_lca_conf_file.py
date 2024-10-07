# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import yaml
import pathlib

RESOURCE_FOLDER_PATH = pathlib.Path(__file__).parents[0].parents[0] / "resources"
HEADER_PATH = RESOURCE_FOLDER_PATH / "lca_conf_header.yml"
DUMMY_PATH = RESOURCE_FOLDER_PATH / "dummy_built.yml"
METHODS_TO_FILE = {
    "EF v3.1 no LT": "ef_no_lt_methods.yml",
    "EF v3.1": "ef_methods.yml",
    "IMPACT World+ v2.0.1": "impact_wolrd_methods.yml",
    "ReCiPe 2016 v1.03": "recipe_methods.yml",
}


def write_methods(path_to_yaml: pathlib.Path, dict_with_methods):
    with open(path_to_yaml, "a") as my_file:
        my_file.write("methods:\n")
        for method in dict_with_methods["methods"]:
            my_file.write('    - "' + method + '"\n')


if __name__ == "__main__":
    project_name = "dummy_project"
    eco_invent_version = "3.9.1"
    methods = "ReCiPe 2016 v1.03"

    header = {}
    header["project"] = project_name
    header["ecoinvent"] = {"version": eco_invent_version, "model": "cutoff"}

    methods_file = RESOURCE_FOLDER_PATH / METHODS_TO_FILE[methods]

    with open(methods_file, "r") as methods_file_stream:
        methods_dict = yaml.safe_load(methods_file_stream)

    with open(DUMMY_PATH, "w") as new_file:
        yaml.safe_dump(header, new_file)

    write_methods(DUMMY_PATH, methods_dict)
