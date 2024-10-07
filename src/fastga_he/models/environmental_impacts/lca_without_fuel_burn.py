# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pathlib
import dotenv

from typing import Dict

import yaml

import pandas as pd

import openmdao.api as om

import lca_algebraic as agb
from lcav.io.configuration import LCAProblemConfigurator

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

RESOURCE_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "resources"
METHODS_TO_FILE = {
    "EF v3.1 no LT": "ef_no_lt_methods.yml",
    "EF v3.1": "ef_methods.yml",
    "IMPACT World+ v2.0.1": "impact_wolrd_methods.yml",
    "ReCiPe 2016 v1.03": "recipe_methods.yml",
}


class LCAWithoutFuelBurn(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = None
        self.methods = None
        self.axis = None
        self.lambdas = None
        self.params_names = None
        self.params_dict = {}

        self.configurator = FASTGAHEPowerTrainConfigurator()

        # Seems required to do it here
        dotenv.load_dotenv()

    def initialize(self):
        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the power",
            allow_none=False,
        )
        self.options.declare(
            name="axis",
            default="phase",
            desc="?",
            values=["phase"],
        )
        self.options.declare(
            name="impact_assessment_method",
            default="ReCiPe 2016 v1.03",
            desc="Impact assessment method to be used",
            values=list(METHODS_TO_FILE.keys()),
        )
        self.options.declare(
            name="ecoinvent_version",
            default="3.9.1",
            desc="EcoInvent version to use",
            values=["3.9.1"],
        )

    def setup(self):
        self.configurator.load(self.options["power_train_file_path"])
        lca_conf_file_path = self.write_lca_conf_file()

        self.add_input("dummy_input", val=1.0, units="kg")

        self.add_output("dummy_output", val=0.0, units="kg")

        conf_file_path = RESOURCE_FOLDER_PATH / "dummy_conf.yml"

        self.axis = self.options["axis"]

        _, self.model, self.methods = LCAProblemConfigurator(conf_file_path).generate(reset=False)

        # noinspection PyProtectedMember
        self.lambdas = agb.lca._preMultiLCAAlgebric(self.model, self.methods, axis=self.axis)

        # Required to do it after loading the configuration
        self.params_names = agb.all_params().keys()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        params = {
            "aircraft_production_long_range": 1,
            "energy_consumption_kerosene": 8,
            "lhv_kerosene": 3,
        }

        res = self.compute_impacts_from_lambdas(**params)

        outputs["dummy_output"] = 2.0 * inputs["dummy_input"]

    def compute_impacts_from_lambdas(
        self,
        **params: Dict[str, agb.SingleOrMultipleFloat],
    ):
        """
        Modified version of compute_impacts from lca_algebraic.
        More like a wrapper of _postLCAAlgebraic, to avoid calling _preLCAAlgebraic which is
        unnecessarily time-consuming when lambdas have already been calculated and doesn't have to
        be updated.
        """
        dfs = dict()

        dbname = self.model.key[0]
        with agb.DbContext(dbname):
            # Check no params are passed for FixedParams
            for key in params:
                # noinspection PyProtectedMember
                if key in agb.params._fixed_params():
                    print("Param '%s' is marked as FIXED, but passed in parameters : ignored" % key)

            # this is the time-consuming part
            # lambdas = _preMultiLCAAlgebric(model, methods, alpha=alpha, axis=axis)

            # noinspection PyProtectedMember
            df = agb.lca._postMultiLCAAlgebric(self.methods, self.lambdas, **params)

            # noinspection PyProtectedMember
            model_name = agb.base_utils._actName(self.model)
            while model_name in dfs:
                model_name += "'"

            # param with several values
            list_params = {k: vals for k, vals in params.items() if isinstance(vals, list)}

            # Shapes the output / index according to the axis or multi param entry
            if self.axis:
                df[self.axis] = self.lambdas[0].axis_keys
                df = df.set_index(self.axis)
                df.index.set_names([self.axis])

                # Filter out line with zero output
                df = df.loc[
                    df.apply(
                        lambda row: not (row.name is None and row.values[0] == 0.0),
                        axis=1,
                    )
                ]

                # Rename "None" to others
                df = df.rename(index={None: "_other_"})

                # Sort index
                df.sort_index(inplace=True)

                # Add "total" line
                df.loc["*sum*"] = df.sum(numeric_only=True)

            elif len(list_params) > 0:
                for k, vals in list_params.items():
                    df[k] = vals
                df = df.set_index(list(list_params.keys()))

            else:
                # Single output ? => give the single row the name of the model activity
                df = df.rename(index={0: model_name})

            dfs[model_name] = df

        if len(dfs) == 1:
            df = list(dfs.values())[0]
        else:
            # Concat several dataframes for several models
            df = pd.concat(list(dfs.values()))

        return df

    @staticmethod
    def write_ecoinvent_version(path_to_yaml: pathlib.Path, ecoinvent_version: str):
        with open(path_to_yaml, "a") as my_file:
            my_file.write("\n")
            my_file.write("ecoinvent:\n")
            my_file.write("    version: " + ecoinvent_version + "\n")
            my_file.write("    model: cutoff\n")

    @staticmethod
    def write_methods(path_to_yaml: pathlib.Path, dict_with_methods):
        with open(path_to_yaml, "a") as my_file:
            my_file.write("\n")
            my_file.write("methods:\n")
            for method in dict_with_methods["methods"]:
                my_file.write('    - "' + method + '"\n')

    @staticmethod
    def write_production(path_to_yaml: pathlib.Path, dict_with_production):
        with open(path_to_yaml, "a") as my_file:
            my_file.write("\n")
            my_file.write("model:\n")
            my_file.write("    production:\n")
            my_file.write("        custom_attributes:\n")
            my_file.write('            - attribute: "phase"\n')
            my_file.write('              value: "production"\n')
            my_file.write("\n")

            for component_name in dict_with_production.keys():
                for line_to_copy in dict_with_production[component_name]:
                    my_file.write("        " + line_to_copy)
                my_file.write("\n")

    def write_lca_conf_file(self):
        ecoinvent_version = self.options["ecoinvent_version"]
        methods = self.options["impact_assessment_method"]

        power_train_file_path = self.options["power_train_file_path"]
        parent_folder = power_train_file_path.parents[0]
        power_train_file_name = power_train_file_path.name
        project_name = power_train_file_name.replace(".yml", "")
        lca_conf_file_name = power_train_file_name.replace(".yml", "_lca.yml")
        lca_conf_file_path = parent_folder / lca_conf_file_name

        header = {}
        header["project"] = project_name

        with open(lca_conf_file_path, "w") as new_file:
            yaml.safe_dump(header, new_file)

        self.write_ecoinvent_version(lca_conf_file_path, ecoinvent_version)

        dict_with_production = self.configurator.get_lca_production_element_list()

        # This function writes the "model: " in the yml so it must be run before anything else
        # that need to go in the model section in the model section
        self.write_production(lca_conf_file_path, dict_with_production)

        methods_file = RESOURCE_FOLDER_PATH / METHODS_TO_FILE[methods]

        with open(methods_file, "r") as methods_file_stream:
            methods_dict = yaml.safe_load(methods_file_stream)

        self.write_methods(lca_conf_file_path, methods_dict)

        return lca_conf_file_path
