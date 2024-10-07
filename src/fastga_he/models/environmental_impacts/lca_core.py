# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pathlib
import dotenv

from typing import Dict

import yaml

import pandas as pd
import numpy as np
import sympy as sym

import openmdao.api as om

import lca_algebraic as agb
from lcav.io.configuration import LCAProblemConfigurator

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

RESOURCE_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "resources"
METHODS_TO_FILE = {
    "EF v3.1 no LT": "ef_no_lt_methods.yml",
    "EF v3.1": "ef_methods.yml",
    "IMPACT World+ v2.0.1": "impact_world_methods.yml",
    "ReCiPe 2016 v1.03": "recipe_methods.yml",
}
NAME_TO_UNIT = {"mass": "kg", "length": "m"}


class LCACore(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = None
        self.methods = None
        self.axis = None
        self.axis_keys = None
        self.lambdas = None
        self.partial_lambdas_dict = None
        self.parameters = None
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

        self.add_output("dummy_output", val=0.0, units="kg")

        self.axis = self.options["axis"]

        _, self.model, self.methods = LCAProblemConfigurator(lca_conf_file_path).generate(
            reset=False
        )

        # noinspection PyProtectedMember
        self.lambdas = agb.lca._preMultiLCAAlgebric(self.model, self.methods, axis=self.axis)

        # Compile expressions for partial derivatives of impacts w.r.t. parameters
        self.partial_lambdas_dict = _preMultiLCAAlgebricPartials(
            self.model, self.methods, axis=self.options["axis"]
        )

        # Get axis keys to ventilate results by e.g. life-cycle phase
        self.axis_keys = self.lambdas[0].axis_keys

        # Retrieve LCA parameters declared in model, required to do it after loading the
        # configuration
        self.parameters = agb.all_params().values()

        for parameter in self.parameters:
            if parameter.type == "float":
                parameter_name = parameter.name.replace(
                    "__", ":"
                )  # refactor names (':' is not supported in LCA parameters)
                self.add_input(
                    parameter_name, val=np.nan, units=NAME_TO_UNIT[parameter_name.split(":")[-1]]
                )

    def setup_partials(self):
        # TODO: Change that, if only to see how much faster it is to properly declare partials !
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        parameters = {name.replace(":", "__"): value[0] for name, value in inputs.items()}

        res = self.compute_impacts_from_lambdas(**parameters)
        print(res)

        outputs["dummy_output"] = 2.0 * 1.0

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

        header = {"project": project_name}

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


def _preMultiLCAAlgebricPartials(model, methods, alpha=1, axis=None):
    """
    Modified version of _preMultiLCAAlgebric from lca_algebraic
    to compute partial derivatives of impacts w.r.t. parameters instead of expressions of impacts.
    """
    with agb.DbContext(model):
        # noinspection PyProtectedMember
        expressions = agb.lca._modelToExpr(model, methods, alpha=alpha, axis=axis)

        # Replace ceiling function by identity for better derivatives
        expressions = [expr.replace(sym.ceiling, lambda x: x) for expr in expressions]

        # Lambdify (compile) expressions
        if isinstance(expressions[0], agb.AxisDict):
            return {
                param.name: [
                    agb.lca.LambdaWithParamNames(
                        agb.AxisDict({axis_tag: res.diff(param) for axis_tag, res in expr.items()})
                    )
                    for expr in expressions
                ]
                for param in agb.all_params().values()
            }
        else:
            return {
                param.name: [agb.lca.LambdaWithParamNames(expr.diff(param)) for expr in expressions]
                for param in agb.all_params().values()
            }
