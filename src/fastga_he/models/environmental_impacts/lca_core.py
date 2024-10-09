# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pathlib
import dotenv

from typing import Dict, List

import yaml

import pandas as pd
import numpy as np
import sympy as sym

import re

import openmdao.api as om

import lca_algebraic as agb
import brightway2 as bw
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
LCA_PREFIX = "data:environmental_impact:"


class LCACore(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = None
        self.methods = None
        self.axis = None
        self.axis_keys_dict = None
        self.lambdas_dict = None
        self.partial_lambdas_dict_dict = None
        self.parameters = None

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
            name="component_level_breakdown",
            default=False,
            types=bool,
            desc="If true in addition to a breakdown, phase by phase, adds a breakdown component "
            "by component",
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

        self.axis = ["phase"]
        if self.options["component_level_breakdown"]:
            self.axis.append("component")

        _, self.model, self.methods = LCAProblemConfigurator(lca_conf_file_path).generate(
            reset=False
        )

        # noinspection PyProtectedMember
        self.lambdas_dict = {
            axis: agb.lca._preMultiLCAAlgebric(self.model, self.methods, axis=axis)
            for axis in self.axis
        }

        # Compile expressions for partial derivatives of impacts w.r.t. parameters
        self.partial_lambdas_dict_dict = {
            axis: self._preMultiLCAAlgebricPartials(self.model, self.methods, axis=axis)
            for axis in self.axis
        }

        # Get axis keys to ventilate results by e.g. life-cycle phase
        self.axis_keys_dict = {axis: self.lambdas_dict[axis][0].axis_keys for axis in self.axis}

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

        for m in self.methods:
            clean_method_name = re.sub(r": |/| ", "_", m[1])

            # "Phase" is always inside the axis, so we can do that
            for phase in self.axis_keys_dict["phase"]:
                if phase != "_other_":
                    # For each impact assessment method we give the impact by phase regardless of
                    # the case
                    self.add_output(
                        LCA_PREFIX + clean_method_name + ":" + phase + ":sum",
                        val=0.0,
                        units=None,
                        desc=bw.Method(m).metadata["unit"] + "for the whole " + phase + " phase",
                    )

            if "component" in self.axis:
                # Components are tagged with the phase just in case, so we can do this. However,
                # production seems like the only phase in which we can attribute impacts on the
                # component themselves. So it is actually a bit dangerous to do it like this.
                # Beware of only putting the "component" custom attribute only for production or for
                # phase in which the impacts can be attributed to the component themselves.
                for component_phase in self.axis_keys_dict["component"]:
                    if component_phase != "_other_":
                        # Now we give the value component by component
                        clean_phase_name = component_phase.split("_")[-1]
                        clean_component_name = "_".join(component_phase.split("_")[:-1])
                        self.add_output(
                            LCA_PREFIX
                            + clean_method_name
                            + ":"
                            + clean_phase_name
                            + ":"
                            + clean_component_name,
                            val=0.0,
                            units=None,
                            desc=bw.Method(m).metadata["unit"]
                            + "for component "
                            + clean_component_name
                            + " in "
                            + clean_phase_name
                            + " phase",
                        )

    def setup_partials(self):
        # Create a fake inputs list with constant value. It's not going to be relevant anyway
        # since we will use it here for the partials and the partials are constant given the
        # nature of LCA
        parameters = {str(name): 1.0 for name in self.parameters}

        # Then we run a "fake" computation of the partials, and if the value returned is nil,
        # we simply don't declare the partial
        for axis_to_evaluate in self.axis:
            partial_lambda_dict = self.partial_lambdas_dict_dict[axis_to_evaluate]

            # Compute partials from pre-compiled expressions and current parameters values
            res = {
                param_name: self.compute_impacts_from_lambdas(
                    partial_lambdas, axis_to_evaluate, **parameters
                )
                for param_name, partial_lambdas in partial_lambda_dict.items()
            }
            if axis_to_evaluate == "phase":
                for param_name, res_param in res.items():
                    for m in res_param:
                        clean_method_name = re.sub(r": |/| ", "_", m.split(" - ")[0])
                        input_name = param_name.replace("__", ":")
                        for phase in self.axis_keys_dict["phase"]:
                            if phase != "_other_":
                                partial_value = res_param[m][phase]
                                if partial_value != 0:
                                    self.declare_partials(
                                        of=LCA_PREFIX + clean_method_name + ":" + phase + ":sum",
                                        wrt=input_name,
                                        val=partial_value,
                                    )

            if axis_to_evaluate == "component":
                for param_name, res_param in res.items():
                    for m in res_param:
                        clean_method_name = re.sub(r": |/| ", "_", m.split(" - ")[0])
                        input_name = param_name.replace("__", ":")

                        for component_phase in self.axis_keys_dict["component"]:
                            if component_phase != "_other_":
                                # Now we give the value component by component
                                clean_phase_name = component_phase.split("_")[-1]
                                clean_component_name = "_".join(component_phase.split("_")[:-1])

                                partial_value = res_param[m][component_phase]
                                if partial_value != 0:
                                    self.declare_partials(
                                        of=LCA_PREFIX
                                        + clean_method_name
                                        + ":"
                                        + clean_phase_name
                                        + ":"
                                        + clean_component_name,
                                        wrt=input_name,
                                        val=partial_value,
                                    )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        parameters = {name.replace(":", "__"): value[0] for name, value in inputs.items()}

        for axis_to_evaluate in self.axis:
            res = self.compute_impacts_from_lambdas(
                lambdas=self.lambdas_dict[axis_to_evaluate], axis=axis_to_evaluate, **parameters
            )

            if axis_to_evaluate == "phase":
                for m in res:  # for each LCIA method
                    clean_method_name = re.sub(r": |/| ", "_", m.split(" - ")[0])

                    for phase in self.axis_keys_dict["phase"]:
                        if phase != "_other_":
                            outputs[LCA_PREFIX + clean_method_name + ":" + phase + ":sum"] = res[m][
                                phase
                            ]

            if axis_to_evaluate == "component":
                for m in res:  # for each LCIA method
                    clean_method_name = re.sub(r": |/| ", "_", m.split(" - ")[0])

                    # Components are tagged with the phase just in case, so we can do this
                    for component_phase in self.axis_keys_dict["component"]:
                        if component_phase != "_other_":
                            # Now we give the value component by component
                            clean_phase_name = component_phase.split("_")[-1]
                            clean_component_name = "_".join(component_phase.split("_")[:-1])
                            outputs[
                                LCA_PREFIX
                                + clean_method_name
                                + ":"
                                + clean_phase_name
                                + ":"
                                + clean_component_name
                            ] = res[m][component_phase]

    def compute_impacts_from_lambdas(
        self,
        lambdas: List[agb.LambdaWithParamNames],
        axis: str,
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
            df = agb.lca._postMultiLCAAlgebric(self.methods, lambdas, **params)

            # noinspection PyProtectedMember
            model_name = agb.base_utils._actName(self.model)
            while model_name in dfs:
                model_name += "'"

            # param with several values
            list_params = {k: vals for k, vals in params.items() if isinstance(vals, list)}

            # Shapes the output / index according to the axis or multi param entry
            if axis:
                df[axis] = lambdas[0].axis_keys
                df = df.set_index(axis)
                df.index.set_names([axis])

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

    @staticmethod
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
                            agb.AxisDict(
                                {axis_tag: res.diff(param) for axis_tag, res in expr.items()}
                            )
                        )
                        for expr in expressions
                    ]
                    for param in agb.all_params().values()
                }
            else:
                return {
                    param.name: [
                        agb.lca.LambdaWithParamNames(expr.diff(param)) for expr in expressions
                    ]
                    for param in agb.all_params().values()
                }
