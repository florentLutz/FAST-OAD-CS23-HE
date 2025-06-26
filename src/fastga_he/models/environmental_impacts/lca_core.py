# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import pathlib
import re
import shutil
import logging
from typing import Dict, List

import brightway2 as bw
import dotenv
import lca_algebraic as agb
import numpy as np
import openmdao.api as om
import pandas as pd
import sympy as sym
import yaml
from lca_modeller.io.configuration import LCAProblemConfigurator

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator
from .resources.constants import METHODS_TO_FILE, LCA_PREFIX

RESOURCE_FOLDER_PATH = pathlib.Path(__file__).parents[0] / "resources"

NAME_TO_UNIT = {
    "mass": "kg",
    "length": "m",
    "OWE": "kg",
    "energy": "W*h",
    "cargo_transport": "t*km",
    "material": None,
}

_LOGGER = logging.getLogger(__name__)


class LCACore(om.ExplicitComponent):
    # Cache for storing LCA model, methods and lambdas.
    # This avoids recompiling everything if already done in previous setup of FAST-OAD
    _cache = {}

    # Cache for storing if a file with identical content has already been written or not. Avoid
    # rewriting an identical file (which changes time of last modification, which makes it so that
    # the LCA cache counts a change).
    _cache_file = {}

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

        self.name_to_unit = NAME_TO_UNIT

        self.outputs_list = []
        self.clean_method_name = set()

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
            values=["3.9.1", "3.10.1"],
        )
        self.options.declare(
            name="airframe_material",
            default="aluminium",
            desc="Material used for the airframe which include wing, fuselage, HTP and VTP. LG will"
            " always be in aluminium and flight controls in steel",
            allow_none=False,
            values=["aluminium", "composite"],
        )
        self.options.declare(
            name="delivery_method",
            default="flight",
            desc="Method with which the aircraft will be brought from the assembly plant to the "
            "end user. Can be either flown or carried by train",
            allow_none=False,
            values=["flight", "train"],
        )
        self.options.declare(
            name="electric_mix",
            default="default",
            desc="By default to construct the aircraft, a European electric mix is used. This "
            "forces all higher level process to use a different mix. This will not affect "
            "subprocesses of proxies directly taken from EcoInvent",
            allow_none=False,
            values=["default", "french", "slovenia"],
        )
        self.options.declare(
            name="write_lca_conf",
            default=True,
            types=bool,
            desc="By default the code will write a new configuration file for the LCA in the same "
            "folder as the powertrain file at each setup of the LCA module. This can be "
            "turned off if an LCA file is already available",
        )
        self.options.declare(
            name="lca_conf_file_path",
            default="",
            types=(str, pathlib.Path),
            desc="If an existing LCA configuration file is to be used, its path can be provided "
            "here. If nothing is put for this option, the code will assume it is located in "
            "the same folder as the powertrain file and will have the same name except for a "
            "_lca suffix",
        )

    @staticmethod
    def _check_existing_instance(lca_conf_file_path: pathlib.Path):
        """
        Checks the cache to see if an instance of the cache already exists and is usable. Usable
        means there was no modification to the LCA configuration file.
        """

        # If cache is empty, no instance is usable
        if not LCACore._cache:
            return False

        key = str(lca_conf_file_path)

        # If cache is not empty but there is no instance of that particular configuration file, no
        # instance is usable.
        if key not in LCACore._cache:
            return False

        # Finally, if an instance exists, but it has been modified since, no instance is usable.
        if LCACore._cache[key]["last_mod_time"] < lca_conf_file_path.lstat().st_mtime:
            return False

        return True

    @staticmethod
    def _get_cache_instance(lca_conf_file_path: pathlib.Path):
        """
        Access the latest usable instance of the cache that has been registered for this LCA
        configuration file.
        """

        key = str(lca_conf_file_path)

        cache_instance = LCACore._cache[key]

        return (
            cache_instance["model"],
            cache_instance["methods"],
            cache_instance["lambdas_dict"],
            cache_instance["partial_lambdas_dict_dict"],
        )

    @staticmethod
    def _add_cache_instance(
        lca_conf_file_path, model, methods, lambdas_dict, partial_lambdas_dict_dict
    ):
        """
        In the case where no instance were usable and the compilitation needed to be redone, we add
        said compilation to the cache.
        """

        key = str(lca_conf_file_path)

        cache_instance = {
            "last_mod_time": lca_conf_file_path.lstat().st_mtime,
            "model": model,
            "methods": methods,
            "lambdas_dict": lambdas_dict,
            "partial_lambdas_dict_dict": partial_lambdas_dict_dict,
        }
        LCACore._cache[key] = cache_instance

    def setup(self):
        self.configurator.load(self.options["power_train_file_path"])
        lca_conf_file_path = self.write_lca_conf_file()

        self.axis = ["phase"]
        if self.options["component_level_breakdown"]:
            self.axis.append("component")

        # If a usable instance exist, we use it. Otherwise, we create and cache it.
        if self._check_existing_instance(lca_conf_file_path):
            self.model, self.methods, self.lambdas_dict, self.partial_lambdas_dict_dict = (
                self._get_cache_instance(lca_conf_file_path)
            )
            _LOGGER.info("Loading cached data for LCA")

        else:
            _LOGGER.info(
                "LCA module: No cache found or configuration file has been modified. "
                "Compiling LCA model and functions."
            )

            _, self.model, self.methods = LCAProblemConfigurator(lca_conf_file_path).generate()

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

            self._add_cache_instance(
                lca_conf_file_path,
                self.model,
                self.methods,
                self.lambdas_dict,
                self.partial_lambdas_dict_dict,
            )

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
                    parameter_name,
                    val=np.nan,
                    units=self.name_to_unit[parameter_name.split(":")[-1].replace("_per_fu", "")],
                )

        for m in self.methods:
            clean_method_name = re.sub(r": |/| ", "_", m[1])
            clean_method_name = clean_method_name.replace(",_", "")
            self.clean_method_name.add(clean_method_name)

            # "Phase" is always inside the axis, so we can do that
            for phase in self.axis_keys_dict["phase"]:
                if phase != "_other_":
                    # For each impact assessment method we give the impact by phase regardless of
                    # the case
                    self.add_output(
                        LCA_PREFIX + clean_method_name + ":" + phase + ":sum",
                        val=0.0,
                        units=None,
                        desc=bw.Method(m).metadata["unit"] + " for the whole " + phase + " phase",
                    )
                    self.outputs_list.append(LCA_PREFIX + clean_method_name + ":" + phase + ":sum")

            self.add_output(
                LCA_PREFIX + clean_method_name + ":sum",
                val=0.0,
                units=None,
                desc=bw.Method(m).metadata["unit"] + " for the whole process",
            )
            self.outputs_list.append(LCA_PREFIX + clean_method_name + ":sum")

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
                            + " for component "
                            + clean_component_name
                            + " in "
                            + clean_phase_name
                            + " phase",
                        )
                        self.outputs_list.append(
                            LCA_PREFIX
                            + clean_method_name
                            + ":"
                            + clean_phase_name
                            + ":"
                            + clean_component_name
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
                        clean_method_name = clean_method_name.replace(",_", "")
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

                        partial_value = res_param[m]["*sum*"]
                        if partial_value != 0:
                            self.declare_partials(
                                of=LCA_PREFIX + clean_method_name + ":sum",
                                wrt=input_name,
                                val=partial_value,
                            )

            if axis_to_evaluate == "component":
                for param_name, res_param in res.items():
                    for m in res_param:
                        clean_method_name = re.sub(r": |/| ", "_", m.split(" - ")[0])
                        clean_method_name = clean_method_name.replace(",_", "")
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
                    clean_method_name = clean_method_name.replace(",_", "")

                    for phase in self.axis_keys_dict["phase"]:
                        if phase != "_other_":
                            outputs[LCA_PREFIX + clean_method_name + ":" + phase + ":sum"] = res[m][
                                phase
                            ]

                    outputs[LCA_PREFIX + clean_method_name + ":sum"] = res[m]["*sum*"]

            if axis_to_evaluate == "component":
                for m in res:  # for each LCIA method
                    clean_method_name = re.sub(r": |/| ", "_", m.split(" - ")[0])
                    clean_method_name = clean_method_name.replace(",_", "")

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

    def write_production(self, path_to_yaml: pathlib.Path, dict_with_production):
        with open(path_to_yaml, "a") as my_file:
            my_file.write("\n")
            my_file.write("model:\n")
            my_file.write("    production:\n")
            my_file.write("        custom_attributes:\n")
            my_file.write('            - attribute: "phase"\n')
            my_file.write('              value: "production"\n')
            my_file.write("\n")

            # Write the production of the aircraft.
            airframe_material = self.options["airframe_material"]
            if airframe_material == "aluminium":
                airframe_material_str = "market for aluminium, cast alloy"
            else:
                airframe_material_str = (
                    "market for carbon fibre reinforced plastic, injection moulded"
                )

            path_to_airframe_conf_file = RESOURCE_FOLDER_PATH / "lca_conf_airframe.yml"
            with open(path_to_airframe_conf_file, "r") as airframe_conf:
                for line_to_copy in airframe_conf.readlines():
                    line_to_copy = line_to_copy.replace(
                        "ANCHOR_AIRFRAME_MATERIAL", airframe_material_str
                    )
                    my_file.write("        " + line_to_copy)
                my_file.write("\n")

            for component_name in dict_with_production.keys():
                for line_to_copy in dict_with_production[component_name]:
                    my_file.write("        " + line_to_copy)
                my_file.write("\n")

    def write_manufacturing(self, path_to_yaml: pathlib.Path, dict_with_manufacturing):
        """
        Copy of the docstring of the get_lca_manufacturing_phase_element_list() method

        Theoretically, the manufacturing contains the assembly of the airframe plus tests
        plus the construction of the assembly plant. In our case, the assembly plant will be
        discarded and because we lack data, the assembly of the airframe has been aggregated in the
        production. So all that remains are the line tests, which will be very similar to the use
        phase. Except we won't attribute emission to each component (which means not adding the
        custom attributes), however we'll need to differentiate each CO2, NOx emissions ... so we'll
        tag them with component name.
        """
        with open(path_to_yaml, "a") as my_file:
            my_file.write("\n")
            my_file.write("    manufacturing:\n")
            my_file.write("        custom_attributes:\n")
            my_file.write('            - attribute: "phase"\n')
            my_file.write('              value: "manufacturing"\n')
            my_file.write("\n")

            for component_name in dict_with_manufacturing.keys():
                for line_to_copy in dict_with_manufacturing[component_name]:
                    my_file.write("        " + line_to_copy)
                my_file.write("\n")

            battery_names, _ = self.configurator.get_battery_list()

            if battery_names:
                path_to_electricity_prod_file = RESOURCE_FOLDER_PATH / "electricity_production.yml"
                with open(path_to_electricity_prod_file, "r") as electricity_prod_conf:
                    lines_to_copy = electricity_prod_conf.readlines()
                    for idx, line_to_copy in enumerate(lines_to_copy):
                        if not self.configurator.belongs_to_custom_attribute_definition(
                            line_to_copy, idx, lines_to_copy
                        ):
                            line_to_add = line_to_copy.replace("__operation__", "__manufacturing__")
                            my_file.write("        " + line_to_add)
                    my_file.write("\n")

            tank_names, tank_types, contents = self.configurator.get_fuel_tank_list_and_fuel()

            if tank_names:
                if "jet_fuel" in contents:
                    path_to_kerosene_prod_file = RESOURCE_FOLDER_PATH / "kerosene_production.yml"
                    with open(path_to_kerosene_prod_file, "r") as kerosene_prod_conf:
                        lines_to_copy = kerosene_prod_conf.readlines()
                        for idx, line_to_copy in enumerate(lines_to_copy):
                            if not self.configurator.belongs_to_custom_attribute_definition(
                                line_to_copy, idx, lines_to_copy
                            ):
                                line_to_add = line_to_copy.replace(
                                    "__operation__", "__manufacturing__"
                                )
                                my_file.write("        " + line_to_add)
                        my_file.write("\n")

                if "avgas" in contents:
                    path_to_gasoline_prod_file = RESOURCE_FOLDER_PATH / "gasoline_production.yml"
                    with open(path_to_gasoline_prod_file, "r") as gasoline_prod_conf:
                        lines_to_copy = gasoline_prod_conf.readlines()
                        for idx, line_to_copy in enumerate(lines_to_copy):
                            if not self.configurator.belongs_to_custom_attribute_definition(
                                line_to_copy, idx, lines_to_copy
                            ):
                                line_to_add = line_to_copy.replace(
                                    "__operation__", "__manufacturing__"
                                )
                                my_file.write("        " + line_to_add)
                        my_file.write("\n")

    def write_distribution(self, path_to_yaml: pathlib.Path, dict_with_distribution):
        """
        Writes in the LCA configuration file, the steps necessary to evaluate the impact of the
        distribution step. What we will write will depend on the method selected. If the aircraft
        is distributed via the air, it will be very similar to what is done for the use and
        manufacturing (line tests) steps.
        """

        with open(path_to_yaml, "a") as my_file:
            my_file.write("\n")
            my_file.write("    distribution:\n")
            my_file.write("        custom_attributes:\n")
            my_file.write('            - attribute: "phase"\n')
            my_file.write('              value: "distribution"\n')
            my_file.write("\n")

            if self.options["delivery_method"] == "flight":
                for component_name in dict_with_distribution.keys():
                    for line_to_copy in dict_with_distribution[component_name]:
                        my_file.write("        " + line_to_copy)
                    my_file.write("\n")

                battery_names, _ = self.configurator.get_battery_list()

                # TODO: the following s very long and could probably be automated/refactored
                if battery_names:
                    path_to_electricity_prod_file = (
                        RESOURCE_FOLDER_PATH / "electricity_production.yml"
                    )
                    with open(path_to_electricity_prod_file, "r") as electricity_prod_conf:
                        lines_to_copy = electricity_prod_conf.readlines()
                        for idx, line_to_copy in enumerate(lines_to_copy):
                            if not self.configurator.belongs_to_custom_attribute_definition(
                                line_to_copy, idx, lines_to_copy
                            ):
                                line_to_add = line_to_copy.replace(
                                    "__operation__", "__distribution__"
                                )
                                my_file.write("        " + line_to_add)
                        my_file.write("\n")

                tank_names, tank_types, contents = self.configurator.get_fuel_tank_list_and_fuel()

                if tank_names:
                    if "jet_fuel" in contents:
                        path_to_kerosene_prod_file = (
                            RESOURCE_FOLDER_PATH / "kerosene_production.yml"
                        )
                        with open(path_to_kerosene_prod_file, "r") as kerosene_prod_conf:
                            lines_to_copy = kerosene_prod_conf.readlines()
                            for idx, line_to_copy in enumerate(lines_to_copy):
                                if not self.configurator.belongs_to_custom_attribute_definition(
                                    line_to_copy, idx, lines_to_copy
                                ):
                                    line_to_add = line_to_copy.replace(
                                        "__operation__", "__distribution__"
                                    )
                                    my_file.write("        " + line_to_add)
                            my_file.write("\n")

                    if "avgas" in contents:
                        path_to_gasoline_prod_file = (
                            RESOURCE_FOLDER_PATH / "gasoline_production.yml"
                        )
                        with open(path_to_gasoline_prod_file, "r") as gasoline_prod_conf:
                            lines_to_copy = gasoline_prod_conf.readlines()
                            for idx, line_to_copy in enumerate(lines_to_copy):
                                if not self.configurator.belongs_to_custom_attribute_definition(
                                    line_to_copy, idx, lines_to_copy
                                ):
                                    line_to_add = line_to_copy.replace(
                                        "__operation__", "__distribution__"
                                    )
                                    my_file.write("        " + line_to_add)
                            my_file.write("\n")

            elif self.options["delivery_method"] == "train":
                path_to_train_distribution_conf_file = (
                    RESOURCE_FOLDER_PATH / "delivery_via_train.yml"
                )
                with open(path_to_train_distribution_conf_file, "r") as delivery_via_train:
                    for line_to_copy in delivery_via_train.readlines():
                        my_file.write("        " + line_to_copy)
                    my_file.write("\n")

    def write_use(self, path_to_yaml: pathlib.Path, dict_with_use):
        with open(path_to_yaml, "a") as my_file:
            my_file.write("\n")
            my_file.write("    use:\n")
            # The use of use seem to cause problem when written
            # somewhere else in the code but not here, so it'll stay like that there
            my_file.write("        custom_attributes:\n")
            my_file.write('            - attribute: "phase"\n')
            my_file.write('              value: "operation"\n')
            my_file.write("\n")

            for component_name in dict_with_use.keys():
                for line_to_copy in dict_with_use[component_name]:
                    my_file.write("        " + line_to_copy)
                my_file.write("\n")

            battery_names, _ = self.configurator.get_battery_list()

            if battery_names:
                path_to_electricity_prod_file = RESOURCE_FOLDER_PATH / "electricity_production.yml"
                with open(path_to_electricity_prod_file, "r") as electricity_prod_conf:
                    for line_to_copy in electricity_prod_conf.readlines():
                        my_file.write("        " + line_to_copy)
                    my_file.write("\n")

            tank_names, tank_types, contents = self.configurator.get_fuel_tank_list_and_fuel()

            if tank_names:
                if "jet_fuel" in contents:
                    path_to_kerosene_prod_file = RESOURCE_FOLDER_PATH / "kerosene_production.yml"
                    with open(path_to_kerosene_prod_file, "r") as kerosene_prod_conf:
                        for line_to_copy in kerosene_prod_conf.readlines():
                            my_file.write("        " + line_to_copy)
                        my_file.write("\n")

                if "avgas" in contents:
                    path_to_gasoline_prod_file = RESOURCE_FOLDER_PATH / "gasoline_production.yml"
                    with open(path_to_gasoline_prod_file, "r") as gasoline_prod_conf:
                        for line_to_copy in gasoline_prod_conf.readlines():
                            my_file.write("        " + line_to_copy)
                        my_file.write("\n")

    def change_electric_mix(self, lca_conf_file_path):
        path_to_folder = lca_conf_file_path.parents[0]
        tmp_copy_path = path_to_folder / "tmp_copy.yml"

        shutil.copy(lca_conf_file_path, tmp_copy_path)

        next_line_to_ignore = False

        with open(tmp_copy_path, "w") as new_file:
            with open(lca_conf_file_path, "r") as old_file:
                for line in old_file:
                    if "market group for electricity, high voltage" not in line:
                        if not next_line_to_ignore:
                            new_file.write(line)
                        else:
                            next_line_to_ignore = False
                    else:
                        # line should be starting with "name:"
                        indent_pattern = line.split("name")[0]
                        new_file.write(
                            indent_pattern + "name: 'electricity, high voltage, production mix'\n"
                        )

                        # TODO: Add more eventually
                        if self.options["electric_mix"] == "french":
                            new_file.write(indent_pattern + "loc: 'FR'\n")
                        if self.options["electric_mix"] == "slovenia":
                            new_file.write(indent_pattern + "loc: 'SI'\n")

                        next_line_to_ignore = True

        pathlib.Path.unlink(lca_conf_file_path)
        shutil.move(tmp_copy_path, lca_conf_file_path)

    def _check_existing_file_instance(
        self, lca_conf_file_path: pathlib.Path, project_name, component_names
    ):
        """
        Check if a file at the same address and with the same contents already exists to avoid
        rewriting something strictly similar. Content will be the same if the project name is the
        same, if the eco_invent version is the same, if the LCIA method is the same, if there are
        the same components (but their mass will be able to vary), if the materials used for the
        airframe are the same, if the same delivery method is used, if the same electric mix is
        used and if a component level breakdown is required or not.
        """

        # If cache is empty, can't use an existing file
        if not LCACore._cache_file:
            return False

        key = str(lca_conf_file_path)

        # If cache is not empty but there is no instance of that particular configuration file,
        # can't use existing file
        if key not in LCACore._cache_file:
            return False

        # Otherwise we check if options and contents are the same
        cache_instance = LCACore._cache_file[key]
        existing_file_in_cache = all(
            (
                project_name == cache_instance["project_name"],
                self.options["component_level_breakdown"]
                == cache_instance["component_level_breakdown"],
                self.options["impact_assessment_method"]
                == cache_instance["impact_assessment_method"],
                self.options["ecoinvent_version"] == cache_instance["ecoinvent_version"],
                self.options["airframe_material"] == cache_instance["airframe_material"],
                self.options["delivery_method"] == cache_instance["delivery_method"],
                self.options["electric_mix"] == cache_instance["electric_mix"],
                component_names == cache_instance["component_names"],
            )
        )

        return existing_file_in_cache

    def _add_cache_file_instance(self, lca_conf_file_path, project_name, component_names):
        """
        In the case where no file exists with the exact same constant, we register its
        instance after rewriting it.
        """

        key = str(lca_conf_file_path)

        file_cache_instance = {
            "project_name": project_name,
            "component_level_breakdown": self.options["component_level_breakdown"],
            "impact_assessment_method": self.options["impact_assessment_method"],
            "ecoinvent_version": self.options["ecoinvent_version"],
            "airframe_material": self.options["airframe_material"],
            "delivery_method": self.options["delivery_method"],
            "electric_mix": self.options["electric_mix"],
            "component_names": component_names,
        }
        LCACore._cache_file[key] = file_cache_instance

    def write_lca_conf_file(self):
        ecoinvent_version = self.options["ecoinvent_version"]
        methods = self.options["impact_assessment_method"]

        power_train_file_path = self.options["power_train_file_path"]

        # For the rest of the operations to work, power_train_file_path must be a pathlib Path.
        # By default, FAST-OAD will return file path in option as posix
        if type(power_train_file_path) is str:
            power_train_file_path = pathlib.Path(power_train_file_path)

        parent_folder = power_train_file_path.parents[0]
        power_train_file_name = power_train_file_path.name
        project_name = power_train_file_name.replace(".yml", "")
        lca_conf_file_name = power_train_file_name.replace(".yml", "_lca.yml")
        lca_conf_file_path = parent_folder / lca_conf_file_name

        self.configurator._get_components()

        dict_with_manufacturing, species_list = (
            self.configurator.get_lca_manufacturing_phase_element_list()
        )
        for specie in species_list:
            self.name_to_unit[specie] = "kg"

        dict_with_distribution, species_list = (
            self.configurator.get_lca_distribution_phase_element_list()
        )

        for specie in species_list:
            self.name_to_unit[specie] = "kg"

        dict_with_use, species_list = self.configurator.get_lca_use_phase_element_list()

        for specie in species_list:
            self.name_to_unit[specie] = "kg"

        if self.options["write_lca_conf"]:
            # Before writing anything we check if a file with the exact same content at the exact
            # same address doesn't exist. We'll judge if the same components are there if the same
            # variables names for their masses are there
            variables_names_mass = self.configurator.get_mass_element_lists()

            if not self._check_existing_file_instance(
                lca_conf_file_path, project_name, variables_names_mass
            ):
                header = {"project": project_name}

                with open(lca_conf_file_path, "w") as new_file:
                    yaml.safe_dump(header, new_file)

                self.write_ecoinvent_version(lca_conf_file_path, ecoinvent_version)

                dict_with_production = self.configurator.get_lca_production_element_list()

                # This function writes the "model: " in the yml so it must be run before anything
                # else that need to go in the model section in the model section
                self.write_production(lca_conf_file_path, dict_with_production)

                self.write_manufacturing(lca_conf_file_path, dict_with_manufacturing)

                self.write_distribution(lca_conf_file_path, dict_with_distribution)

                self.write_use(lca_conf_file_path, dict_with_use)

                methods_file = RESOURCE_FOLDER_PATH / METHODS_TO_FILE[methods]

                with open(methods_file, "r") as methods_file_stream:
                    methods_dict = yaml.safe_load(methods_file_stream)

                self.write_methods(lca_conf_file_path, methods_dict)

                if self.options["electric_mix"] != "default":
                    self.change_electric_mix(lca_conf_file_path)

                self._add_cache_file_instance(
                    lca_conf_file_path, project_name, variables_names_mass
                )

        elif self.options["lca_conf_file_path"]:
            lca_conf_file_path = self.options["lca_conf_file_path"]

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
