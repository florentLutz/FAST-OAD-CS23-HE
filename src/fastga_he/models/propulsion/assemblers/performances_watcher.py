# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os

import numpy as np
import pandas as pd

import openmdao.api as om

import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import (
    FASTGAHEPowerTrainConfigurator,
    PROMOTION_FROM_MISSION,
)

from fastga_he.models.performances.mission_vector.constants import HE_SUBMODEL_DEP_EFFECT
from fastga_he.models.propulsion.assemblers.delta_from_pt_file import DEP_EFFECT_FROM_PT_FILE


class PowerTrainPerformancesWatcher(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()
        self.header_name = []

        self.right_submodel_slip_effect = False

    def initialize(self):
        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the power",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        """
        This component is only added to the problem when we are sure that we want to save the
        performances of the power train inside a file (watcher_file_path field is not empty).
        This means we can freely add any input but we will still have to have a fake input.
        """

        number_of_points = self.options["number_of_points"]

        self.configurator.load(self.options["power_train_file_path"])

        # See mission_vector.py for the reason why we need this boolean
        self.right_submodel_slip_effect = (
            oad.RegisterSubmodel.active_models[HE_SUBMODEL_DEP_EFFECT] == DEP_EFFECT_FROM_PT_FILE
        )

        (
            components_names,
            components_performances_watchers_names,
            components_performances_watchers_units,
        ) = self.configurator.get_performance_watcher_elements_list()

        for (
            component_name,
            component_performances_watcher_name,
            component_performances_watcher_unit,
        ) in zip(
            components_names,
            components_performances_watchers_names,
            components_performances_watchers_units,
        ):
            self.add_input(
                component_name + "_" + component_performances_watcher_name,
                units=component_performances_watcher_unit,
                val=np.nan,
                shape=number_of_points,
            )

            if component_performances_watcher_unit is None:
                component_performances_watcher_unit = "-"

            self.header_name.append(
                component_name
                + " "
                + component_performances_watcher_name
                + " ["
                + component_performances_watcher_unit
                + "]"
            )

        for mission_variable_name in list(PROMOTION_FROM_MISSION.keys()):
            self.add_input(
                mission_variable_name,
                units=PROMOTION_FROM_MISSION[mission_variable_name],
                val=np.nan,
                shape=number_of_points,
            )

            self.header_name.append(
                mission_variable_name + " [" + PROMOTION_FROM_MISSION[mission_variable_name] + "]"
            )

        if self.right_submodel_slip_effect:
            (
                components_slip_names,
                components_slip_performances_watchers_names,
                components_slip_performances_watchers_units,
            ) = self.configurator.get_slipstream_performance_watcher_elements_list()

            for (
                components_slip_name,
                components_slip_performances_watchers_name,
                components_slip_performances_watchers_unit,
            ) in zip(
                components_slip_names,
                components_slip_performances_watchers_names,
                components_slip_performances_watchers_units,
            ):
                self.add_input(
                    components_slip_name + "_" + components_slip_performances_watchers_name,
                    units=components_slip_performances_watchers_unit,
                    val=np.nan,
                    shape=number_of_points - 2,
                )
                # I hate that -2 but it works and is a quick fix. Basically, I didn't intend on
                # using slipstream effect for taxi but for the performances we need it so the
                # quick fix is to do that. More shenanigans to follow

                if components_slip_performances_watchers_unit is None:
                    components_slip_performances_watchers_unit = "-"

                self.header_name.append(
                    components_slip_name
                    + " "
                    + components_slip_performances_watchers_name
                    + " ["
                    + components_slip_performances_watchers_unit
                    + "]"
                )

        self.add_output("dummy_output", val=1337.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        file_path = self.configurator.get_watcher_file_path()

        if os.path.exists(file_path):
            os.remove(file_path)

        if not os.path.exists(os.path.dirname(file_path)):
            os.mkdir(os.path.dirname(file_path))

        (
            components_name,
            components_performances_watchers_names,
            _,
        ) = self.configurator.get_performance_watcher_elements_list()

        mission_variable_names = list(PROMOTION_FROM_MISSION.keys())

        components_name_with_mission = components_name + [None] * len(mission_variable_names)
        inputs_names = components_performances_watchers_names + mission_variable_names

        # Said shenanigans
        is_slip_list = [False] * len(components_name) + [False] * len(mission_variable_names)

        if self.right_submodel_slip_effect:
            (
                components_slip_names,
                components_slip_performances_watchers_names,
                _,
            ) = self.configurator.get_slipstream_performance_watcher_elements_list()

            components_name_with_mission += components_slip_names
            inputs_names += components_slip_performances_watchers_names
            is_slip_list += [True] * len(components_slip_names)

        results_df = pd.DataFrame(columns=self.header_name)

        for (
            component_name,
            component_performances_watcher_name,
            corresponding_header,
            is_slip,
        ) in zip(components_name_with_mission, inputs_names, self.header_name, is_slip_list):
            if not component_name:
                value_to_save = inputs[component_performances_watcher_name]
            elif not is_slip:
                value_to_save = inputs[component_name + "_" + component_performances_watcher_name]
            else:  # Means we are registering slipstream effects
                value_to_save = np.concatenate(
                    (
                        np.zeros(1),
                        inputs[component_name + "_" + component_performances_watcher_name],
                        np.zeros(1),
                    )
                )

            results_df[corresponding_header] = value_to_save

        results_df.to_csv(file_path)
