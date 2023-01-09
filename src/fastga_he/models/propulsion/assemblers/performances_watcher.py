# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import os

import numpy as np
import pandas as pd

import openmdao.api as om
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator


class PowerTrainPerformancesWatcher(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()
        self.header_name = []

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

        (
            components_name,
            components_performances_watchers_names,
            components_performances_watchers_units,
        ) = self.configurator.get_performance_watcher_elements_list()

        for (
            component_name,
            component_performances_watcher_name,
            component_performances_watcher_unit,
        ) in zip(
            components_name,
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

        results_df = pd.DataFrame(columns=self.header_name)

        for (component_name, component_performances_watcher_name, corresponding_header) in zip(
            components_name, components_performances_watchers_names, self.header_name
        ):
            results_df[corresponding_header] = inputs[
                component_name + "_" + component_performances_watcher_name
            ]

            results_df.to_csv(file_path)
