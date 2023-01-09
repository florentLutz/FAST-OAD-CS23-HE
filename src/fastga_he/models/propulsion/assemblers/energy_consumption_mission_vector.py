# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from fastga_he.models.performances.mission_vector.constants import HE_SUBMODEL_ENERGY_CONSUMPTION
from fastga_he.models.propulsion.assemblers.constants import (
    SUBMODEL_POWER_TRAIN_PERF,
    SUBMODEL_POWER_RATE,
)


@oad.RegisterSubmodel(
    HE_SUBMODEL_ENERGY_CONSUMPTION,
    "fastga_he.submodel.performances.energy_consumption.from_pt_file",
)
class PowerTrainPerformancesFromFileWithInterface(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

    def initialize(self):

        # We have to declare them even if not used to preserve compatibility
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare(
            name="power_train_file_path",
            default="",
            desc="Path to the file containing the description of the power",
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        # Two points for taxi out and taxi in
        number_of_points = self.options["number_of_points"] + 2
        power_train_file_path = self.options["power_train_file_path"]

        self.configurator.load(self.options["power_train_file_path"])

        (
            propulsive_load_names,
            propulsive_load_types,
        ) = self.configurator.get_propulsive_element_list()

        # TODO: Finish the mock-up interface
        options_pt_perf = {
            "power_train_file_path": power_train_file_path,
            "number_of_points": number_of_points,
        }

        # For some reasons that I only knew when I coded the mission vector, all flight
        # parameters have "econ" in them. Additionally for some reason that I only knew when
        # coding the power train builder components, I didn't think of re-using the
        # aforementioned naming convention. Consequently, we will have to use the OpenMDAO trick
        # which consists in renaming variable during promotion. Thanks @christophe-david for
        # telling me it is possible
        self.add_subsystem(
            "power_train_performances",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_POWER_TRAIN_PERF, options=options_pt_perf),
            promotes=[
                "data:*",
                "settings:*",
                "fuel_consumed_t_econ",
                "non_consumable_energy_t_econ",
                ("thrust", "thrust_econ"),
                ("altitude", "altitude_econ"),
                ("time_step", "time_step_econ"),
                ("true_airspeed", "true_airspeed_econ"),
                ("exterior_temperature", "exterior_temperature_econ"),
            ],
        )
        self.add_subsystem(
            "mock_up_interface",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_POWER_RATE, options=options_pt_perf),
            promotes=["data:*", "thrust_rate_t_econ", "engine_setting_econ"],
        )

        for propulsive_load_name in propulsive_load_names:
            self.connect(
                "power_train_performances." + propulsive_load_name + ".shaft_power_for_power_rate",
                "mock_up_interface." + propulsive_load_name + "_shaft_power_out",
            )
