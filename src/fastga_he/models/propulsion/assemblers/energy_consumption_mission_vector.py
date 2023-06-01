# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO
import itertools

import openmdao.api as om

import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from fastga_he.models.performances.mission_vector.constants import HE_SUBMODEL_ENERGY_CONSUMPTION
from fastga_he.models.propulsion.assemblers.constants import (
    SUBMODEL_POWER_TRAIN_PERF,
    SUBMODEL_POWER_RATE,
)

ENERGY_CONSUMPTION_FROM_PT_FILE = "fastga_he.submodel.performances.energy_consumption.from_pt_file"


@oad.RegisterSubmodel(
    HE_SUBMODEL_ENERGY_CONSUMPTION,
    ENERGY_CONSUMPTION_FROM_PT_FILE,
)
class PowerTrainPerformancesFromFileWithInterface(om.Group):
    """
    This group is solely meant to be used as an interface with the vector mission, if the user
    wish to run a standalone analysis of a power train, use the SUBMODEL_POWER_TRAIN_PERF
    service.

    This is the reason why the add_solver options will be set to False without access to it on
    mission level. This also allows to not have to set up another dummy option in the
    propulsion_basic.py file.
    """

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
        self.options.declare(
            name="pre_condition_voltage",
            default=False,
            desc="Boolean to pre_condition the voltages of the different components of the PT, "
            "can save some time in specific cases",
            allow_none=False,
        )

    def setup(self):

        # Two points for taxi out and taxi in
        number_of_points = self.options["number_of_points"] + 2
        power_train_file_path = self.options["power_train_file_path"]
        pre_condition_voltage = self.options["pre_condition_voltage"]

        self.configurator.load(self.options["power_train_file_path"])

        (
            propulsive_load_names,
            propulsive_load_types,
        ) = self.configurator.get_propulsive_element_list()

        options_pt_perf = {
            "power_train_file_path": power_train_file_path,
            "number_of_points": number_of_points,
            "add_solver": False,
            "pre_condition_voltage": pre_condition_voltage,
        }

        # For some reasons that I only knew when I coded the mission vector, all flight
        # parameters have "econ" in them. Additionally for some reason that I only knew when
        # coding the power train builder components, I didn't think of re-using the
        # aforementioned naming convention. Consequently, we will have to use the OpenMDAO trick
        # which consists in renaming variable during promotion. Thanks @christophe-david for
        # telling me it is possible.

        # Before promoting though, we must look at the components in the powertrain and check
        # what aircraft level parameters they need. We will remove the settings, as they are by
        # default promotes but for the rest we need to ensure no duplicate

        promotes_list = self.configurator.get_performances_element_lists()[7]
        promotes_list = list(itertools.chain.from_iterable(promotes_list))
        promotes_list = [*set(promotes_list)]
        if "settings:*" in promotes_list:
            promotes_list.remove("settings:*")

        definitive_list = []

        for promotion in promotes_list:
            definitive_list.append((promotion, promotion + "_econ"))

        self.add_subsystem(
            "power_train_performances",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_POWER_TRAIN_PERF, options=options_pt_perf),
            promotes=[
                "data:*",
                "settings:*",
                "fuel_consumed_t_econ",
                "non_consumable_energy_t_econ",
                ("thrust", "thrust_econ"),
            ]
            + definitive_list,
        )

        options_power_rate = {
            "power_train_file_path": power_train_file_path,
            "number_of_points": number_of_points,
        }

        self.add_subsystem(
            "mock_up_interface",
            oad.RegisterSubmodel.get_submodel(SUBMODEL_POWER_RATE, options=options_power_rate),
            promotes=[
                "data:*",
                "thrust_rate_t_econ",
                "engine_setting_econ",
                "exterior_temperature_econ",
                "altitude_econ",
                "density_econ",
            ],
        )

        for propulsive_load_name in propulsive_load_names:
            self.connect(
                "power_train_performances." + propulsive_load_name + ".shaft_power_for_power_rate",
                "mock_up_interface." + propulsive_load_name + "_shaft_power_out",
            )
