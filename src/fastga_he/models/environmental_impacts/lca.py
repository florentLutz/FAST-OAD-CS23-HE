# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2024 ISAE-SUPAERO

import numpy as np
import openmdao.api as om

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator
import fastga_he.models.propulsion.components as he_comp

from .lca_aircraft_per_fu import LCAAircraftPerFU
from .lca_core import LCACore, METHODS_TO_FILE


class LCA(om.Group):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

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

        self.add_subsystem(name="aircraft_per_fu", subsys=LCAAircraftPerFU(), promotes=["*"])

        # For the most part we can reuse what is done for the sizing, no need to write a new
        # function
        (
            components_name,
            components_name_id,
            components_type,
            components_om_type,
            _,
        ) = self.configurator.get_sizing_element_lists()

        for (
            component_name,
            component_name_id,
            component_type,
            component_om_type,
        ) in zip(
            components_name,
            components_name_id,
            components_type,
            components_om_type,
        ):
            local_sub_sys = he_comp.__dict__["PreLCA" + component_om_type]()
            local_sub_sys.options[component_name_id] = component_name

            self.add_subsystem(name=component_name, subsys=local_sub_sys, promotes=["*"])

        self.add_subsystem(
            name="lca_core",
            subsys=LCACore(
                power_train_file_path=self.options["power_train_file_path"],
                component_level_breakdown=self.options["component_level_breakdown"],
                impact_assessment_method=self.options["impact_assessment_method"],
                ecoinvent_version=self.options["ecoinvent_version"],
            ),
            promotes=["*"],
        )
