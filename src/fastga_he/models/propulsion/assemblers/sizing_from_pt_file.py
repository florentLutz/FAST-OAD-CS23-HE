# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import fastoad.api as oad

from fastoad.module_management.constants import ModelDomain

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

# noinspection PyUnresolvedReferences
from fastga_he.models.propulsion.components import (
    SizingPropeller,
    SizingPMSM,
    SizingInverter,
    SizingDCBus,
    SizingHarness,
    SizingDCDCConverter,
    SizingBatteryPack,
    SizingDCSSPC,
    SizingDCSplitter,
)

from .constants import SUBMODEL_POWER_TRAIN_MASS, SUBMODEL_POWER_TRAIN_CG, SUBMODEL_POWER_TRAIN_DRAG


@oad.RegisterOpenMDAOSystem("fastga_he.power_train.sizing", domain=ModelDomain.OTHER)
class PowerTrainSizingFromFile(om.Group):
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

    def setup(self):

        self.configurator.load(self.options["power_train_file_path"])

        (
            components_name,
            components_name_id,
            components_type,
            components_om_type,
            components_position,
        ) = self.configurator.get_sizing_element_lists()

        for (
            component_name,
            component_name_id,
            component_type,
            component_om_type,
            component_position,
        ) in zip(
            components_name,
            components_name_id,
            components_type,
            components_om_type,
            components_position,
        ):

            klass = globals()["Sizing" + component_om_type]
            local_sub_sys = klass()
            local_sub_sys.options[component_name_id] = component_name
            if component_position:
                local_sub_sys.options["position"] = component_position

            self.add_subsystem(name=component_name, subsys=local_sub_sys, promotes=["*"])

        option_pt_file = {"power_train_file_path": self.options["power_train_file_path"]}
        self.add_subsystem(
            name="power_train_mass",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_POWER_TRAIN_MASS, options=option_pt_file
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="power_train_CG",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_POWER_TRAIN_CG, options=option_pt_file
            ),
            promotes=["*"],
        )
        self.add_subsystem(
            name="power_train_drag",
            subsys=oad.RegisterSubmodel.get_submodel(
                SUBMODEL_POWER_TRAIN_DRAG, options=option_pt_file
            ),
            promotes=["*"],
        )
