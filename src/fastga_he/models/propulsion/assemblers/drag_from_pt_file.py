# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from .constants import SUBMODEL_POWER_TRAIN_DRAG

DRAG_FROM_PT_FILE = "fastga_he.submodel.propulsion.drag.from_pt_file"
oad.RegisterSubmodel.active_models[SUBMODEL_POWER_TRAIN_DRAG] = DRAG_FROM_PT_FILE


@oad.RegisterSubmodel(SUBMODEL_POWER_TRAIN_DRAG, DRAG_FROM_PT_FILE)
class PowerTrainDragFromFile(om.ExplicitComponent):
    """
    Computes the profile drag of the power train by summing the contribution of all its components.
    """

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

        drag_names_ls, drag_names_cruise = self.configurator.get_drag_element_lists()

        self.add_output("data:propulsion:he_power_train:low_speed:CD0", val=0.0)
        self.add_output("data:propulsion:he_power_train:cruise:CD0", val=0.0)

        for drag_ls, drag_cruise in zip(drag_names_ls, drag_names_cruise):
            self.add_input(drag_ls, val=np.nan)
            self.add_input(drag_cruise, val=np.nan)

            self.declare_partials(
                of="data:propulsion:he_power_train:low_speed:CD0", wrt=drag_ls, val=1.0
            )
            self.declare_partials(
                of="data:propulsion:he_power_train:cruise:CD0", wrt=drag_cruise, val=1.0
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        drag_names_ls, drag_names_cruise = self.configurator.get_drag_element_lists()

        cd0_ls = 0.0
        cd0_cruise = 0.0

        for drag_ls, drag_cruise in zip(drag_names_ls, drag_names_cruise):
            cd0_ls += inputs[drag_ls]
            cd0_cruise += inputs[drag_cruise]

        outputs["data:propulsion:he_power_train:low_speed:CD0"] = cd0_ls
        outputs["data:propulsion:he_power_train:cruise:CD0"] = cd0_cruise
