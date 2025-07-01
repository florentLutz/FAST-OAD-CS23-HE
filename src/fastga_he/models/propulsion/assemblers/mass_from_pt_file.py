# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from .constants import SUBMODEL_POWER_TRAIN_MASS

MASS_FROM_PT = "fastga_he.submodel.propulsion.mass.from_pt_file"
oad.RegisterSubmodel.active_models[SUBMODEL_POWER_TRAIN_MASS] = MASS_FROM_PT


@oad.RegisterSubmodel(SUBMODEL_POWER_TRAIN_MASS, "fastga_he.submodel.propulsion.mass.from_pt_file")
class PowerTrainMassFromFile(om.ExplicitComponent):
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

        variables_names = self.configurator.get_mass_element_lists()

        self.add_output("data:propulsion:he_power_train:mass", val=350.0, units="kg")

        for variable_name in variables_names:
            self.add_input(variable_name, val=np.nan, units="kg")
            self.declare_partials(
                of="data:propulsion:he_power_train:mass", wrt=variable_name, val=1.0
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["data:propulsion:he_power_train:mass"] = sum(inputs.values())
