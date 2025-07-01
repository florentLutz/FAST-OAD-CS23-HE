# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from .constants import SUBMODEL_POWER_TRAIN_DELTA_CM

DELTA_CM_FROM_PT = "fastga_he.submodel.propulsion.delta_cm.from_pt_file"
oad.RegisterSubmodel.active_models[SUBMODEL_POWER_TRAIN_DELTA_CM] = DELTA_CM_FROM_PT


@oad.RegisterSubmodel(SUBMODEL_POWER_TRAIN_DELTA_CM, DELTA_CM_FROM_PT)
class PowerTrainDeltaCmFromFile(om.ExplicitComponent):
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
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]
        self.configurator.load(self.options["power_train_file_path"])

        (
            components_name,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.configurator.get_slipstream_element_lists()

        self.add_output("delta_Cm", val=0.0, shape=number_of_points)

        for component_name in components_name:
            component_delta_cm = component_name + "_delta_Cm"

            self.add_input(component_delta_cm, val=np.full(number_of_points, np.nan))

            self.declare_partials(
                of="delta_Cm",
                wrt=component_delta_cm,
                rows=np.arange(number_of_points),
                cols=np.arange(number_of_points),
                val=np.ones(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        (
            components_name,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.configurator.get_slipstream_element_lists()

        total_cm = 0.0

        for component_name in components_name:
            component_delta_cm = component_name + "_delta_Cm"

            total_cm += inputs[component_delta_cm]

        outputs["delta_Cm"] = total_cm
