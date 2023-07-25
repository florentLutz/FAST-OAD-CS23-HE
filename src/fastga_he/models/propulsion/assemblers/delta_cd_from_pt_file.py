# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from .constants import SUBMODEL_POWER_TRAIN_DELTA_CD


@oad.RegisterSubmodel(
    SUBMODEL_POWER_TRAIN_DELTA_CD, "fastga_he.submodel.propulsion.delta_cd.from_pt_file"
)
class PowerTrainDeltaCdFromFile(om.ExplicitComponent):
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

        self.add_input("delta_Cdi", val=np.nan, shape=number_of_points)

        self.add_output("delta_Cd", val=0.0, shape=number_of_points)

        self.declare_partials(of="delta_Cd", wrt="delta_Cdi", val=np.eye(number_of_points))

        for component_name in components_name:

            component_delta_cd = component_name + "_delta_Cd"

            self.add_input(component_delta_cd, val=np.full(number_of_points, np.nan))

            self.declare_partials(
                of="delta_Cd", wrt=component_delta_cd, val=np.eye(number_of_points)
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

        total_cd = 0.0 + inputs["delta_Cdi"]

        for component_name in components_name:

            component_delta_cd = component_name + "_delta_Cd"

            total_cd += inputs[component_delta_cd]

        outputs["delta_Cd"] = total_cd
