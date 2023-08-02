# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from .constants import SUBMODEL_POWER_TRAIN_DELTA_CL


@oad.RegisterSubmodel(
    SUBMODEL_POWER_TRAIN_DELTA_CL, "fastga_he.submodel.propulsion.delta_cl.from_pt_file"
)
class PowerTrainDeltaClFromFile(om.ExplicitComponent):
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
            components_slipstream_wing_lift,
        ) = self.configurator.get_slipstream_element_lists()

        self.add_output("delta_Cl", val=0.0, shape=number_of_points)
        self.add_output("delta_Cl_wing", val=0.0, shape=number_of_points)

        for component_name, component_slipstream_wing_lift in zip(
            components_name, components_slipstream_wing_lift
        ):

            component_delta_cl = component_name + "_delta_Cl"

            self.add_input(component_delta_cl, val=np.full(number_of_points, np.nan))

            self.declare_partials(
                of="delta_Cl", wrt=component_delta_cl, val=np.eye(number_of_points)
            )

            if component_slipstream_wing_lift:

                self.declare_partials(
                    of="delta_Cl_wing", wrt=component_delta_cl, val=np.eye(number_of_points)
                )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        (
            components_name,
            _,
            _,
            _,
            _,
            _,
            components_slipstream_wing_lift,
        ) = self.configurator.get_slipstream_element_lists()

        total_cl = 0.0
        wing_cl = 0.0

        for component_name, component_slipstream_wing_lift in zip(
            components_name, components_slipstream_wing_lift
        ):

            component_delta_cl = component_name + "_delta_Cl"

            total_cl += inputs[component_delta_cl]

            if component_slipstream_wing_lift:

                wing_cl += inputs[component_delta_cl]

        outputs["delta_Cl"] = total_cl
        outputs["delta_Cl_wing"] = wing_cl
