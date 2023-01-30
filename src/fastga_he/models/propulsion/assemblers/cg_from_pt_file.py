# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator

from .constants import SUBMODEL_POWER_TRAIN_CG


@oad.RegisterSubmodel(SUBMODEL_POWER_TRAIN_CG, "fastga_he.submodel.propulsion.cg.from_pt_file")
class PowerTrainCGFromFile(om.ExplicitComponent):
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

        variables_names_mass = self.configurator.get_mass_element_lists()
        variables_names_cg = self.configurator.get_cg_element_lists()

        self.add_output("data:propulsion:he_power_train:CG:x", val=2.5, units="m")

        for mass_name, cg_name in zip(variables_names_mass, variables_names_cg):

            self.add_input(mass_name, val=np.nan, units="kg")
            self.add_input(cg_name, val=np.nan, units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        variables_names_mass = self.configurator.get_mass_element_lists()
        variables_names_cg = self.configurator.get_cg_element_lists()

        pt_mass = 0.0
        pt_moment_arm = 0.0

        for mass_name, cg_name in zip(variables_names_mass, variables_names_cg):

            # By convention, if CG is negative, we do not account for the component in the
            # computation of the power train CG, though its mass will be taken into account.
            cg_component = inputs[cg_name]
            mass_component = inputs[mass_name]
            if cg_component >= 0.0:
                pt_moment_arm += cg_component * mass_component
                pt_mass += mass_component

        outputs["data:propulsion:he_power_train:CG:x"] = pt_moment_arm / pt_mass

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        variables_names_mass = self.configurator.get_mass_element_lists()
        variables_names_cg = self.configurator.get_cg_element_lists()

        pt_mass = 0.0
        pt_moment_arm = 0.0

        # Need to run it once to get the denominator
        for mass_name, cg_name in zip(variables_names_mass, variables_names_cg):
            cg_component = inputs[cg_name]
            mass_component = inputs[mass_name]
            if cg_component >= 0.0:
                pt_mass += mass_component
                pt_moment_arm += cg_component * mass_component

        for mass_name, cg_name in zip(variables_names_mass, variables_names_cg):

            # By convention, if CG is negative, we do not account for the component in the
            # computation of the power train CG, though its mass will be taken into account.
            cg_component = inputs[cg_name]
            mass_component = inputs[mass_name]

            if cg_component < 0.0:

                partials["data:propulsion:he_power_train:CG:x", cg_name] = 0.0
                partials["data:propulsion:he_power_train:CG:x", mass_name] = 0.0

            else:

                partials["data:propulsion:he_power_train:CG:x", cg_name] = mass_component / pt_mass
                partials["data:propulsion:he_power_train:CG:x", mass_name] = (
                    cg_component * pt_mass - pt_moment_arm
                ) / pt_mass ** 2.0
