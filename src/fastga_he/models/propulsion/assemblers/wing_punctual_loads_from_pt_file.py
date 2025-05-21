# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator, PT_DATA_PREFIX

from .constants import SUBMODEL_POWER_TRAIN_WING_PUNCTUAL_LOADS


@oad.RegisterSubmodel(
    SUBMODEL_POWER_TRAIN_WING_PUNCTUAL_LOADS,
    "fastga_he.submodel.propulsion.wing.punctual_loads.from_pt_file",
)
class PowerTrainPunctualLoadsFromFile(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.configurator = FASTGAHEPowerTrainConfigurator()

        self.curated_name_list = None
        self.curated_type_list = None

    def initialize(self):
        self.options.declare(
            name="power_train_file_path",
            default=None,
            desc="Path to the file containing the description of the power",
            allow_none=False,
        )

    def setup(self):
        self.configurator.load(self.options["power_train_file_path"])

        # First we get the list of punctual masses and the pairs
        (
            wing_punctual_mass_list,
            wing_punctual_mass_type_list,
            symmetrical_pair_list,
        ) = self.configurator.get_wing_punctual_mass_element_list()

        name_to_type = dict(zip(wing_punctual_mass_list, wing_punctual_mass_type_list))

        # Then we remove the second element of each pair. We know the pairs are unique from the
        # way it was coded and all pairs should have only component that exists or else an error
        # would have been raised earlier
        for pair in symmetrical_pair_list:
            wing_punctual_mass_list.remove(pair[1])
            wing_punctual_mass_type_list.remove(name_to_type[pair[1]])

        self.curated_name_list = wing_punctual_mass_list
        self.curated_type_list = wing_punctual_mass_type_list

        self.add_output(
            "data:weight:airframe:wing:punctual_mass:y_ratio",
            shape=len(wing_punctual_mass_list),
            val=0.0,
        )
        self.add_output(
            "data:weight:airframe:wing:punctual_mass:mass",
            units="kg",
            shape=len(wing_punctual_mass_list),
            val=0.0,
        )

        for punctual_mass_name, punctual_mass_type in zip(
            wing_punctual_mass_list, wing_punctual_mass_type_list
        ):
            y_ratio_name = (
                PT_DATA_PREFIX + punctual_mass_type + ":" + punctual_mass_name + ":CG:y_ratio"
            )
            mass_name = PT_DATA_PREFIX + punctual_mass_type + ":" + punctual_mass_name + ":mass"

            self.add_input(y_ratio_name, val=np.nan)
            self.add_input(mass_name, val=0.0, units="kg")

            self.declare_partials(
                of="data:weight:airframe:wing:punctual_mass:y_ratio",
                wrt=y_ratio_name,
                method="exact",
            )
            self.declare_partials(
                of="data:weight:airframe:wing:punctual_mass:mass", wrt=mass_name, method="exact"
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        punctual_masses_y_ratio = np.array([])
        punctual_masses_masses = np.array([])

        for punctual_mass_name, punctual_mass_type in zip(
            self.curated_name_list, self.curated_type_list
        ):
            y_ratio_name = (
                PT_DATA_PREFIX + punctual_mass_type + ":" + punctual_mass_name + ":CG:y_ratio"
            )
            mass_name = PT_DATA_PREFIX + punctual_mass_type + ":" + punctual_mass_name + ":mass"

            punctual_masses_y_ratio = np.append(punctual_masses_y_ratio, inputs[y_ratio_name])
            punctual_masses_masses = np.append(punctual_masses_masses, inputs[mass_name])

        outputs["data:weight:airframe:wing:punctual_mass:y_ratio"] = punctual_masses_y_ratio
        outputs["data:weight:airframe:wing:punctual_mass:mass"] = punctual_masses_masses

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        name_and_type_array = np.rec.fromarrays([self.curated_name_list, self.curated_type_list])
        nb_punctual_masses = len(self.curated_name_list)

        for idx, value in enumerate(name_and_type_array.tolist()):
            punctual_mass_name = value[0]
            punctual_mass_type = value[1]

            y_ratio_name = (
                PT_DATA_PREFIX + punctual_mass_type + ":" + punctual_mass_name + ":CG:y_ratio"
            )
            mass_name = PT_DATA_PREFIX + punctual_mass_type + ":" + punctual_mass_name + ":mass"

            partials_value = np.zeros(nb_punctual_masses)
            partials_value[idx] = 1.0

            partials["data:weight:airframe:wing:punctual_mass:y_ratio", y_ratio_name] = (
                partials_value
            )
            partials["data:weight:airframe:wing:punctual_mass:mass", mass_name] = partials_value
