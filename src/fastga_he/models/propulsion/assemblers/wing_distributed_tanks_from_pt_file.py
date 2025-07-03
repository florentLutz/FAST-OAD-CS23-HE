# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import openmdao.api as om
import numpy as np
import fastoad.api as oad

from fastga_he.powertrain_builder.powertrain import FASTGAHEPowerTrainConfigurator, PT_DATA_PREFIX

from .constants import SUBMODEL_POWER_TRAIN_WING_DISTRIBUTED_TANKS

DISTRIBUTED_TANK_FROM_PT_FILE = "fastga_he.submodel.propulsion.wing.distributed_tanks.from_pt_file"
oad.RegisterSubmodel.active_models[SUBMODEL_POWER_TRAIN_WING_DISTRIBUTED_TANKS] = (
    DISTRIBUTED_TANK_FROM_PT_FILE
)


@oad.RegisterSubmodel(SUBMODEL_POWER_TRAIN_WING_DISTRIBUTED_TANKS, DISTRIBUTED_TANK_FROM_PT_FILE)
class PowerTrainDistributedTanksFromFile(om.ExplicitComponent):
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
            wing_distributed_tanks_list,
            wing_distributed_tanks_type_list,
            symmetrical_pair_list,
        ) = self.configurator.get_wing_distributed_fuel_element_list()

        name_to_type = dict(zip(wing_distributed_tanks_list, wing_distributed_tanks_type_list))

        # Then we remove the second element of each pair. We know the pairs are unique from the
        # way it was coded and all pairs should have only component that exists or else an error
        # would have been raised earlier.
        for pair in symmetrical_pair_list:
            wing_distributed_tanks_list.remove(pair[1])
            wing_distributed_tanks_type_list.remove(name_to_type[pair[1]])

        self.curated_name_list = wing_distributed_tanks_list
        self.curated_type_list = wing_distributed_tanks_type_list

        self.add_output(
            "data:weight:airframe:wing:distributed_tanks:y_ratio_start",
            shape=len(wing_distributed_tanks_list),
            val=0.0,
        )
        self.add_output(
            "data:weight:airframe:wing:distributed_tanks:y_ratio_end",
            shape=len(wing_distributed_tanks_list),
            val=0.0,
        )
        self.add_output(
            "data:weight:airframe:wing:distributed_tanks:start_chord",
            shape=len(wing_distributed_tanks_list),
            val=0.0,
            units="m",
        )
        self.add_output(
            "data:weight:airframe:wing:distributed_tanks:chord_slope",
            shape=len(wing_distributed_tanks_list),
            val=0.0,
        )
        self.add_output(
            "data:weight:airframe:wing:distributed_tanks:fuel_inside",
            units="kg",
            shape=len(wing_distributed_tanks_list),
            val=0.0,
        )

        for distributed_tank_name, distributed_tank_type in zip(
            wing_distributed_tanks_list, wing_distributed_tanks_type_list
        ):
            y_ratio_start_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:y_ratio_start"
            )
            y_ratio_end_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:y_ratio_end"
            )
            chord_start_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:start_chord"
            )
            chord_slope_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:chord_slope"
            )
            mass_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":fuel_total_mission"
            )

            self.add_input(y_ratio_start_name, val=0.0)
            self.add_input(y_ratio_end_name, val=0.0)
            self.add_input(chord_start_name, val=0.0, units="m")
            self.add_input(chord_slope_name, val=0.0)
            self.add_input(mass_name, val=0.0, units="kg")

            self.declare_partials(
                of="data:weight:airframe:wing:distributed_tanks:y_ratio_start",
                wrt=y_ratio_start_name,
                method="exact",
            )
            self.declare_partials(
                of="data:weight:airframe:wing:distributed_tanks:y_ratio_end",
                wrt=y_ratio_end_name,
                method="exact",
            )
            self.declare_partials(
                of="data:weight:airframe:wing:distributed_tanks:start_chord",
                wrt=chord_start_name,
                method="exact",
            )
            self.declare_partials(
                of="data:weight:airframe:wing:distributed_tanks:chord_slope",
                wrt=chord_slope_name,
                method="exact",
            )

            self.declare_partials(
                of="data:weight:airframe:wing:distributed_tanks:fuel_inside",
                wrt=mass_name,
                method="exact",
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        distributed_tanks_y_ratio_start = np.array([])
        distributed_tanks_y_ratio_end = np.array([])
        distributed_tanks_chord_start = np.array([])
        distributed_tanks_chord_slope = np.array([])
        distributed_tanks_masses = np.array([])

        for distributed_tank_name, distributed_tank_type in zip(
            self.curated_name_list, self.curated_type_list
        ):
            y_ratio_start_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:y_ratio_start"
            )
            y_ratio_end_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:y_ratio_end"
            )
            chord_start_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:start_chord"
            )
            chord_slope_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:chord_slope"
            )
            mass_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":fuel_total_mission"
            )

            distributed_tanks_y_ratio_start = np.append(
                distributed_tanks_y_ratio_start, inputs[y_ratio_start_name]
            )
            distributed_tanks_y_ratio_end = np.append(
                distributed_tanks_y_ratio_end, inputs[y_ratio_end_name]
            )
            distributed_tanks_chord_start = np.append(
                distributed_tanks_chord_start, inputs[chord_start_name]
            )
            distributed_tanks_chord_slope = np.append(
                distributed_tanks_chord_slope, inputs[chord_slope_name]
            )
            distributed_tanks_masses = np.append(distributed_tanks_masses, inputs[mass_name])

        outputs["data:weight:airframe:wing:distributed_tanks:y_ratio_start"] = (
            distributed_tanks_y_ratio_start
        )
        outputs["data:weight:airframe:wing:distributed_tanks:y_ratio_end"] = (
            distributed_tanks_y_ratio_end
        )
        outputs["data:weight:airframe:wing:distributed_tanks:start_chord"] = (
            distributed_tanks_chord_start
        )
        outputs["data:weight:airframe:wing:distributed_tanks:chord_slope"] = (
            distributed_tanks_chord_slope
        )
        outputs["data:weight:airframe:wing:distributed_tanks:fuel_inside"] = (
            distributed_tanks_masses
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        name_and_type_array = np.rec.fromarrays([self.curated_name_list, self.curated_type_list])
        nb_punctual_masses = len(self.curated_name_list)

        for idx, value in enumerate(name_and_type_array.tolist()):
            distributed_tank_name = value[0]
            distributed_tank_type = value[1]

            y_ratio_start_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:y_ratio_start"
            )
            y_ratio_end_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:y_ratio_end"
            )
            chord_start_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:start_chord"
            )
            chord_slope_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":distributed_tanks:chord_slope"
            )
            mass_name = (
                PT_DATA_PREFIX
                + distributed_tank_type
                + ":"
                + distributed_tank_name
                + ":fuel_total_mission"
            )

            partials_value = np.zeros(nb_punctual_masses)
            partials_value[idx] = 1.0

            partials[
                "data:weight:airframe:wing:distributed_tanks:y_ratio_start", y_ratio_start_name
            ] = partials_value
            partials[
                "data:weight:airframe:wing:distributed_tanks:y_ratio_end", y_ratio_end_name
            ] = partials_value
            partials[
                "data:weight:airframe:wing:distributed_tanks:start_chord", chord_start_name
            ] = partials_value
            partials[
                "data:weight:airframe:wing:distributed_tanks:chord_slope", chord_slope_name
            ] = partials_value
            partials["data:weight:airframe:wing:distributed_tanks:fuel_inside", mass_name] = (
                partials_value
            )
