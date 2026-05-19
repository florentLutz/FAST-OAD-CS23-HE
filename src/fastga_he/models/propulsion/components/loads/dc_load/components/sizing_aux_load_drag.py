# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingDCAuxLoadDrag(om.ExplicitComponent):
    """
    Class that computes the drag coefficient of the auxiliary load based on its position. Will
    be 0.0 all the time as we wil make the assumption that it is "inside" any part of the aircraft
    it is located in.
    """

    def initialize(self):
        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="in_the_back",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the auxiliary load, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        # For refractoring purpose we just match the option to the tag in the variable name and
        # use it
        aux_load_id = self.options["aux_load_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        self.add_input(
            "data:TLAR:v_cruise",
            val=np.nan,
            units="m/s",
        )
        self.add_input(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cruise_air_density",
            val=np.nan,
            units="kg/m**3",
        )
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass",
            val=np.nan,
            units="kg",
        )
        self.add_input(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":weight_to_drag_ratio",
            val=3.0,
            units="unitless",
        )

        self.add_output(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":" + ls_tag + ":CD0",
            val=0.0,
        )

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aux_load_id = self.options["aux_load_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        v_cruise = inputs["data:TLAR:v_cruise"]
        air_density = inputs[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cruise_air_density"
        ]
        wing_area = inputs["data:geometry:wing:area"]
        aux_load_mass = inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass"]
        weight_to_drag_ratio = inputs[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":weight_to_drag_ratio"
        ]

        outputs[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":" + ls_tag + ":CD0"
        ] = (aux_load_mass / weight_to_drag_ratio) / (0.5 * air_density * v_cruise**2 *
                                                      wing_area)*9.81

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aux_load_id = self.options["aux_load_id"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        v_cruise = inputs["data:TLAR:v_cruise"]
        air_density = inputs[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cruise_air_density"
        ]
        wing_area = inputs["data:geometry:wing:area"]
        aux_load_mass = inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass"]
        weight_to_drag_ratio = inputs[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":weight_to_drag_ratio"
        ]

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":" + ls_tag + ":CD0",
            "data:TLAR:v_cruise",
        ] = (
            -2
            * (aux_load_mass / weight_to_drag_ratio)
            / (0.5 * air_density * v_cruise**3 * wing_area)
        )*9.81

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":" + ls_tag + ":CD0",
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":cruise_air_density",
        ] = (
            -1
            * (aux_load_mass / weight_to_drag_ratio)
            / (0.5 * air_density**2 * v_cruise**2 * wing_area)
        )*9.81

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":" + ls_tag + ":CD0",
            "data:geometry:wing:area",
        ] = (
            -1
            * (aux_load_mass / weight_to_drag_ratio)
            / (0.5 * air_density * v_cruise**2 * wing_area**2)
        )*9.81

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":" + ls_tag + ":CD0",
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass",
        ] = 1 / weight_to_drag_ratio / (0.5 * air_density * v_cruise**2 * wing_area)*9.81

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":" + ls_tag + ":CD0",
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":weight_to_drag_ratio",
        ] = (
            -1
            * (aux_load_mass / weight_to_drag_ratio**2)
            / (0.5 * air_density * v_cruise**2 * wing_area)
        )*9.81
