# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingFuelTankDrag(om.ExplicitComponent):
    """
    Class that computes the contribution to profile drag of the fuel tanks according to the
    position given in the options. For now this will be 0.0 regardless of the option except when
    in a pod.
    """

    def initialize(self):
        self.options.declare(
            name="fuel_tank_id",
            default=None,
            desc="Identifier of the fuel tank",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the fuel tank, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )
        # Not as useful as the ones in aerodynamics, here it will just be run twice in the sizing
        # group
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]
        # For refractoring purpose we just match the option to the tag in the variable name and
        # use it
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        # At least one input is needed regardless of the case
        self.add_input(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width",
            units="m",
            val=np.nan,
            desc="Width of the battery, as in the size of the battery along the Y-axis",
        )

        if position == "wing_pod":
            self.add_input(
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height",
                units="m",
                val=np.nan,
                desc="Height of the battery, as in the size of the battery along the Z-axis",
            )
            self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":" + ls_tag + ":CD0",
            val=0.0,
        )

    def setup_partials(self):
        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        if position == "wing_pod":
            # According to :cite:`gudmundsson:2013`. the drag of a streamlined external tank can
            # be computed using the following formula. It highly depends on the tank/wing
            # interface so we will take a middle. Also, there is no dependency on the tank length

            wing_area = inputs["data:geometry:wing:area"]

            frontal_area = (
                np.pi
                * inputs[
                    "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width"
                ]
                * inputs[
                    "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height"
                ]
                / 4.0
            )

            cd0 = 0.10 * frontal_area / wing_area

        else:
            cd0 = 0.0

        outputs[
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":" + ls_tag + ":CD0"
        ] = cd0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]
        ls_tag = "low_speed" if self.options["low_speed_aero"] else "cruise"

        if position == "wing_pod":
            frontal_area = (
                np.pi
                * inputs[
                    "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width"
                ]
                * inputs[
                    "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height"
                ]
                / 4.0
            )

            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":" + ls_tag + ":CD0",
                "data:geometry:wing:area",
            ] = -0.10 * frontal_area / inputs["data:geometry:wing:area"] ** 2.0
            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":" + ls_tag + ":CD0",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width",
            ] = (
                0.10
                * np.pi
                * inputs[
                    "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height"
                ]
                / (4.0 * inputs["data:geometry:wing:area"])
            )
            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":" + ls_tag + ":CD0",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height",
            ] = (
                0.10
                * np.pi
                * inputs[
                    "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width"
                ]
                / (4.0 * inputs["data:geometry:wing:area"])
            )

        else:
            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":" + ls_tag + ":CD0",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:width",
            ] = 0.0
