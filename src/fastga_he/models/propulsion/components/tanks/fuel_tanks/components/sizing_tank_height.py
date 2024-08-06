# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np

import openmdao.api as om

from ..constants import POSSIBLE_POSITION

FLOOR_HEIGHT_RATIO = 0.1  # Ratio between the fuselage height and the floor height
THICKNESS_MARGIN_RATIO = 0.1


class SizingFuelTankHeight(om.ExplicitComponent):
    """
    Computation of the reference height for the computation of the tank width. If the tank is in
    a pod, it will depend on volume and a fineness ratio. If it is in the wing, it will depend on
    the wing chord and thickness ratio. If it is in the fuselage it depend on the fuselage max
    height.
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

    def setup(self):
        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        self.add_output(
            "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height",
            val=0.1,
            units="m",
            desc="Value of the length of the tank in the z-direction, computed differently based "
            "on the location of the tank",
        )

        if position == "inside_the_wing":
            self.add_input(
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:ref_chord",
                val=0.0,
                units="m",
                desc="Reference wing chord for the tank",
            )
            self.add_input("data:geometry:wing:thickness_ratio", val=np.nan)

            self.declare_partials(of="*", wrt="*", method="exact")

        elif position == "wing_pod":
            self.add_input(
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume",
                units="m**3",
                val=np.nan,
                desc="Capacity of the tank in terms of volume",
            )
            self.add_input(
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":dimension:fineness_ratio",
                val=15.0,
                desc="Ratio between the wing pod length and width/height ",
            )

            self.declare_partials(of="*", wrt="*", method="exact")

        else:
            self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")

            self.declare_partials(
                of="*", wrt="data:geometry:fuselage:maximum_height", val=FLOOR_HEIGHT_RATIO
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        if position == "inside_the_wing":
            ref_chord = inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:ref_chord"
            ]
            thickness_ratio = inputs["data:geometry:wing:thickness_ratio"]

            outputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height"
            ] = ref_chord * thickness_ratio * (1.0 - THICKNESS_MARGIN_RATIO)

        elif position == "wing_pod":
            fineness_ratio = inputs[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":dimension:fineness_ratio"
            ]
            tank_volume = inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume"
            ]

            outputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height"
            ] = (tank_volume / fineness_ratio) ** (1.0 / 3.0)

        else:
            outputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height"
            ] = FLOOR_HEIGHT_RATIO * inputs["data:geometry:fuselage:maximum_height"]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        fuel_tank_id = self.options["fuel_tank_id"]
        position = self.options["position"]

        if position == "inside_the_wing":
            ref_chord = inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:ref_chord"
            ]
            thickness_ratio = inputs["data:geometry:wing:thickness_ratio"]

            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:ref_chord",
            ] = thickness_ratio * (1.0 - THICKNESS_MARGIN_RATIO)
            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height",
                "data:geometry:wing:thickness_ratio",
            ] = ref_chord * (1.0 - THICKNESS_MARGIN_RATIO)

        elif position == "wing_pod":
            fineness_ratio = inputs[
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":dimension:fineness_ratio"
            ]
            tank_volume = inputs[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume"
            ]

            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height",
                "data:propulsion:he_power_train:fuel_tank:"
                + fuel_tank_id
                + ":dimension:fineness_ratio",
            ] = (
                -1.0
                / 3.0
                * (tank_volume / fineness_ratio) ** (-2.0 / 3.0)
                * tank_volume
                / fineness_ratio**2.0
            )
            partials[
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":dimension:height",
                "data:propulsion:he_power_train:fuel_tank:" + fuel_tank_id + ":volume",
            ] = 1.0 / 3.0 * (tank_volume / fineness_ratio) ** (-2.0 / 3.0) / fineness_ratio
