# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION

UNDERBELLY_RATIO = 0.8  # Ratio between underbelly width and fuselage width
CHORD_RATIO = 2.0 / 3.0  # Ratio between chord occupied by batteries and total chord
SPAN_RATIO = 0.8  # Ratio between half-span occupied by batteries and total half-span


class SizingBatteryDimensions(om.ExplicitComponent):
    """
    Computation of the different dimensions of the battery, il will heavily depend on the
    position of the battery. If the batteries are in the rear, front or in pods,
    we will use ratios. If the batteries are in the underbelly/wing, we will use fuselage/wing
    dimensions.
    """

    def initialize(self):
        self.options.declare(
            name="battery_pack_id",
            default=None,
            desc="Identifier of the battery pack",
            allow_none=False,
        )
        self.options.declare(
            name="position",
            default="inside_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the battery, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):
        battery_pack_id = self.options["battery_pack_id"]
        position = self.options["position"]

        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":dimension:length",
            units="m",
            val=0.5,
            desc="Length of the battery, as in the size of the battery along the X-axis",
        )
        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":dimension:width",
            units="m",
            val=0.5,
            desc="Width of the battery, as in the size of the battery along the Y-axis",
        )
        self.add_output(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":dimension:height",
            units="m",
            val=0.5,
            desc="Height of the battery, as in the size of the battery along the Z-axis",
        )

        self.add_input(
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            units="m**3",
            val=np.nan,
            desc="Volume of the battery pack",
        )

        if position in ["in_the_front", "in_the_back"]:
            # If in the fuselage, more of a box shape
            self.add_input(
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height_length_ratio",
                val=1.4,
                desc="Ratio between the battery height and length",
            )
            self.add_input(
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width_length_ratio",
                val=2.06,
                desc="Ratio between the battery width and length",
            )

        elif position == "wing_pod":
            # If in the fuselage, more of a box shape
            self.add_input(
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:fineness_ratio",
                val=15.0,
                desc="Ratio between the battery width/height and length",
            )

        elif position == "underbelly":
            self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
            self.add_input("data:geometry:cabin:length", val=np.nan, units="m")

        else:
            self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
            self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
            self.add_input("data:geometry:wing:span", val=np.nan, units="m")

        self.declare_partials(of="*", wrt="*")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        battery_pack_id = self.options["battery_pack_id"]
        position = self.options["position"]

        battery_volume = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume"
        ]

        if position in ["in_the_front", "in_the_back"]:
            hl_ratio = inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height_length_ratio"
            ]
            wl_ratio = inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width_length_ratio"
            ]

            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length"
            ] = (battery_volume / hl_ratio / wl_ratio) ** (1.0 / 3.0)
            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width"
            ] = (battery_volume / hl_ratio) ** (1.0 / 3.0) * wl_ratio ** (2.0 / 3.0)
            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height"
            ] = (battery_volume / wl_ratio) ** (1.0 / 3.0) * hl_ratio ** (2.0 / 3.0)

        elif position == "wing_pod":
            fineness = inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:fineness_ratio"
            ]

            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length"
            ] = (battery_volume * fineness**2.0) ** (1.0 / 3.0)
            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width"
            ] = (battery_volume / fineness) ** (1.0 / 3.0)
            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height"
            ] = (battery_volume / fineness) ** (1.0 / 3.0)

        elif position == "underbelly":
            cabin_length = inputs["data:geometry:cabin:length"]
            max_width = inputs["data:geometry:fuselage:maximum_width"]

            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length"
            ] = cabin_length
            # 0.8 Ratio to account for the fact that the underbelly is not as wide as the fuselage
            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width"
            ] = UNDERBELLY_RATIO * max_width
            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height"
            ] = battery_volume / (cabin_length * UNDERBELLY_RATIO * max_width)

        else:
            tip_chord = inputs["data:geometry:wing:tip:chord"]
            root_chord = inputs["data:geometry:wing:root:chord"]

            half_span = inputs["data:geometry:wing:span"] / 2.0

            # Take average chord plus coefficient to leave space in front and behind
            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length"
            ] = (tip_chord + root_chord) / 2.0 * CHORD_RATIO
            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width"
            ] = SPAN_RATIO * half_span
            outputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height"
            ] = battery_volume / (
                (tip_chord + root_chord) / 2.0 * CHORD_RATIO * half_span * SPAN_RATIO
            )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        battery_pack_id = self.options["battery_pack_id"]
        position = self.options["position"]

        battery_volume = inputs[
            "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume"
        ]

        if position in ["in_the_front", "in_the_back"]:
            hl_ratio = inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height_length_ratio"
            ]
            wl_ratio = inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width_length_ratio"
            ]

            # Partials for length
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = 1.0 / 3.0 * (hl_ratio * wl_ratio) ** (-1.0 / 3.0) * battery_volume ** (-2.0 / 3.0)
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height_length_ratio",
            ] = -1.0 / 3.0 * (battery_volume / wl_ratio) ** (1.0 / 3.0) * hl_ratio ** (-4.0 / 3.0)
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width_length_ratio",
            ] = -1.0 / 3.0 * (battery_volume / hl_ratio) ** (1.0 / 3.0) * wl_ratio ** (-4.0 / 3.0)

            # Partials for width
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = (
                1.0
                / 3.0
                * hl_ratio ** (-1.0 / 3.0)
                * wl_ratio ** (2.0 / 3.0)
                * battery_volume ** (-2.0 / 3.0)
            )
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width_length_ratio",
            ] = 2.0 / 3.0 * (battery_volume / hl_ratio) ** (1.0 / 3.0) * wl_ratio ** (-1.0 / 3.0)
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height_length_ratio",
            ] = (
                -1.0
                / 3.0
                * battery_volume ** (1.0 / 3.0)
                * (wl_ratio / hl_ratio**2.0) ** (2.0 / 3.0)
            )

            # Partials for height
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = (
                1.0
                / 3.0
                * wl_ratio ** (-1.0 / 3.0)
                * hl_ratio ** (2.0 / 3.0)
                * battery_volume ** (-2.0 / 3.0)
            )
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height_length_ratio",
            ] = 2.0 / 3.0 * (battery_volume / wl_ratio) ** (1.0 / 3.0) * hl_ratio ** (-1.0 / 3.0)
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width_length_ratio",
            ] = (
                -1.0
                / 3.0
                * battery_volume ** (1.0 / 3.0)
                * (hl_ratio / wl_ratio**2.0) ** (2.0 / 3.0)
            )

        elif position == "wing_pod":
            fineness = inputs[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:fineness_ratio"
            ]

            # Partials for length
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = 1.0 / 3.0 * (fineness / battery_volume) ** (2.0 / 3.0)
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:fineness_ratio",
            ] = 2.0 / 3.0 * (battery_volume * fineness**-1.0) ** (1.0 / 3.0)

            # Partials for width
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = 1.0 / 3.0 * (battery_volume**-2.0 / fineness) ** (1.0 / 3.0)
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:fineness_ratio",
            ] = -1.0 / 3.0 * (battery_volume * fineness**-4.0) ** (1.0 / 3.0)

            # Partials for height
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = 1.0 / 3.0 * (battery_volume**-2.0 / fineness) ** (1.0 / 3.0)
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:fineness_ratio",
            ] = -1.0 / 3.0 * (battery_volume * fineness**-4.0) ** (1.0 / 3.0)

        elif position == "underbelly":
            cabin_length = inputs["data:geometry:cabin:length"]
            max_width = inputs["data:geometry:fuselage:maximum_width"]

            # Partials for length
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:geometry:cabin:length",
            ] = 1.0
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:geometry:fuselage:maximum_width",
            ] = 0.0

            # Partials for width
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:geometry:cabin:length",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:geometry:fuselage:maximum_width",
            ] = UNDERBELLY_RATIO

            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = 1.0 / (cabin_length * UNDERBELLY_RATIO * max_width)
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:geometry:cabin:length",
            ] = -battery_volume / (cabin_length**2.0 * UNDERBELLY_RATIO * max_width)
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:geometry:fuselage:maximum_width",
            ] = -battery_volume / (cabin_length * UNDERBELLY_RATIO * max_width**2.0)

        else:
            tip_chord = inputs["data:geometry:wing:tip:chord"]
            root_chord = inputs["data:geometry:wing:root:chord"]

            half_span = inputs["data:geometry:wing:span"] / 2.0

            # Take average chord plus coefficient to leave space in front and behind
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:geometry:wing:tip:chord",
            ] = CHORD_RATIO / 2.0
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:geometry:wing:root:chord",
            ] = CHORD_RATIO / 2.0
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:length",
                "data:geometry:wing:span",
            ] = 0.0

            # Partials for width
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:geometry:wing:tip:chord",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:geometry:wing:root:chord",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:width",
                "data:geometry:wing:span",
            ] = SPAN_RATIO / 2.0

            # Partials for height
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:propulsion:he_power_train:battery_pack:" + battery_pack_id + ":volume",
            ] = 1.0 / ((tip_chord + root_chord) / 2.0 * CHORD_RATIO * half_span * SPAN_RATIO)
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:geometry:wing:tip:chord",
            ] = -battery_volume / (
                (tip_chord + root_chord) ** 2.0 / 2.0 * CHORD_RATIO * half_span * SPAN_RATIO
            )
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:geometry:wing:root:chord",
            ] = -battery_volume / (
                (tip_chord + root_chord) ** 2.0 / 2.0 * CHORD_RATIO * half_span * SPAN_RATIO
            )
            partials[
                "data:propulsion:he_power_train:battery_pack:"
                + battery_pack_id
                + ":dimension:height",
                "data:geometry:wing:span",
            ] = (
                -battery_volume
                / ((tip_chord + root_chord) / 2.0 * CHORD_RATIO * half_span**2.0 * SPAN_RATIO)
                * 0.5
            )
