# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingPropellerFlappedRatio(om.ExplicitComponent):
    """
    Computation the percent of the span, behind the propeller that is "flapped". If the flaps end
    at the center of the propeller this ratio is gonna be equal equal to 50%, if there are no
    flaps, it is gonna be equal to 0%.
    """

    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            name="position",
            default="on_the_wing",
            values=POSSIBLE_POSITION,
            desc="Option to give the position of the propeller, possible position include "
            + ", ".join(POSSIBLE_POSITION),
            allow_none=False,
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio",
            val=np.nan,
            desc="Location of the propeller along the span as a fraction of the span",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:"
            + propeller_id
            + ":diameter_to_span_ratio",
            val=np.nan,
            desc="Diameter of the propeller as a ratio of the wing half span",
        )
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)

        self.add_output(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
            val=0.0,
            desc="Portion of the span, downstream of the propeller, which has flaps",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        flap_span_ratio = inputs["data:geometry:flap:span_ratio"]
        prop_y_ratio = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio"
        ]
        prop_span_to_dia_ratio = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_span_ratio"
        ]

        if self.options["position"] == "on_the_wing":
            flapped_ratio = np.clip(
                flap_span_ratio / (prop_y_ratio + 0.5 * prop_span_to_dia_ratio), 0.0, 1.0
            )
        else:
            flapped_ratio = 0.0

        outputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio"
        ] = flapped_ratio

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]

        flap_span_ratio = inputs["data:geometry:flap:span_ratio"]
        prop_y_ratio = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio"
        ]
        prop_span_to_dia_ratio = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_span_ratio"
        ]

        if (
            abs(prop_y_ratio - flap_span_ratio) > 0.5 * prop_span_to_dia_ratio
            or self.options["position"] != "on_the_wing"
        ):
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                "data:geometry:flap:span_ratio",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio",
            ] = 0.0
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                "data:propulsion:he_power_train:propeller:"
                + propeller_id
                + ":diameter_to_span_ratio",
            ] = 0.0
        else:

            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                "data:geometry:flap:span_ratio",
            ] = 1.0 / (prop_y_ratio + 0.5 * prop_span_to_dia_ratio)
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":y_ratio",
            ] = (
                -flap_span_ratio / (prop_y_ratio + 0.5 * prop_span_to_dia_ratio) ** 2.0
            )
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":flapped_ratio",
                "data:propulsion:he_power_train:propeller:"
                + propeller_id
                + ":diameter_to_span_ratio",
            ] = (
                -flap_span_ratio / (prop_y_ratio + 0.5 * prop_span_to_dia_ratio) ** 2.0 * 0.5
            )
