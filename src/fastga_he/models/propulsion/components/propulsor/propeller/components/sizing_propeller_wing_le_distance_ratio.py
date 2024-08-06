# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

from ..constants import POSSIBLE_POSITION


class SizingPropellerWingLEDistanceRatio(om.ExplicitComponent):
    """
    Computation of the distance between the propeller and the wing leading edge as a ratio of the
    reference chord behind the propeller
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
        position = self.options["position"]
        propeller_id = self.options["propeller_id"]

        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
            val=np.nan,
            units="m",
            desc="Value of the wing chord behind the propeller",
        )

        self.add_output(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
            val=0.5,
            desc="Distance between the propeller and the wing leading edge as a ratio of the "
            "reference chord behind the propeller",
        )

        if position == "on_the_wing":
            self.add_input(
                name="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_LE",
                val=np.nan,
                units="m",
                desc="Distance between the propeller and the leading edge",
            )

        else:
            self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        position = self.options["position"]
        propeller_id = self.options["propeller_id"]

        if position == "on_the_wing":
            distance_from_le = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_LE"
            ]

        else:
            distance_from_le = inputs["data:geometry:wing:MAC:at25percent:x"]

        outputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio"
        ] = (
            distance_from_le
            / inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        position = self.options["position"]
        propeller_id = self.options["propeller_id"]

        c_ref = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref"
        ]

        if position == "on_the_wing":
            distance_from_le = inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_LE"
            ]
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_LE",
            ] = 1.0 / c_ref

        else:
            distance_from_le = inputs["data:geometry:wing:MAC:at25percent:x"]
            partials[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
                "data:geometry:wing:MAC:at25percent:x",
            ] = 1.0 / c_ref

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
        ] = -distance_from_le / c_ref**2.0
