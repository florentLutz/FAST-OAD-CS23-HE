# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingPropellerChord(om.ExplicitComponent):
    def initialize(self):

        self.options.declare(
            name="propeller_id",
            default=None,
            desc="Identifier of the propeller",
            allow_none=False,
        )
        self.options.declare("elements_number", default=7, types=int)

    def setup(self):

        propeller_id = self.options["propeller_id"]
        elements_number = self.options["elements_number"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius",
            shape=elements_number,
            units="m",
            val=np.full(elements_number, np.nan),
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            units="m",
            val=np.nan,
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":radius_ratio_vect",
            val=np.nan,
            shape_by_conn=True,
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":chord_vect",
            val=np.nan,
            units="m",
            shape_by_conn=True,
        )

        self.add_output(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_chord",
            shape=elements_number,
            units="m",
        )

        self.declare_partials(of="*", wrt="*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        radius_max = (
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"] / 2.0
        )

        chord = np.interp(
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius"]
            / radius_max,
            inputs[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":radius_ratio_vect"
            ],
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":chord_vect"],
        )

        outputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_chord"
        ] = chord
