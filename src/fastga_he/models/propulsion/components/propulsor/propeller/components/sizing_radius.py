# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingPropellerRadius(om.ExplicitComponent):
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
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":hub_diameter",
            units="m",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            units="m",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius",
            shape=elements_number,
            units="m",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        radius_min = (
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":hub_diameter"]
            / 2.0
        )
        radius_max = (
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"] / 2.0
        )
        length = radius_max - radius_min
        elements_number = np.arange(self.options["elements_number"])
        element_length = length / self.options["elements_number"]

        radius = radius_min + (elements_number + 0.5) * element_length

        outputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius"
        ] = radius

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        elements_number = np.arange(self.options["elements_number"])

        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":hub_diameter",
        ] = (1.0 - (elements_number + 0.5) / self.options["elements_number"]) / 2.0
        partials[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = (
            (elements_number + 0.5) / self.options["elements_number"] / 2.0
        )
