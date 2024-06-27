# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingRectifierHeatSinkDimension(om.ExplicitComponent):
    """
    Computation of the dimension of the heat sink of the component (plaque froide in french).
    Implementation of the formula from :cite:`giraud:2014`.
    """

    def initialize(self):

        self.options.declare(
            name="rectifier_id",
            default=None,
            desc="Identifier of the rectifier",
            allow_none=False,
        )

    def setup(self):

        rectifier_id = self.options["rectifier_id"]

        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:length",
            units="m",
            val=np.nan,
            desc="Length of one module",
        )
        self.add_input(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:width",
            units="m",
            val=np.nan,
            desc="Width of one module",
        )

        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":heat_sink:length",
            units="m",
            val=0.20,
            desc="Length of the heat sink",
        )
        self.add_output(
            name="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":heat_sink:width",
            units="m",
            val=0.17,
            desc="Width of the heat sink",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":heat_sink:length",
            wrt="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:width",
            method="exact",
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":heat_sink:width",
            wrt="data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:length",
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        rectifier_id = self.options["rectifier_id"]

        outputs[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":heat_sink:length"
        ] = (
            1.1
            * inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:width"]
            * 3.0
        )
        outputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":heat_sink:width"] = (
            1.1
            * inputs["data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:length"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        rectifier_id = self.options["rectifier_id"]

        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":heat_sink:length",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:width",
        ] = (
            1.1 * 3.0
        )
        partials[
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":heat_sink:width",
            "data:propulsion:he_power_train:rectifier:" + rectifier_id + ":module:length",
        ] = 1.1
