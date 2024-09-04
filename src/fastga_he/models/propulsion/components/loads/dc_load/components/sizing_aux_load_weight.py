# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingDCAuxLoadWeight(om.ExplicitComponent):
    """
    Computation of the weight of the auxiliary load. Will be based on a simple power density.
    """

    def initialize(self):
        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )

    def setup(self):
        aux_load_id = self.options["aux_load_id"]

        self.add_input(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_rating",
            units="kW",
            val=np.nan,
            desc="Max continuous power of the auxiliary load",
        )
        self.add_input(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_density",
            units="kW/kg",
            val=np.nan,
            desc="Power density of the auxiliary load",
        )

        self.add_output(
            name="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass",
            val=10.0,
            units="kg",
            desc="Mass of the auxiliary load",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aux_load_id = self.options["aux_load_id"]

        outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass"] = (
            inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_rating"]
            / inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_density"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        aux_load_id = self.options["aux_load_id"]

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass",
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_rating",
        ] = (
            1.0
            / inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_density"]
        )
        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":mass",
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_density",
        ] = (
            -inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_rating"]
            / inputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_density"]
            ** 2.0
        )
