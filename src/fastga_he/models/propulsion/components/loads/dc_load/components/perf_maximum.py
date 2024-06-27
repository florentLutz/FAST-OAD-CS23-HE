# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum power requested by the aux load.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="aux_load_id",
            default=None,
            desc="Identifier of the auxiliary load",
            allow_none=False,
        )

    def setup(self):
        aux_load_id = self.options["aux_load_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "power_in",
            val=np.full(number_of_points, np.nan),
            units="W",
            desc="Power at the input side of the load",
        )

        self.add_output(
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max",
            units="W",
            val=10e3,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max",
            wrt="power_in",
            method="exact",
            rows=np.zeros(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        aux_load_id = self.options["aux_load_id"]

        outputs["data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max"] = np.max(
            inputs["power_in"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        aux_load_id = self.options["aux_load_id"]

        partials[
            "data:propulsion:he_power_train:aux_load:" + aux_load_id + ":power_max",
            "power_in",
        ] = np.where(inputs["power_in"] == np.max(inputs["power_in"]), 1.0, 0.0)
