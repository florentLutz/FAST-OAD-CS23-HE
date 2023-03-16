# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingICEUninstalledWeight(om.ExplicitComponent):
    """
    Computation of the uninstalled ICE weight, based on a formula from :cite:`gudmundsson:2013`
    """

    def initialize(self):

        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

    def setup(self):

        ice_id = self.options["ice_id"]

        self.add_input(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
            units="hp",
            val=np.nan,
            desc="Maximum power the motor can provide at Sea Level",
        )

        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":uninstalled_mass",
            units="lbm",
            val=400.0,
            desc="Uninstalled weight of the ICE engine",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ice_id = self.options["ice_id"]

        max_power = inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL"]
        uninstalled_mass = (max_power - 21.55) / 0.5515

        outputs[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":uninstalled_mass"
        ] = uninstalled_mass

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ice_id = self.options["ice_id"]

        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":uninstalled_mass",
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_rating_SL",
        ] = (
            1.0 / 0.5515
        )
