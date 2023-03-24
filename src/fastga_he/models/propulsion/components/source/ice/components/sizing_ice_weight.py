# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingICEWeight(om.ExplicitComponent):
    """
    Computation of the installed ICE weight, Based on a statistical analysis, see
    :cite:`raymer:2012`.
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
            "data:propulsion:he_power_train:ICE:" + ice_id + ":uninstalled_mass",
            units="kg",
            val=np.nan,
            desc="Uninstalled weight of the ICE engine",
        )

        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":mass",
            units="kg",
            val=250.0,
            desc="Installed weight of the ICE engine",
        )
        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":fuel_lines_mass",
            units="kg",
            val=250.0,
            desc="Installed weight of the ICE engine",
        )

        self.declare_partials(
            of="data:propulsion:he_power_train:ICE:" + ice_id + ":mass",
            wrt="data:propulsion:he_power_train:ICE:" + ice_id + ":uninstalled_mass",
            val=1.2 * 1.2,
        )
        self.declare_partials(
            of="data:propulsion:he_power_train:ICE:" + ice_id + ":fuel_lines_mass",
            wrt="data:propulsion:he_power_train:ICE:" + ice_id + ":uninstalled_mass",
            val=0.2,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ice_id = self.options["ice_id"]

        # The mass of fuel lines typically depends on the number of tank and engine, which is not
        # known a priori, instead, we will consider and added 20% weight, until a different
        # approach is found

        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":fuel_lines_mass"] = (
            0.2 * inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":uninstalled_mass"]
        )

        # We take 1.2 for the installed mass as now, the propeller weight is properly computed
        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":mass"] = (
            1.2 * 1.2 * inputs["data:propulsion:he_power_train:ICE:" + ice_id + ":uninstalled_mass"]
        )
