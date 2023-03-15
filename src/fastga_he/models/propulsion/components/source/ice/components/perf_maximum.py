# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesMaximum(om.ExplicitComponent):
    """
    Class to identify the maximum sea level equivalent power to use for the ICE sizing.
    """

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="ice_id",
            default=None,
            desc="Identifier of the Internal Combustion Engine",
            allow_none=False,
        )

    def setup(self):

        ice_id = self.options["ice_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input("equivalent_SL_power", units="W", val=np.full(number_of_points, np.nan))

        self.add_output(
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_max_SL",
            units="W",
            val=250e3,
            desc="Maximum power the motor has to provide at Sea Level",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ice_id = self.options["ice_id"]

        outputs["data:propulsion:he_power_train:ICE:" + ice_id + ":power_max_SL"] = np.max(
            inputs["equivalent_SL_power"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        ice_id = self.options["ice_id"]

        partials[
            "data:propulsion:he_power_train:ICE:" + ice_id + ":power_max_SL",
            "equivalent_SL_power",
        ] = np.where(
            inputs["equivalent_SL_power"] == np.max(inputs["equivalent_SL_power"]), 1.0, 0.0
        )
