# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesUpdateSOC(om.ExplicitComponent):
    """
    Computation of the evolutions of the state of charge of the battery based on the variation
    computed.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "state_of_charge_decrease", units="percent", val=np.full(number_of_points, np.nan)
        )

        self.add_output("state_of_charge", units="percent", val=np.full(number_of_points, 100.0))

        partials = -(np.tri(number_of_points, number_of_points) - np.eye(number_of_points))

        self.declare_partials(
            of="state_of_charge",
            wrt="state_of_charge_decrease",
            method="exact",
            val=-np.ones(len(np.where(partials == -1)[0])),
            rows=np.where(partials == -1)[0],
            cols=np.where(partials == -1)[1],
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        number_of_points = self.options["number_of_points"]

        outputs["state_of_charge"] = np.full(number_of_points, 100.0) - np.cumsum(
            np.concatenate((np.zeros(1), inputs["state_of_charge_decrease"][:-1]))
        )
