# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDCSSPCEfficiency(om.ExplicitComponent):
    """
    This module computes the efficiency of the SSPC.
    """

    def initialize(self):

        self.options.declare(
            name="dc_sspc_id",
            default=None,
            desc="Identifier of the DC SSPC",
            allow_none=False,
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            "closed",
            default=True,
            desc="Boolean to choose whether the breaker is closed or not.",
            types=bool,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        dc_sspc_id = self.options["dc_sspc_id"]

        self.add_input(
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":efficiency",
            val=np.nan,
            desc="Value of the SSPC efficiency, assumed constant during operations (eases convergence)",
        )

        self.add_output(
            "efficiency",
            val=np.full(number_of_points, 1.0),
        )

        if self.options["closed"]:
            self.declare_partials(
                of="*",
                wrt="*",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
                val=np.ones(number_of_points),
            )
        else:
            self.declare_partials(
                of="*",
                wrt="*",
                method="exact",
                rows=np.arange(number_of_points),
                cols=np.zeros(number_of_points),
                val=np.zeros(number_of_points),
            )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]
        dc_sspc_id = self.options["dc_sspc_id"]

        if self.options["closed"]:
            efficiency = np.full(
                number_of_points,
                inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":efficiency"],
            )
        else:
            efficiency = np.ones(number_of_points)

        outputs["efficiency"] = efficiency
