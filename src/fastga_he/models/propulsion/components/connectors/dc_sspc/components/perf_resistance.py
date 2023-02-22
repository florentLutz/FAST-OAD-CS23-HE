# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesDCSSPCResistance(om.ExplicitComponent):
    """
    Computation of the actual resistance during the mission. Depends on the state of the SSPC,
    either open or closed. When open it should also depend on the temperature, which we won't
    consider for now. We will also only consider a lumped resistance (sum of the diode and IGBT
    one) for practical purposes.
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
            desc="Boolean to choose whether the breaker is closed or " "not.",
            types=bool,
        )

    def setup(self):

        dc_sspc_id = self.options["dc_sspc_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:resistance",
            val=np.nan,
            units="ohm",
        )
        self.add_input(
            "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:resistance",
            val=np.nan,
            units="ohm",
        )

        self.add_output(
            "resistance_sspc",
            val=np.full(number_of_points, 1.0e-3),
            units="ohm",
            shape=number_of_points,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dc_sspc_id = self.options["dc_sspc_id"]
        number_of_points = self.options["number_of_points"]

        if self.options["closed"]:

            outputs["resistance_sspc"] = np.full(
                number_of_points,
                inputs["data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:resistance"]
                + inputs[
                    "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:resistance"
                ],
            )

        else:

            outputs["resistance_sspc"] = np.full(number_of_points, np.inf)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        dc_sspc_id = self.options["dc_sspc_id"]
        number_of_points = self.options["number_of_points"]

        if self.options["closed"]:

            partials[
                "resistance_sspc",
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:resistance",
            ] = np.full(number_of_points, 1.0)
            partials[
                "resistance_sspc",
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:resistance",
            ] = np.full(number_of_points, 1.0)

        else:

            partials[
                "resistance_sspc",
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":igbt:resistance",
            ] = np.zeros(number_of_points)
            partials[
                "resistance_sspc",
                "data:propulsion:he_power_train:DC_SSPC:" + dc_sspc_id + ":diode:resistance",
            ] = np.zeros(number_of_points)
