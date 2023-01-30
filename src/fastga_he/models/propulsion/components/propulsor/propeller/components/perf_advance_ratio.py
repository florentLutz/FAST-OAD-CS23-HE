# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class PerformancesAdvanceRatio(om.ExplicitComponent):
    """Computation of the advance ratio of the propeller."""

    def initialize(self):

        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )

    def setup(self):

        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the propeller",
        )
        self.add_input("rpm", units="min**-1", val=np.nan, shape=number_of_points)
        self.add_input("true_airspeed", units="m/s", val=50.0, shape=number_of_points)

        self.add_output("advance_ratio", val=0.7, shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]

        outputs["advance_ratio"] = inputs["true_airspeed"] / (inputs["rpm"] / 60.0 * diameter)

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]

        partials["advance_ratio", "true_airspeed"] = np.diag(
            1.0 / (inputs["rpm"] / 60.0 * diameter)
        )
        partials["advance_ratio", "rpm"] = -np.diag(
            inputs["true_airspeed"] / (inputs["rpm"] ** 2.0 / 60.0 * diameter)
        )
        partials[
            "advance_ratio",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = -inputs["true_airspeed"] / (inputs["rpm"] / 60.0 * diameter ** 2.0)
