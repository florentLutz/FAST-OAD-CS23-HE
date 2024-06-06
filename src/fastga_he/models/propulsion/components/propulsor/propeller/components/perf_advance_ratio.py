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
        self.add_input(
            name="settings:propulsion:he_power_train:propeller:"
            + propeller_id
            + ":effective_advance_ratio",
            val=1.0,
            desc="Decrease in power coefficient due to installation effects of the propeller",
        )

        self.add_output("advance_ratio", val=0.7, shape=number_of_points)

        self.declare_partials(
            of="*",
            wrt=[
                "settings:propulsion:he_power_train:propeller:"
                + propeller_id
                + ":effective_advance_ratio",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="*",
            wrt=["rpm", "true_airspeed"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        effective_j = inputs[
            "settings:propulsion:he_power_train:propeller:"
            + propeller_id
            + ":effective_advance_ratio"
        ]

        outputs["advance_ratio"] = (
            inputs["true_airspeed"] / (inputs["rpm"] / 60.0 * diameter) * effective_j
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]

        diameter = inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"]
        effective_j = inputs[
            "settings:propulsion:he_power_train:propeller:"
            + propeller_id
            + ":effective_advance_ratio"
        ]

        partials["advance_ratio", "true_airspeed"] = (
            1.0 / (inputs["rpm"] / 60.0 * diameter) * effective_j
        )
        partials["advance_ratio", "rpm"] = -(
            inputs["true_airspeed"] / (inputs["rpm"] ** 2.0 / 60.0 * diameter) * effective_j
        )
        partials[
            "advance_ratio",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = (
            -inputs["true_airspeed"] / (inputs["rpm"] / 60.0 * diameter ** 2.0) * effective_j
        )
        partials[
            "advance_ratio",
            "settings:propulsion:he_power_train:propeller:"
            + propeller_id
            + ":effective_advance_ratio",
        ] = inputs["true_airspeed"] / (inputs["rpm"] / 60.0 * diameter)
