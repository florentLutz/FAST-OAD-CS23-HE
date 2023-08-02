# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

LIMIT_DIAMETER_RATIO = (0.25, 3.0)


class SlipstreamPropellerHeightImpact(om.ExplicitComponent):
    """
    Implementation of the surrogate model from :cite:`patterson:2016` to account for height
    impact on the slipstream effect. This component computes the beta coefficient.
    """

    def initialize(self):
        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="propeller_id", default=None, desc="Identifier of the propeller", allow_none=False
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        propeller_id = self.options["propeller_id"]

        self.add_input(
            name="f_0",
            val=np.nan,
            shape=number_of_points,
            desc="Order zero coefficient in the beta surrogate model",
        )
        self.add_input(
            name="f_1",
            val=np.nan,
            shape=number_of_points,
            desc="First order coefficient in the beta surrogate model",
        )
        self.add_input(
            name="f_2",
            val=np.nan,
            shape=number_of_points,
            desc="Second order coefficient in the beta surrogate model",
        )
        self.add_input(
            name="f_3",
            val=np.nan,
            shape=number_of_points,
            desc="Third order coefficient in the beta surrogate model",
        )
        self.add_input(
            name="f_4",
            val=np.nan,
            shape=number_of_points,
            desc="Fourth order coefficient in the beta surrogate model",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:"
            + propeller_id
            + ":diameter_to_chord_ratio",
            val=0.2,
            desc="Diameter of the propeller as a ratio of the wing chord behind the propeller",
        )

        self.add_output(name="beta", val=1.0, shape=number_of_points, desc="Height impact factor")

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        f_0 = inputs["f_0"]
        f_1 = inputs["f_1"]
        f_2 = inputs["f_2"]
        f_3 = inputs["f_3"]
        f_4 = inputs["f_4"]
        r_c = np.clip(
            inputs[
                "data:propulsion:he_power_train:propeller:"
                + propeller_id
                + ":diameter_to_chord_ratio"
            ]
            / 2.0,
            LIMIT_DIAMETER_RATIO[0],
            LIMIT_DIAMETER_RATIO[1],
        )

        # Fool proofing the results, just in case. Based on the graphs displayed in the source
        # material
        outputs["beta"] = np.clip(
            f_0 + f_1 * r_c + f_2 * r_c ** 2.0 + f_3 * r_c ** 3.0 + f_4 * r_c ** 4.0, 0.3, 1.5
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]
        number_of_points = self.options["number_of_points"]

        f_1 = inputs["f_1"]
        f_2 = inputs["f_2"]
        f_3 = inputs["f_3"]
        f_4 = inputs["f_4"]
        r_c_unclipped = (
            inputs[
                "data:propulsion:he_power_train:propeller:"
                + propeller_id
                + ":diameter_to_chord_ratio"
            ]
            / 2.0
        )
        r_c = np.clip(r_c_unclipped, LIMIT_DIAMETER_RATIO[0], LIMIT_DIAMETER_RATIO[1])

        partials["beta", "f_0"] = np.eye(number_of_points)
        partials["beta", "f_1"] = np.eye(number_of_points) * r_c
        partials["beta", "f_2"] = np.eye(number_of_points) * r_c ** 2.0
        partials["beta", "f_3"] = np.eye(number_of_points) * r_c ** 3.0
        partials["beta", "f_4"] = np.eye(number_of_points) * r_c ** 4.0

        partials[
            "beta",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter_to_chord_ratio",
        ] = (
            np.where(
                r_c == r_c_unclipped,
                f_1 + 2.0 * f_2 * r_c + 3.0 * f_3 * r_c ** 2.0 + 4.0 * f_4 * r_c ** 3.0,
                0.0,
            )
            / 2.0
        )
