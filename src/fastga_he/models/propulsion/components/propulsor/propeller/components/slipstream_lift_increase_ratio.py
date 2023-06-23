# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np

MIN_AOA = 0.5 * np.pi / 180.0  # 0.5 degree in rad


class SlipstreamPropellerLiftIncreaseRatio(om.ExplicitComponent):
    """
    Adaptation of the formula taken from :cite:`patterson:2016`. As highlighted in
    :cite:`de:2019`, there is a discontinuity at an AOA of 0.0, which we will solve using a
    variant of the ostrich algorithm which I'll call the "is that a division by zero ? Actually
    it's divided by 0.5!" algorithm
    """

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

        self.add_input(name="alpha", val=np.full(number_of_points, 5.0), units="rad")
        self.add_input(name="beta", val=1.0, shape=number_of_points, desc="Height impact factor")
        self.add_input(
            name="axial_induction_factor_wing_ac",
            val=np.nan,
            shape=number_of_points,
            desc="Value of the axial induction factor at the wing aerodynamic chord",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":installation_angle",
            units="rad",
            val=np.nan,
            desc="Diameter of the propeller as a ratio of the wing chord behind the propeller",
        )

        self.add_output(
            "lift_increase_ratio",
            val=0.0,
            shape=number_of_points,
            desc="Increase in lift due to the slipstream effect behind the propeller expressed a a ratio of the clean lift",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propeller_id = self.options["propeller_id"]

        i_p = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":installation_angle"
        ]
        beta = inputs["beta"]
        a_p = inputs["axial_induction_factor_wing_ac"]

        # Applying the "is that a division by zero ? Actually it's divided by 0.5!" algorithm
        alpha_untreated = inputs["alpha"]
        alpha = np.where(
            np.abs(alpha_untreated) < MIN_AOA, np.sign(alpha_untreated) * MIN_AOA, alpha_untreated
        )

        lift_increase_ratio = (1.0 - beta * a_p * np.sin(i_p) / np.sin(alpha)) * np.sqrt(
            1.0 + 2.0 * beta * a_p * np.cos(alpha + i_p) + (a_p * beta) ** 2.0
        ) - 1.0

        outputs["lift_increase_ratio"] = lift_increase_ratio

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        propeller_id = self.options["propeller_id"]

        i_p = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":installation_angle"
        ]
        beta = inputs["beta"]
        a_p = inputs["axial_induction_factor_wing_ac"]

        # Applying the "is that a division by zero ? Actually it's divided by 0.5!" algorithm
        alpha_untreated = inputs["alpha"]
        alpha = np.where(
            np.abs(alpha_untreated) < MIN_AOA, np.sign(alpha_untreated) * MIN_AOA, alpha_untreated
        )

        inside_square = 1.0 + 2.0 * beta * a_p * np.cos(alpha + i_p) + (a_p * beta) ** 2.0

        d_inside_square_d_beta = 2.0 * a_p * np.cos(alpha + i_p) + 2.0 * beta * a_p ** 2.0
        d_inside_square_d_a_p = 2.0 * beta * np.cos(alpha + i_p) + 2.0 * a_p * beta ** 2.0
        d_inside_square_d_alpha = -2.0 * beta * a_p * np.sin(alpha + i_p)
        d_inside_square_d_i_p = -2.0 * beta * a_p * np.sin(alpha + i_p)

        partials["lift_increase_ratio", "beta"] = np.diag(
            -a_p * np.sin(i_p) / np.sin(alpha) * np.sqrt(inside_square)
            + 0.5
            * (1.0 - beta * a_p * np.sin(i_p) / np.sin(alpha))
            / np.sqrt(inside_square)
            * d_inside_square_d_beta
        )
        partials["lift_increase_ratio", "axial_induction_factor_wing_ac"] = np.diag(
            -beta * np.sin(i_p) / np.sin(alpha) * np.sqrt(inside_square)
            + 0.5
            * (1.0 - beta * a_p * np.sin(i_p) / np.sin(alpha))
            / np.sqrt(inside_square)
            * d_inside_square_d_a_p
        )
        partials_alpha = (
            beta * a_p * np.sin(i_p) / np.sin(alpha) ** 2.0 * np.cos(alpha) * np.sqrt(inside_square)
            + 0.5
            * (1.0 - beta * a_p * np.sin(i_p) / np.sin(alpha))
            / np.sqrt(inside_square)
            * d_inside_square_d_alpha
        )
        partials["lift_increase_ratio", "alpha"] = np.diag(
            np.where(alpha == alpha_untreated, partials_alpha, 0.0)
        )
        partials[
            "lift_increase_ratio",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":installation_angle",
        ] = (
            -beta * a_p * np.cos(i_p) / np.sin(alpha) * np.sqrt(inside_square)
            + 0.5
            * (1.0 - beta * a_p * np.sin(i_p) / np.sin(alpha))
            / np.sqrt(inside_square)
            * d_inside_square_d_i_p
        )
