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

        self.add_input(name="alpha", val=np.nan, shape=number_of_points, units="rad")
        self.add_input(name="beta", val=np.nan, shape=number_of_points, desc="Height impact factor")
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
            desc="Increase in lift due to the slipstream effect behind the propeller expressed as "
            "a ratio of the clean lift",
        )
        self.add_output(
            "lift_increase_ratio_AOA_0",
            val=0.0,
            shape=number_of_points,
            desc="Increase in lift due to the slipstream effect behind the propeller expressed as "
            "a ratio of the clean lift, for a zero angle of attack",
        )

        self.declare_partials(
            of="lift_increase_ratio",
            wrt=["alpha", "beta", "axial_induction_factor_wing_ac"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="lift_increase_ratio",
            wrt="data:propulsion:he_power_train:propeller:" + propeller_id + ":installation_angle",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

        self.declare_partials(
            of="lift_increase_ratio_AOA_0",
            wrt="data:propulsion:he_power_train:propeller:" + propeller_id + ":installation_angle",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )
        self.declare_partials(
            of="lift_increase_ratio_AOA_0",
            wrt=["beta", "axial_induction_factor_wing_ac"],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        i_p = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":installation_angle"
        ]
        beta = inputs["beta"]
        a_p = inputs["axial_induction_factor_wing_ac"]

        # Applying the "is that a division by zero ? Actually it's divided by 0.5!" algorithm
        alpha_untreated = inputs["alpha"]
        sign_alpha = np.where(
            alpha_untreated != 0.0, np.sign(alpha_untreated), np.ones_like(alpha_untreated)
        )
        alpha = np.where(np.abs(alpha_untreated) < MIN_AOA, sign_alpha * MIN_AOA, alpha_untreated)

        lift_increase_ratio = (1.0 - beta * a_p * np.sin(i_p) / np.sin(alpha)) * np.sqrt(
            1.0 + 2.0 * beta * a_p * np.cos(alpha + i_p) + (a_p * beta) ** 2.0
        ) - 1.0

        # Since we can't really do zero, we'll do it at MIN_AOA
        alpha_0 = np.full_like(alpha, MIN_AOA)
        lift_increase_ratio_0 = (1.0 - beta * a_p * np.sin(i_p) / np.sin(alpha_0)) * np.sqrt(
            1.0 + 2.0 * beta * a_p * np.cos(alpha_0 + i_p) + (a_p * beta) ** 2.0
        ) - 1.0

        outputs["lift_increase_ratio"] = lift_increase_ratio
        outputs["lift_increase_ratio_AOA_0"] = lift_increase_ratio_0

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        i_p = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":installation_angle"
        ]
        beta = inputs["beta"]
        a_p = inputs["axial_induction_factor_wing_ac"]

        # Applying the "is that a division by zero ? Actually it's divided by 0.5!" algorithm
        alpha_untreated = inputs["alpha"]
        sign_alpha = np.where(
            alpha_untreated != 0.0, np.sign(alpha_untreated), np.ones_like(alpha_untreated)
        )
        alpha = np.where(np.abs(alpha_untreated) < MIN_AOA, sign_alpha * MIN_AOA, alpha_untreated)
        alpha_0 = np.full_like(alpha, MIN_AOA)

        inside_square = 1.0 + 2.0 * beta * a_p * np.cos(alpha + i_p) + (a_p * beta) ** 2.0
        inside_square_0 = 1.0 + 2.0 * beta * a_p * np.cos(alpha_0 + i_p) + (a_p * beta) ** 2.0

        d_inside_square_d_alpha = -2.0 * beta * a_p * np.sin(alpha + i_p)

        partials["lift_increase_ratio", "beta"] = partials_lift_increase_ratio_beta(
            alpha, beta, a_p, i_p, inside_square
        )
        partials["lift_increase_ratio_AOA_0", "beta"] = partials_lift_increase_ratio_beta(
            alpha_0, beta, a_p, i_p, inside_square_0
        )

        partials["lift_increase_ratio", "axial_induction_factor_wing_ac"] = (
            partials_lift_increase_ratio_a_p(alpha, beta, a_p, i_p, inside_square)
        )
        partials["lift_increase_ratio_AOA_0", "axial_induction_factor_wing_ac"] = (
            partials_lift_increase_ratio_a_p(alpha_0, beta, a_p, i_p, inside_square_0)
        )

        partials[
            "lift_increase_ratio",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":installation_angle",
        ] = partials_lift_increase_ratio_i_p(alpha, beta, a_p, i_p, inside_square)
        partials[
            "lift_increase_ratio_AOA_0",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":installation_angle",
        ] = partials_lift_increase_ratio_i_p(alpha_0, beta, a_p, i_p, inside_square_0)

        partials_alpha = (
            beta * a_p * np.sin(i_p) / np.sin(alpha) ** 2.0 * np.cos(alpha) * np.sqrt(inside_square)
            + 0.5
            * (1.0 - beta * a_p * np.sin(i_p) / np.sin(alpha))
            / np.sqrt(inside_square)
            * d_inside_square_d_alpha
        )
        partials["lift_increase_ratio", "alpha"] = np.where(
            alpha == alpha_untreated, partials_alpha, 1e-6
        )


def partials_lift_increase_ratio_beta(
    alpha: np.ndarray, beta: np.ndarray, a_p: np.ndarray, i_p: float, inside_square: np.ndarray
) -> np.ndarray:
    """
    Computes the partials derivative of the lift increase ratio with respect to the beta
    coefficient, is written in a function to simplify readability since only the AoA changes from
    one output to the other

    :param alpha: angle of attack at which the output is computed
    :param beta: height impact coefficient
    :param a_p: axial induction factor at the wing aerodynamic chord
    :param i_p: installation angle of the propeller
    :param inside_square: value of the coefficient inside the square root in the original formula
    """

    d_inside_square_d_beta = 2.0 * a_p * np.cos(alpha + i_p) + 2.0 * beta * a_p**2.0

    partials_ratio_beta = (
        -a_p * np.sin(i_p) / np.sin(alpha) * np.sqrt(inside_square)
        + 0.5
        * (1.0 - beta * a_p * np.sin(i_p) / np.sin(alpha))
        / np.sqrt(inside_square)
        * d_inside_square_d_beta
    )

    return partials_ratio_beta


def partials_lift_increase_ratio_a_p(
    alpha: np.ndarray, beta: np.ndarray, a_p: np.ndarray, i_p: float, inside_square: np.ndarray
) -> np.ndarray:
    """
    Computes the partials derivative of the lift increase ratio with respect to the axial
    induction factor coefficient, is written in a function to simplify readability since only the
    AoA changes from one output to the other

    :param alpha: angle of attack at which the output is computed
    :param beta: height impact coefficient
    :param a_p: axial induction factor at the wing aerodynamic chord
    :param i_p: installation angle of the propeller
    :param inside_square: value of the coefficient inside the square root in the original formula
    """

    d_inside_square_d_a_p = 2.0 * beta * np.cos(alpha + i_p) + 2.0 * a_p * beta**2.0

    partials_ratio_a_p = (
        -beta * np.sin(i_p) / np.sin(alpha) * np.sqrt(inside_square)
        + 0.5
        * (1.0 - beta * a_p * np.sin(i_p) / np.sin(alpha))
        / np.sqrt(inside_square)
        * d_inside_square_d_a_p
    )

    return partials_ratio_a_p


def partials_lift_increase_ratio_i_p(
    alpha: np.ndarray, beta: np.ndarray, a_p: np.ndarray, i_p: float, inside_square: np.ndarray
) -> np.ndarray:
    """
    Computes the partials derivative of the lift increase ratio with respect to the installation
    angle, is written in a function to simplify readability since only the
    AoA changes from one output to the other

    :param alpha: angle of attack at which the output is computed
    :param beta: height impact coefficient
    :param a_p: axial induction factor at the wing aerodynamic chord
    :param i_p: installation angle of the propeller
    :param inside_square: value of the coefficient inside the square root in the original formula
    """

    d_inside_square_d_i_p = -2.0 * beta * a_p * np.sin(alpha + i_p)

    partials_ratio_i_p = (
        -beta * a_p * np.cos(i_p) / np.sin(alpha) * np.sqrt(inside_square)
        + 0.5
        * (1.0 - beta * a_p * np.sin(i_p) / np.sin(alpha))
        / np.sqrt(inside_square)
        * d_inside_square_d_i_p
    )

    return partials_ratio_i_p
