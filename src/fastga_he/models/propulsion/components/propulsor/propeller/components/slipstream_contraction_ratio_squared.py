# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamPropellerContractionRatioSquared(om.ExplicitComponent):
    """
    Adaptation of the formula taken from :cite:`de:2019` for the computation of the slipstream
    contraction ratio. Also, we will first compute the square value to make it simpler for the
    rest of the computation.
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

        self.add_input("axial_induction_factor", val=np.nan, shape=number_of_points)
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
            val=np.nan,
            desc="Distance between the propeller and the wing leading edge as a ratio of the "
            "reference chord behind the propeller",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
            val=np.nan,
            units="m",
            desc="Value of the wing chord behind the propeller",
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            val=np.nan,
            units="m",
            desc="Diameter of the propeller",
        )

        self.add_output(
            "contraction_ratio_squared",
            val=1,
            shape=number_of_points,
            desc="Square of the contraction ratio of the propeller slipstream evaluated at the wing"
            " AC",
        )

        self.declare_partials(
            of="contraction_ratio_squared",
            wrt="axial_induction_factor",
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.arange(number_of_points),
        )
        self.declare_partials(
            of="contraction_ratio_squared",
            wrt=[
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
                "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
            ],
            method="exact",
            rows=np.arange(number_of_points),
            cols=np.zeros(number_of_points),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propeller_id = self.options["propeller_id"]

        prop_rad = (
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"] / 2.0
        )
        prop_dist_from_le_ratio_chord = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio"
        ]
        ref_chord = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref"
        ]
        prop_dist_from_le_ratio = prop_dist_from_le_ratio_chord * ref_chord / prop_rad
        a_p = inputs["axial_induction_factor"]

        sigma_2 = (1.0 + a_p) / (
            1.0
            + a_p * (1.0 + prop_dist_from_le_ratio / np.sqrt(prop_dist_from_le_ratio**2.0 + 1.0))
        )

        outputs["contraction_ratio_squared"] = sigma_2

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propeller_id = self.options["propeller_id"]

        prop_rad = (
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"] / 2.0
        )
        prop_dist_from_le_ratio_chord = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio"
        ]
        ref_chord = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref"
        ]
        prop_dist_from_le_ratio = prop_dist_from_le_ratio_chord * ref_chord / prop_rad
        a_p = inputs["axial_induction_factor"]

        sigma_2 = (1.0 + a_p) / (
            1.0
            + a_p * (1.0 + prop_dist_from_le_ratio / np.sqrt(prop_dist_from_le_ratio**2.0 + 1.0))
        )

        d_sigma_2_d_ratio = (
            -sigma_2
            * a_p
            / (
                1.0
                + a_p
                * (1.0 + prop_dist_from_le_ratio / np.sqrt(prop_dist_from_le_ratio**2.0 + 1.0))
            )
            / (prop_dist_from_le_ratio**2.0 + 1.0) ** 1.5
        )

        partials[
            "contraction_ratio_squared",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
        ] = -d_sigma_2_d_ratio * prop_dist_from_le_ratio_chord * ref_chord / prop_rad**2.0 / 2.0
        partials[
            "contraction_ratio_squared",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":from_wing_LE_ratio",
        ] = d_sigma_2_d_ratio * ref_chord / prop_rad
        partials[
            "contraction_ratio_squared",
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":wing_chord_ref",
        ] = d_sigma_2_d_ratio * prop_dist_from_le_ratio_chord / prop_rad
        partials["contraction_ratio_squared", "axial_induction_factor"] = -(
            prop_dist_from_le_ratio
            / np.sqrt(prop_dist_from_le_ratio**2.0 + 1.0)
            / (
                1.0
                + a_p
                * (1.0 + prop_dist_from_le_ratio / np.sqrt(prop_dist_from_le_ratio**2.0 + 1.0))
            )
            ** 2.0
        )
