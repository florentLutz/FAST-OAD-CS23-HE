# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingPlanetaryGearWeight(om.ExplicitComponent):
    """
    Computation of the weight of a gearbox. Based on the formula from :cite:`anderson:2018` with
    K_gb estimated based on the results displayed.
    """

    def initialize(self):
        self.options.declare(
            name="planetary_gear_id",
            default=None,
            desc="Identifier of the planetary gear",
            allow_none=False,
        )

    def setup(self):

        planetary_gear_id = self.options["planetary_gear_id"]

        self.add_input(
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_rating",
            units="N*m",
            val=np.nan,
            desc="Max continuous output torque of the gearbox",
        )
        self.add_input(
            name="data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":gear_ratio",
            val=np.nan,
            desc="Gear ratio of the planetary gear",
        )
        self.add_input(
            name="data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":tech_level_constant",
            val=6.157,
            desc="Technology level constant for the gearbox, default value computed with the "
            "results from the source paper",
        )

        self.add_output(
            name="data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass",
            val=40.0,
            units="lbm",
            desc="Mass of the planetary gear",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        planetary_gear_id = self.options["planetary_gear_id"]

        gear_ratio = inputs[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":gear_ratio"
        ]
        torque_rating = inputs[
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_rating"
        ]
        k_gb = inputs[
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":tech_level_constant"
        ]

        mass = k_gb * (torque_rating / 745.7) ** 0.76 * gear_ratio ** 0.13

        outputs[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass"
        ] = mass

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        planetary_gear_id = self.options["planetary_gear_id"]

        gear_ratio = inputs[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":gear_ratio"
        ]
        torque_rating = inputs[
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_rating"
        ]
        k_gb = inputs[
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":tech_level_constant"
        ]

        partials[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass",
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":gear_ratio",
        ] = (
            0.13 * k_gb * (torque_rating / 745.7) ** 0.76 * gear_ratio ** -0.87
        )
        partials[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass",
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":torque_out_rating",
        ] = (
            0.76 * k_gb * torque_rating ** -0.24 * gear_ratio ** 0.13 / 745.7 ** 0.76
        )
        partials[
            "data:propulsion:he_power_train:planetary_gear:" + planetary_gear_id + ":mass",
            "data:propulsion:he_power_train:planetary_gear:"
            + planetary_gear_id
            + ":tech_level_constant",
        ] = (torque_rating / 745.7) ** 0.76 * gear_ratio ** 0.13
