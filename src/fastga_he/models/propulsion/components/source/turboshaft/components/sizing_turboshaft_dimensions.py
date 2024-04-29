# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingTurboshaftDimensions(om.ExplicitComponent):
    """
    Computation of the dimensions of the turboshaft. Based on a regression established based on
    the data from the PT6A family, can be seen in ..methodology.data_pt6_family.xlsx
    """

    def initialize(self):

        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):

        turboshaft_id = self.options["turboshaft_id"]

        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            units="kW",
            val=np.nan,
            desc="Flat rating of the turboshaft",
        )

        self.add_output(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:length",
            val=1.777,
            desc="Length of the turboshaft",
            units="mm",
        )
        self.add_output(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:width",
            val=0.4661,
            desc="Width of the turboshaft",
            units="mm",
        )
        self.add_output(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:height",
            val=0.4661,
            desc="Height of the turboshaft",
            units="mm",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        power_rating = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
        ]

        diameter = 2961.3 * power_rating ** -0.272
        length = 0.6119 * power_rating + 1314.9

        outputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:height"
        ] = diameter
        outputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:width"
        ] = diameter
        outputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:length"
        ] = length

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        power_rating = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
        ]

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:height",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = (
            2961.3 * -0.272 * power_rating ** -1.272
        )
        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:width",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = (
            2961.3 * -0.272 * power_rating ** -1.272
        )
        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:length",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = 0.6119
