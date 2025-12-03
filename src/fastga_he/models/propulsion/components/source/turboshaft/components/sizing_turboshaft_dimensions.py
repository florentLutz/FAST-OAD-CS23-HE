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
        self.add_input(
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_length",
            val=1.0,
            desc="K-factor to adjust the length of the turboshaft",
        )
        self.add_input(
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_diameter",
            val=1.0,
            desc="K-factor to adjust the diameter of the turboshaft",
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

        self.declare_partials(
            of="data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:length",
            wrt=[
                "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_length",
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            ],
            method="exact",
        )
        self.declare_partials(
            of=[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:width",
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:height",
            ],
            wrt=[
                "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_diameter",
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            ],
            method="exact",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        power_rating = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
        ]
        k_length = inputs[
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_length"
        ]
        k_diameter = inputs[
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_diameter"
        ]

        diameter = (2961.3 * power_rating**-0.272) * k_diameter
        length = (0.6119 * power_rating + 1314.9) * k_length

        outputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:height"] = (
            diameter
        )
        outputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:width"] = (
            diameter
        )
        outputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:length"] = (
            length
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        power_rating = inputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
        ]
        k_length = inputs[
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_length"
        ]
        k_diameter = inputs[
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_diameter"
        ]

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:height",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = (2961.3 * -0.272 * power_rating**-1.272) * k_diameter
        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:height",
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_diameter",
        ] = 2961.3 * power_rating**-0.272

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:width",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = (2961.3 * -0.272 * power_rating**-1.272) * k_diameter
        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:width",
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_diameter",
        ] = 2961.3 * power_rating**-0.272

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:length",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = 0.6119 * k_length
        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":engine:length",
            "settings:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":k_length",
        ] = 0.6119 * power_rating + 1314.9
