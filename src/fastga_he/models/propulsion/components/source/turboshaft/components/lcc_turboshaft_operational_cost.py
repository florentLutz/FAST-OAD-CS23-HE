# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2025 ISAE-SUPAERO

import numpy as np
import openmdao.api as om


class LCCTurboshaftOperationalCost(om.ExplicitComponent):
    """
    Computation of the turboshaft engine annual operational cost. The time between overhaul (TBO) is
    based on the PT6s engines data provided by
    https://www.aopa.org/news-and-media/all-news/2020/february/24/turboprop-engine-repairs-for-less.
    The Overhaul cost rate is obtained from
    https://standardaero.com/engines/prattwhitneycanada/pt6a/pt6aflatrateoverhaulprogram/.
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
            name="data:TLAR:flight_hours_per_year",
            val=283.2,
            units="h",
            desc="Expected number of hours flown per year",
        )
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
            units="kW",
            val=np.nan,
        )

        self.add_output(
            name="data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":operational_cost",
            units="USD/yr",
            val=1e4,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        outputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":operational_cost"
        ] = (
            inputs["data:TLAR:flight_hours_per_year"]
            / 3.5
            * (
                0.202
                * inputs[
                    "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"
                ]
                + 259.0
            )
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        turboshaft_id = self.options["turboshaft_id"]

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":operational_cost",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating",
        ] = 0.202 * inputs["data:TLAR:flight_hours_per_year"] / 3.5

        partials[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":operational_cost",
            "data:TLAR:flight_hours_per_year",
        ] = (
            0.202
            * inputs["data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_rating"]
            + 259.0
        ) / 3.5
