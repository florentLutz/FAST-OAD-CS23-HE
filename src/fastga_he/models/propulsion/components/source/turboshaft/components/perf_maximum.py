# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesMaximum(om.ExplicitComponent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_power = None

    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare(
            name="turboshaft_id",
            default=None,
            desc="Identifier of the turboshaft",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        turboshaft_id = self.options["turboshaft_id"]

        self.add_input("shaft_power_out", units="kW", val=np.nan, shape=number_of_points)
        self.add_input(
            "equivalent_rated_power_opr_limit",
            units="kW",
            val=np.nan,
            shape=number_of_points,
            desc="Equivalent rated power of the turboshaft at the design point if the OPR was limiting",
        )
        self.add_input(
            "equivalent_rated_power_itt_limit",
            units="kW",
            val=np.nan,
            shape=number_of_points,
            desc="Equivalent rated power of the turboshaft at the design point if the ITT was limiting",
        )

        self.add_output(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
            units="kW",
            val=750.0,
            desc="Maximum power the turboshaft has to provide",
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        max_rated_power = inputs["shaft_power_out"]
        max_power_opr_limit = inputs["equivalent_rated_power_opr_limit"]
        max_power_itt_limit = inputs["equivalent_rated_power_itt_limit"]

        max_power_each_point = np.maximum(
            np.maximum(max_power_itt_limit, max_power_opr_limit), max_rated_power
        )
        self.max_power = np.max(max_power_each_point)

        outputs[
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max"
        ] = self.max_power

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        turboshaft_id = self.options["turboshaft_id"]

        max_rated_power = inputs["shaft_power_out"]
        max_power_opr_limit = inputs["equivalent_rated_power_opr_limit"]
        max_power_itt_limit = inputs["equivalent_rated_power_itt_limit"]

        if self.max_power in max_power_itt_limit:
            partial = np.where(
                max_power_itt_limit == self.max_power,
                np.ones_like(max_power_itt_limit),
                np.zeros_like(max_power_itt_limit),
            )
            partials[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
                "equivalent_rated_power_itt_limit",
            ] = partial
            partials[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
                "equivalent_rated_power_opr_limit",
            ] = np.zeros_like(max_power_itt_limit)
            partials[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
                "shaft_power_out",
            ] = np.zeros_like(max_power_itt_limit)
        elif self.max_power in max_power_opr_limit:
            partial = np.where(
                max_power_opr_limit == self.max_power,
                np.ones_like(max_power_opr_limit),
                np.zeros_like(max_power_opr_limit),
            )
            partials[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
                "equivalent_rated_power_itt_limit",
            ] = np.zeros_like(max_power_itt_limit)
            partials[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
                "equivalent_rated_power_opr_limit",
            ] = partial
            partials[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
                "shaft_power_out",
            ] = np.zeros_like(max_power_itt_limit)
        else:
            partial = np.where(
                max_rated_power == self.max_power,
                np.ones_like(max_rated_power),
                np.zeros_like(max_rated_power),
            )
            partials[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
                "equivalent_rated_power_itt_limit",
            ] = np.zeros_like(max_power_itt_limit)
            partials[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
                "equivalent_rated_power_opr_limit",
            ] = np.zeros_like(max_power_itt_limit)
            partials[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_max",
                "shaft_power_out",
            ] = partial
