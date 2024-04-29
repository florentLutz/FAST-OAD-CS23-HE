# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SlipstreamRequiredPower(om.ExplicitComponent):
    """
    In addition to the power required on the shaft we might also want to add a mechanical
    offtake straight on the shaft. This component simply adds that possibility for users but the
    default will be 0.0. More advanced way to define this offtake (like proper physical components)
    might come in the future.
    """

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

        self.add_input("shaft_power_out", units="kW", val=np.nan, shape=number_of_points + 2)
        self.add_input(
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_offtake",
            val=0.0,
            units="kW",
            desc="Mechanical offtake on the turboshaft, is added to shaft power out",
        )

        self.add_output("power_required", val=500.0, units="kW", shape=number_of_points)

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]
        turboshaft_id = self.options["turboshaft_id"]

        # Because this variable will need to be connected to the performances computation,
        # it might be bigger than the number of point, in which case, we need to cut it down to
        # size by removing the first and last points which represents the taxi phases
        untreated_shaft_power = inputs["shaft_power_out"]
        shaft_power = untreated_shaft_power[1:-1]

        outputs["power_required"] = (
            shaft_power
            + inputs[
                "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_offtake"
            ]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        number_of_points = self.options["number_of_points"]
        turboshaft_id = self.options["turboshaft_id"]

        partial = np.zeros((number_of_points, number_of_points + 2))
        partial[:, 1 : number_of_points + 1] = np.eye(number_of_points)
        partials["power_required", "shaft_power_out"] = partial

        partials[
            "power_required",
            "data:propulsion:he_power_train:turboshaft:" + turboshaft_id + ":power_offtake",
        ] = np.ones(number_of_points)
