# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesBladesAngleOfAttack(om.ExplicitComponent):
    def initialize(self):

        self.options.declare(
            "number_of_points", default=1, desc="number of equilibrium to be treated"
        )
        self.options.declare("elements_number", default=7, types=int)
        self.options.declare(
            name="propeller_id",
            default=None,
            desc="Identifier of the propeller",
            allow_none=False,
        )

    def setup(self):

        number_of_points = self.options["number_of_points"]
        elements_number = self.options["elements_number"]
        propeller_id = self.options["propeller_id"]

        self.add_input(
            "tot_v_axial",
            val=np.full((elements_number, number_of_points), np.nan),
            units="m/s",
        )
        self.add_input(
            "tot_v_tangential",
            val=np.full((elements_number, number_of_points), np.nan),
            units="m/s",
        )
        self.add_input(
            "twist_75",
            val=np.nan,
            shape=number_of_points,
            units="rad",
        )
        self.add_input(
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_twist",
            val=np.nan,
            shape=elements_number,
            units="rad",
        )

        self.add_output(
            "blades_aoa",
            val=np.zeros((elements_number, number_of_points)),
            units="rad",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        elements_number = self.options["elements_number"]
        number_of_points = self.options["number_of_points"]
        propeller_id = self.options["propeller_id"]

        twist_ref_tile = np.tile(inputs["twist_75"], (elements_number, 1))
        twist_blade = np.tile(
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_twist"],
            (number_of_points, 1),
        ).transpose()
        theta = twist_ref_tile + twist_blade
        v_ax = inputs["tot_v_axial"]
        v_tan = inputs["tot_v_tangential"]

        alpha = theta - np.arctan2(v_ax, v_tan)

        outputs["blades_aoa"] = alpha
