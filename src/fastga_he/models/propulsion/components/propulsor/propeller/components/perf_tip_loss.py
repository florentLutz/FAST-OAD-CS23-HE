# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class PerformancesTipLoss(om.ExplicitComponent):
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
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":blades_number",
            val=np.nan,
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius",
            shape=elements_number,
            units="m",
            val=np.full(elements_number, np.nan),
        )
        self.add_input(
            name="data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter",
            units="m",
            val=np.nan,
        )
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

        self.add_output(
            "tip_loss",
            val=np.ones((elements_number, number_of_points)),
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        number_of_points = self.options["number_of_points"]
        propeller_id = self.options["propeller_id"]

        blades_number = inputs[
            "data:propulsion:he_power_train:propeller:" + propeller_id + ":blades_number"
        ]

        radius_max = (
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":diameter"] / 2.0
        )
        r_tile = np.tile(
            inputs["data:propulsion:he_power_train:propeller:" + propeller_id + ":elements_radius"],
            (number_of_points, 1),
        ).transpose()

        v_ax = inputs["tot_v_axial"]
        v_tan = inputs["tot_v_tangential"]

        f_tip_tile = (
            2.0
            / np.pi
            * np.arccos(
                np.exp(
                    -blades_number
                    / 2.0
                    * (radius_max - r_tile)
                    / r_tile
                    * np.sqrt(1.0 + (v_tan / v_ax) ** 2.0)
                )
            )
        )

        outputs["tip_loss"] = f_tip_tile
