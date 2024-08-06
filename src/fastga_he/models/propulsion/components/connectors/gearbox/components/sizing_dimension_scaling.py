# This file is part of FAST-OAD_CS23-HE : A framework for rapid Overall Aircraft Design of Hybrid
# Electric Aircraft.
# Copyright (C) 2022 ISAE-SUPAERO

import openmdao.api as om
import numpy as np


class SizingGearboxDimensionScaling(om.ExplicitComponent):
    """
    Computation of the scaling factor for the dimensions of a gearbox. Based on the
    formula from :cite:`budinger:2012`. Reference gearbox is single stage planetary gearbox
    presented here: https://www.ato.com/inline-planetary-speed-reducer-gearbox
    """

    def initialize(self):
        self.options.declare(
            name="gearbox_id",
            default=None,
            desc="Identifier of the gearbox",
            allow_none=False,
        )
        self.options.declare(
            "torque_out_ref",
            default=310.0,
            desc="Nominal output torque of the reference gearbox [N*m]",
        )

    def setup(self):
        gearbox_id = self.options["gearbox_id"]

        self.add_input(
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_rating",
            units="N*m",
            val=np.nan,
            desc="Max continuous output torque of the gearbox",
        )

        self.add_output(
            name="data:propulsion:he_power_train:gearbox:" + gearbox_id + ":scaling:dimensions",
            val=1.0,
        )

        self.declare_partials(of="*", wrt="*", method="exact")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        gearbox_id = self.options["gearbox_id"]

        torque_ref = self.options["torque_out_ref"]

        torque_rating = inputs[
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_rating"
        ]

        outputs["data:propulsion:he_power_train:gearbox:" + gearbox_id + ":scaling:dimensions"] = (
            torque_rating / torque_ref
        ) ** (1.0 / 3.0)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gearbox_id = self.options["gearbox_id"]
        torque_ref = self.options["torque_out_ref"]

        torque_rating = inputs[
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_rating"
        ]

        partials[
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":scaling:dimensions",
            "data:propulsion:he_power_train:gearbox:" + gearbox_id + ":torque_out_rating",
        ] = 1.0 / 3.0 * torque_rating ** (-2.0 / 3.0) / torque_ref ** (1.0 / 3.0)
